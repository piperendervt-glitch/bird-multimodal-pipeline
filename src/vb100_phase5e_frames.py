"""VB100 Phase 5e 段階A: 動画からキーフレームを抽出し YOLO で鳥を検出する。

Phase 4b (bird_phase4b_frames.py) のロジックを VB100 のディレクトリ構造
(data/vb100/vb100_video/{species_name}/*.mp4) に合わせて移植した版。

- 1 fps で動画をフレームサンプリング
- YOLOv8n (COCO bird=14) で検出、padding_ratio=0.3 で切り抜き
- 鳥未検出フレームは元フレーム全体をフォールバック保存
- 切り抜き画像と検出メタ情報を results/vb100_phase5e/ に保存

このスクリプトは DINOv2 をインポートしない。
"""

import json
import os
import sys
import io
import time

import cv2
from ultralytics import YOLO

# Windows cp932 対策
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


DATA_DIR = os.path.join("..", "data", "vb100")
VIDEO_DIR = os.path.join(DATA_DIR, "vb100_video")
OUT_DIR = os.path.join("..", "results", "vb100_phase5e")
CROP_ROOT = os.path.join(OUT_DIR, "crops")

# COCO の bird クラス ID
BIRD_CLASS_ID = 14
FPS_TARGET = 1
PADDING_RATIO = 0.3
YOLO_CONF_THRESHOLD = 0.1
JPEG_QUALITY = 90


def extract_keyframes(video_path, fps_target=FPS_TARGET):
    """動画からキーフレームを抽出（fps_target フレーム/秒）。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], 0.0, 0.0

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0.0

    frame_interval = max(1, int(round(video_fps / fps_target))) if video_fps > 0 else 1

    keyframes = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / video_fps if video_fps > 0 else 0.0
            keyframes.append({
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "image": frame,
            })
        frame_idx += 1

    cap.release()
    return keyframes, video_fps, duration


def detect_birds(model, image, conf_threshold=YOLO_CONF_THRESHOLD):
    """1 枚の画像を YOLO で推論し、bird クラスの検出結果のみ返す。"""
    result = model(image, verbose=False, conf=conf_threshold)[0]
    bird_dets = []
    for box in result.boxes:
        if int(box.cls[0]) != BIRD_CLASS_ID:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        bird_dets.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "confidence": conf,
        })
    return bird_dets


def crop_bird(image, detection, padding_ratio=PADDING_RATIO):
    """検出結果の bbox を padding 付きで切り抜く。"""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = (
        detection["x1"], detection["y1"], detection["x2"], detection["y2"]
    )
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad_x = bw * padding_ratio
    pad_y = bh * padding_ratio
    x1 = max(0.0, x1 - pad_x)
    y1 = max(0.0, y1 - pad_y)
    x2 = min(float(w), x2 + pad_x)
    y2 = min(float(h), y2 + pad_y)
    x1_i, y1_i = int(x1), int(y1)
    x2_i, y2_i = max(x1_i + 1, int(x2)), max(y1_i + 1, int(y2))
    return image[y1_i:y2_i, x1_i:x2_i]


def discover_videos(video_dir):
    """VB100 のディレクトリ構造から動画リストを収集する。
    戻り値: [{key, species, filename, full_path}, ...]
    key は "species/filename.mp4" 形式で一意。
    """
    items = []
    for species in sorted(os.listdir(video_dir)):
        sp_dir = os.path.join(video_dir, species)
        if not os.path.isdir(sp_dir):
            continue
        for fname in sorted(os.listdir(sp_dir)):
            if not fname.lower().endswith(".mp4"):
                continue
            items.append({
                "key": f"{species}/{fname}",
                "species": species,
                "filename": fname,
                "full_path": os.path.join(sp_dir, fname),
            })
    return items


def main():
    print("=== VB100 Phase 5e 段階A: フレーム抽出 + YOLO 検出 ===")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CROP_ROOT, exist_ok=True)

    videos = discover_videos(VIDEO_DIR)
    print(f"対象動画数: {len(videos)} 本")

    # 種ごとの動画数
    species_set = sorted({v["species"] for v in videos})
    print(f"種数: {len(species_set)}")
    species_counts = {}
    for v in videos:
        species_counts[v["species"]] = species_counts.get(v["species"], 0) + 1
    min_sp = min(species_counts.items(), key=lambda x: x[1])
    max_sp = max(species_counts.items(), key=lambda x: x[1])
    print(f"種ごと動画数: min={min_sp[1]} ({min_sp[0]}), "
          f"max={max_sp[1]} ({max_sp[0]})")

    # YOLO 読み込み
    print("\nYOLOv8n 読み込み中...")
    yolo = YOLO("yolov8n.pt")

    all_results = {}
    total_frames = 0
    detected_frames = 0

    start_total = time.time()

    for i, vinfo in enumerate(videos):
        key = vinfo["key"]
        species = vinfo["species"]
        video_path = vinfo["full_path"]
        stem = vinfo["filename"][:-4]  # 拡張子除去

        keyframes, video_fps, duration = extract_keyframes(video_path,
                                                           fps_target=FPS_TARGET)
        if not keyframes:
            print(f"  警告: {key} からフレームを抽出できませんでした")
            all_results[key] = {
                "species": species,
                "video_file": vinfo["filename"],
                "duration": 0.0,
                "fps": 0.0,
                "n_keyframes": 0,
                "n_detected": 0,
                "detection_rate": 0.0,
                "crops": [],
            }
            continue

        # 種ごと/動画ごとのディレクトリ
        crop_dir = os.path.join(CROP_ROOT, species, stem)
        os.makedirs(crop_dir, exist_ok=True)

        crop_info = []
        frames_with_birds = 0

        for kf in keyframes:
            dets = detect_birds(yolo, kf["image"], conf_threshold=YOLO_CONF_THRESHOLD)
            if dets:
                frames_with_birds += 1
                best = max(dets, key=lambda d: d["confidence"])
                crop = crop_bird(kf["image"], best, padding_ratio=PADDING_RATIO)
                if crop.size == 0:
                    # 念のため空切り抜きは元フレームへフォールバック
                    fallback_path = os.path.join(
                        crop_dir, f"frame_{kf['frame_idx']:06d}_full.jpg"
                    )
                    cv2.imwrite(fallback_path, kf["image"],
                                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                    crop_info.append({
                        "frame_idx": kf["frame_idx"],
                        "timestamp": kf["timestamp"],
                        "crop_path": fallback_path.replace("\\", "/"),
                        "confidence": 0.0,
                        "is_fallback": True,
                    })
                else:
                    crop_path = os.path.join(
                        crop_dir, f"frame_{kf['frame_idx']:06d}.jpg"
                    )
                    cv2.imwrite(crop_path, crop,
                                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                    crop_info.append({
                        "frame_idx": kf["frame_idx"],
                        "timestamp": kf["timestamp"],
                        "crop_path": crop_path.replace("\\", "/"),
                        "confidence": best["confidence"],
                        "bbox": [best["x1"], best["y1"], best["x2"], best["y2"]],
                        "is_fallback": False,
                    })
            else:
                # 鳥未検出 → 元フレームをフォールバック保存
                fallback_path = os.path.join(
                    crop_dir, f"frame_{kf['frame_idx']:06d}_full.jpg"
                )
                cv2.imwrite(fallback_path, kf["image"],
                            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                crop_info.append({
                    "frame_idx": kf["frame_idx"],
                    "timestamp": kf["timestamp"],
                    "crop_path": fallback_path.replace("\\", "/"),
                    "confidence": 0.0,
                    "is_fallback": True,
                })

        total_frames += len(keyframes)
        detected_frames += frames_with_birds

        all_results[key] = {
            "species": species,
            "video_file": vinfo["filename"],
            "duration": duration,
            "fps": video_fps,
            "n_keyframes": len(keyframes),
            "n_detected": frames_with_birds,
            "detection_rate": frames_with_birds / len(keyframes),
            "crops": crop_info,
        }

        if (i + 1) % 50 == 0 or (i + 1) == len(videos):
            elapsed = time.time() - start_total
            det_rate = detected_frames / max(1, total_frames) * 100
            rate = (i + 1) / max(1e-9, elapsed)
            eta = (len(videos) - (i + 1)) / max(1e-9, rate)
            print(f"  {i+1}/{len(videos)} 処理完了 "
                  f"({elapsed:.0f}秒, 検出率: {det_rate:.1f}%, "
                  f"速度: {rate:.2f} 動画/秒, 残り約 {eta:.0f}秒)")

    elapsed_total = time.time() - start_total

    print(f"\n=== フレーム抽出 + YOLO 検出 サマリー ===")
    print(f"処理動画: {len(all_results)} 本")
    print(f"処理時間: {elapsed_total:.1f} 秒 "
          f"({len(all_results)/max(1e-9, elapsed_total):.2f} 動画/秒)")
    print(f"総キーフレーム: {total_frames}")
    detection_rate = detected_frames / max(1, total_frames)
    print(f"鳥検出フレーム: {detected_frames} ({detection_rate*100:.1f}%)")

    out_json = os.path.join(OUT_DIR, "frame_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "n_videos": len(all_results),
                "n_species": len(species_set),
                "n_keyframes_total": total_frames,
                "n_detected_total": detected_frames,
                "detection_rate": detection_rate,
                "elapsed_sec": elapsed_total,
                "videos_per_sec": len(all_results) / max(1e-9, elapsed_total),
            },
            "videos": all_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"保存: {out_json}")

    print(f"\n--- 失敗条件の判定 ---")
    if detection_rate < 0.50:
        print(f"  NG: 検出率 {detection_rate*100:.1f}% < 50%（動画中の鳥検出が不十分）")
    else:
        print(f"  OK: 検出率 {detection_rate*100:.1f}% >= 50%")


if __name__ == "__main__":
    main()
