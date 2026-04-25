"""Phase 5g 統合 段階A: YouTube Great Tit 動画からキーフレームを抽出し YOLO で鳥検出する。

- 1 fps で動画をフレームサンプリング（labels.csv の start_sec/end_sec で範囲を絞る）
- YOLOv8n (COCO bird=14) で検出、padding_ratio=0.3 で切り抜き
- 鳥未検出フレームは元フレーム全体をフォールバック保存
- 切り抜き画像と検出メタ情報を results/phase5g_youtube/ に保存

このスクリプトは BirdNET や DINOv2 をインポートしない。
"""

import csv
import json
import os
import time

import cv2
from ultralytics import YOLO


DATA_DIR = os.path.join("..", "data", "youtube_greattit")
VIDEO_DIR = os.path.join(DATA_DIR, "videos")
OUT_DIR = os.path.join("..", "results", "phase5g_youtube")
CROP_ROOT = os.path.join(OUT_DIR, "crops")

# COCO の bird クラス ID
BIRD_CLASS_ID = 14


def extract_keyframes(video_path, fps_target=1):
    """動画からキーフレームを抽出する（fps_target フレーム/秒）。"""
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


def detect_birds(model, image, conf_threshold=0.1):
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


def crop_bird(image, detection, padding_ratio=0.3):
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


def parse_float(v, default=0.0):
    """CSV の数値文字列をパース（空文字列は default）。"""
    if v is None:
        return default
    s = str(v).strip()
    if not s:
        return default
    try:
        return float(s)
    except ValueError:
        return default


def main():
    print("=== Phase 5g 統合 段階A: フレーム抽出 + YOLO ===")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CROP_ROOT, exist_ok=True)

    # ラベル読み込み
    labels = {}
    with open(os.path.join(DATA_DIR, "labels.csv"), encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["video_id"]] = row
    print(f"ラベル付き動画: {len(labels)} 本")

    # manifest 読み込み
    with open(os.path.join(DATA_DIR, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    print(f"manifest: {len(manifest)} 本")

    # YOLO モデルのロード
    print("\nYOLOv8n 読み込み中...")
    yolo = YOLO("yolov8n.pt")

    all_results = {}
    total_frames = 0
    detected_frames = 0

    start_total = time.time()

    for i, entry in enumerate(manifest):
        video_id = entry["video_id"]
        filename = entry.get("filename", f"{video_id}.mp4")
        video_path = os.path.join(VIDEO_DIR, filename)

        if not os.path.exists(video_path):
            print(f"  警告: {video_path} が見つかりません")
            continue

        label = labels.get(video_id, {})
        start_sec = parse_float(label.get("start_sec"), 0.0)
        end_sec = parse_float(label.get("end_sec"), 0.0)

        # キーフレーム抽出
        keyframes, video_fps, duration = extract_keyframes(video_path, fps_target=1)

        if not keyframes:
            print(f"  警告: {filename} からフレームを抽出できません")
            continue

        # start_sec〜end_sec の範囲内のフレームだけ使用
        if end_sec > 0:
            keyframes = [kf for kf in keyframes
                         if start_sec <= kf["timestamp"] <= end_sec]

        # 切り抜きディレクトリ
        crop_dir = os.path.join(CROP_ROOT, video_id)
        os.makedirs(crop_dir, exist_ok=True)

        frame_detections = []

        for kf in keyframes:
            dets = detect_birds(yolo, kf["image"], conf_threshold=0.1)

            if dets:
                best = max(dets, key=lambda d: d["confidence"])
                crop = crop_bird(kf["image"], best, padding_ratio=0.3)
                if crop.size == 0:
                    # 切り抜きが空 → フォールバック
                    fallback_path = os.path.join(
                        crop_dir, f"frame_{kf['frame_idx']:06d}_full.jpg"
                    )
                    cv2.imwrite(fallback_path, kf["image"],
                                [cv2.IMWRITE_JPEG_QUALITY, 90])
                    frame_detections.append({
                        "frame_idx": kf["frame_idx"],
                        "timestamp": kf["timestamp"],
                        "crop_path": fallback_path.replace("\\", "/"),
                        "confidence": 0.0,
                        "n_birds": 0,
                        "is_fallback": True,
                    })
                else:
                    crop_path = os.path.join(
                        crop_dir, f"frame_{kf['frame_idx']:06d}.jpg"
                    )
                    cv2.imwrite(crop_path, crop,
                                [cv2.IMWRITE_JPEG_QUALITY, 90])
                    frame_detections.append({
                        "frame_idx": kf["frame_idx"],
                        "timestamp": kf["timestamp"],
                        "crop_path": crop_path.replace("\\", "/"),
                        "confidence": best["confidence"],
                        "n_birds": len(dets),
                        "bbox": [best["x1"], best["y1"], best["x2"], best["y2"]],
                        "is_fallback": False,
                    })
            else:
                # 鳥未検出 → 元フレームをフォールバック保存
                fallback_path = os.path.join(
                    crop_dir, f"frame_{kf['frame_idx']:06d}_full.jpg"
                )
                cv2.imwrite(fallback_path, kf["image"],
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
                frame_detections.append({
                    "frame_idx": kf["frame_idx"],
                    "timestamp": kf["timestamp"],
                    "crop_path": fallback_path.replace("\\", "/"),
                    "confidence": 0.0,
                    "n_birds": 0,
                    "is_fallback": True,
                })

            total_frames += 1
            if dets:
                detected_frames += 1

        n_detected = sum(1 for d in frame_detections if d.get("n_birds", 0) > 0)

        all_results[video_id] = {
            "filename": filename,
            "category": label.get("category", ""),
            "singing_matches_video": label.get("singing_matches_video", ""),
            "duration": duration,
            "fps": video_fps,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "n_keyframes_total": len(keyframes),
            "n_detected": n_detected,
            "detection_rate": n_detected / max(len(frame_detections), 1),
            "crops": frame_detections,
        }

        det_rate = all_results[video_id]["detection_rate"]
        print(f"  {i+1}. {video_id}: {len(keyframes)}フレーム, "
              f"検出率 {det_rate*100:.0f}%, "
              f"ラベル: {label.get('singing_matches_video', '?')}")

    elapsed = time.time() - start_total

    print(f"\n=== サマリー ===")
    print(f"処理動画: {len(all_results)} 本")
    print(f"総フレーム: {total_frames}, 鳥検出: {detected_frames} "
          f"({detected_frames/max(total_frames,1)*100:.1f}%)")
    print(f"処理時間: {elapsed:.1f} 秒")

    out_path = os.path.join(OUT_DIR, "frame_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "n_videos": len(all_results),
                "n_keyframes_total": total_frames,
                "n_detected_total": detected_frames,
                "detection_rate": detected_frames / max(total_frames, 1),
                "elapsed_sec": elapsed,
            },
            "videos": all_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"保存: {out_path}")


if __name__ == "__main__":
    main()
