"""Phase 4b 段階B: 動画から音声を分離し、3秒窓で BirdNET 処理する。

- ffmpeg で動画 → 48kHz mono WAV に変換
- BirdNET v2.4 を 3秒窓 / 1秒ホップ (overlap_duration_s=2.0) で実行
- 各窓について top_k 確信度 + 5統計量の 15次元特徴を生成
- 動画ごとに窓列と統計を results/bird_phase4b/audio_results.json に保存

このスクリプトは YOLO や DINOv2 をインポートしない。
"""

import json
import os
import subprocess
import time
from collections import defaultdict

import numpy as np


DATA_DIR = os.path.join("..", "data", "wetlandbirds")
VIDEO_DIR = os.path.join(DATA_DIR, "videos")
OUT_DIR = os.path.join("..", "results", "bird_phase4b")
AUDIO_DIR = os.path.join(OUT_DIR, "audio")

TOP_K = 10
FEAT_DIM = TOP_K + 5  # 10 確信度 + max/mean/std/count(>0.5)/margin


def check_ffmpeg():
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=10)
        return r.returncode == 0
    except FileNotFoundError:
        return False


def extract_audio(video_path, output_wav):
    """ffmpeg で動画から音声を 48kHz mono の WAV に変換する。"""
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn",  # 映像なし
        "-acodec", "pcm_s16le",
        "-ar", "48000",  # 48kHz（BirdNET 推奨）
        "-ac", "1",  # モノラル
        "-loglevel", "error",
        output_wav,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    return r.returncode == 0, r.stderr


def build_feature_vector(confidences, top_k=TOP_K):
    """上位確信度のリストから固定長ベクトルを生成 (Phase 3 と同一規約)。"""
    confs = sorted([float(c) for c in confidences], reverse=True)[:top_k]
    while len(confs) < top_k:
        confs.append(0.0)
    arr = np.array(confs, dtype=np.float32)
    feats = list(arr)
    feats.append(float(arr.max()))
    feats.append(float(arr.mean()))
    feats.append(float(arr.std()))
    feats.append(float((arr > 0.5).sum()))
    feats.append(float(arr[0] - arr[1]) if len(arr) >= 2 else 0.0)
    return np.array(feats, dtype=np.float32)


def predict_windows(model, audio_path):
    """1 つの音声ファイルに対し 3秒窓 / 1秒ホップで予測し、窓ごとに
    (start_time, end_time, confidences) のリストを返す。"""
    try:
        result = model.predict(
            audio_path,
            top_k=TOP_K,
            apply_sigmoid=True,
            default_confidence_threshold=0.0,
            overlap_duration_s=2.0,  # 3s 窓 - 2s 重なり = 1s ホップ
            show_stats=None,
        )
    except Exception as e:
        print(f"    予測失敗: {os.path.basename(audio_path)}: {e}")
        return []

    try:
        arr = result.to_structured_array()
    except Exception as e:
        print(f"    結果変換失敗: {e}")
        return []

    # 構造化配列は窓ごとに top_k 行を含む。(start,end) でグループ化。
    groups = defaultdict(list)
    names = arr.dtype.names or ()
    if "start_time" in names and "end_time" in names and "confidence" in names:
        for row in arr:
            s = float(row["start_time"])
            e = float(row["end_time"])
            c = float(row["confidence"])
            groups[(s, e)].append(c)
    else:
        return []

    windows = []
    for (s, e), confs in sorted(groups.items(), key=lambda kv: kv[0]):
        windows.append({"start": s, "end": e, "confidences": confs})
    return windows


def main():
    print("=== Phase 4b-B: 音声分離 + BirdNET 処理 ===")

    if not check_ffmpeg():
        print("エラー: ffmpeg が見つかりません")
        print("  Windows: https://ffmpeg.org/download.html からダウンロードし PATH に追加")
        return
    print("ffmpeg: OK")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)

    # 段階Aの結果を読み込み（動画リスト・種名マッピング用）
    frame_results_path = os.path.join(OUT_DIR, "frame_results.json")
    if not os.path.exists(frame_results_path):
        print(f"エラー: {frame_results_path} が見つかりません。段階Aを先に実行してください。")
        return
    with open(frame_results_path, encoding="utf-8") as f:
        frame_data = json.load(f)
    frame_results = frame_data["videos"]
    print(f"対象動画: {len(frame_results)} 本")

    # BirdNET モデルのロード
    print("\nBirdNET モデルをロード中（初回はモデル重みのダウンロードあり）...")
    from birdnet import load
    model = load("acoustic", "2.4", "tf")
    print("  モデルロード完了")

    out_path = os.path.join(OUT_DIR, "audio_results.json")

    # 途中保存ファイルから既存の BirdNET 結果を読み込む（ある場合のみ）
    all_audio_results = {}
    if os.path.exists(out_path):
        try:
            with open(out_path, encoding="utf-8") as f:
                prev = json.load(f)
            prev_videos = prev.get("videos", {})
            # 窓が 0 でなく error でないものだけを再利用（失敗再試行のため）
            for k, v in prev_videos.items():
                if v.get("n_windows", 0) > 0 or v.get("error"):
                    all_audio_results[k] = v
            print(f"途中保存から {len(all_audio_results)} 件を再利用")
        except Exception as e:
            print(f"既存 JSON 読み込み失敗: {e} → 新規で処理します")

    start_total = time.time()
    total = len(frame_results)

    def save_progress(final=False):
        mean_confs_p = [v.get("mean_max_confidence", 0.0)
                        for v in all_audio_results.values()]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "summary": {
                    "n_videos": len(all_audio_results),
                    "mean_max_confidence": float(np.mean(mean_confs_p)) if mean_confs_p else 0.0,
                    "n_videos_gt_05": sum(1 for c in mean_confs_p if c > 0.5),
                    "total_windows": sum(v.get("n_windows", 0) for v in all_audio_results.values()),
                    "feat_dim": FEAT_DIM,
                    "window_sec": 3.0,
                    "hop_sec": 1.0,
                    "final": final,
                },
                "videos": all_audio_results,
            }, f, indent=2, ensure_ascii=False)

    for i, (video_name, vinfo) in enumerate(frame_results.items()):
        if video_name in all_audio_results:
            continue  # 既に処理済みはスキップ

        video_path = os.path.join(VIDEO_DIR, video_name)
        wav_path = os.path.join(AUDIO_DIR, video_name.replace(".mp4", ".wav"))

        # 音声抽出（既存はスキップ）
        if not (os.path.exists(wav_path) and os.path.getsize(wav_path) > 0):
            ok, err = extract_audio(video_path, wav_path)
            if not ok:
                print(f"  警告: {video_name} の音声抽出失敗: {err[:200]}")
                all_audio_results[video_name] = {
                    "species": vinfo["species"],
                    "n_windows": 0,
                    "windows": [],
                    "mean_max_confidence": 0.0,
                    "error": "audio_extract_failed",
                }
                continue

        # BirdNET 予測（3秒窓 / 1秒ホップ）
        windows_raw = predict_windows(model, wav_path)

        windows = []
        for w in windows_raw:
            feats = build_feature_vector(w["confidences"], TOP_K)
            windows.append({
                "start": w["start"],
                "end": w["end"],
                "features": feats.tolist(),
                "max_confidence": float(feats[0]) if len(feats) > 0 else 0.0,
            })

        max_confs = [w["max_confidence"] for w in windows]
        mean_max = float(np.mean(max_confs)) if max_confs else 0.0

        all_audio_results[video_name] = {
            "species": vinfo["species"],
            "n_windows": len(windows),
            "windows": windows,
            "mean_max_confidence": mean_max,
        }

        if (i + 1) % 10 == 0 or (i + 1) == total:
            elapsed = time.time() - start_total
            save_progress(final=False)
            print(f"  {i+1}/{total} 処理完了 ({elapsed:.0f}秒, 途中保存済)",
                  flush=True)

    save_progress(final=True)

    elapsed_total = time.time() - start_total

    print(f"\n=== 音声処理サマリー ===")
    print(f"処理動画: {len(all_audio_results)} 本")
    print(f"処理時間: {elapsed_total:.1f} 秒")

    mean_confs = [v["mean_max_confidence"] for v in all_audio_results.values()]
    mean_all = float(np.mean(mean_confs)) if mean_confs else 0.0
    n_gt_05 = sum(1 for c in mean_confs if c > 0.5)
    print(f"全動画平均 BirdNET 最大確信度: {mean_all:.4f}")
    print(f"確信度 > 0.5 の動画数: {n_gt_05}/{len(mean_confs)}")

    total_windows = sum(v["n_windows"] for v in all_audio_results.values())
    print(f"総窓数: {total_windows}")

    if mean_all < 0.01:
        print("  警告: 平均確信度 < 0.01。音声に鳥の鳴き声がほとんど検出されていません。")

    out_path = os.path.join(OUT_DIR, "audio_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "n_videos": len(all_audio_results),
                "mean_max_confidence": mean_all,
                "n_videos_gt_05": n_gt_05,
                "total_windows": total_windows,
                "elapsed_sec": elapsed_total,
                "feat_dim": FEAT_DIM,
                "window_sec": 3.0,
                "hop_sec": 1.0,
            },
            "videos": all_audio_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"保存: {out_path}")


if __name__ == "__main__":
    main()
