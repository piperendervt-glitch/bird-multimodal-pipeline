"""Phase 5g 統合 段階B: YouTube Great Tit 動画から音声を分離し BirdNET 処理する。

- ffmpeg で動画 → 48kHz mono WAV に変換
- BirdNET v2.4 を 3秒窓 / 1秒ホップ (overlap_duration_s=2.0) で実行
- labels.csv の start_sec / end_sec で範囲を絞る（ffmpeg の -ss/-to を使用）
- 各窓で Great Tit (Parus major) の確信度を特別に追跡
- 各窓のトップ5の種名を保存
- Phase 4b と同じ 15次元の特徴ベクトルも生成

このスクリプトは YOLO や DINOv2 をインポートしない。
"""

import csv
import json
import os
import subprocess
import time
from collections import defaultdict

import numpy as np


DATA_DIR = os.path.join("..", "data", "youtube_greattit")
VIDEO_DIR = os.path.join(DATA_DIR, "videos")
OUT_DIR = os.path.join("..", "results", "phase5g_youtube")
AUDIO_DIR = os.path.join(OUT_DIR, "audio")

TOP_K = 10
FEAT_DIM = TOP_K + 5  # 10 確信度 + max/mean/std/count(>0.5)/margin


def check_ffmpeg():
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=10)
        return r.returncode == 0
    except FileNotFoundError:
        return False


def extract_audio(video_path, output_wav, start_sec=0.0, end_sec=0.0):
    """ffmpeg で動画から 48kHz mono WAV を生成。

    end_sec > 0 の場合は -ss / -to で時間範囲を限定する。
    """
    cmd = ["ffmpeg", "-y"]
    if end_sec > 0:
        cmd += ["-ss", f"{max(0.0, start_sec):.3f}",
                "-to", f"{end_sec:.3f}"]
    cmd += [
        "-i", video_path,
        "-vn",  # 映像なし
        "-acodec", "pcm_s16le",
        "-ar", "48000",  # 48kHz
        "-ac", "1",  # モノラル
        "-loglevel", "error",
        output_wav,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    return r.returncode == 0, r.stderr


def parse_float(v, default=0.0):
    """CSV 数値文字列のパース（空は default）。"""
    if v is None:
        return default
    s = str(v).strip()
    if not s:
        return default
    try:
        return float(s)
    except ValueError:
        return default


def predict_windows(model, audio_path):
    """1 つの音声ファイルを 3秒窓 / 1秒ホップで BirdNET 予測する。

    各窓について [(start, end, [(species, conf), ...])] のリストを返す。
    """
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
        print(f"    構造化配列変換失敗: {e}")
        return []

    # フィールド名: ('input', 'start_time', 'end_time', 'species_name', 'confidence')
    names = arr.dtype.names or ()
    if not ("start_time" in names and "end_time" in names
            and "confidence" in names and "species_name" in names):
        print(f"    予期しない dtype: {names}")
        return []

    # (start, end) でグループ化し、種ごとの確信度リストを構築
    groups = defaultdict(list)
    for row in arr:
        s = float(row["start_time"])
        e = float(row["end_time"])
        species = str(row["species_name"])
        c = float(row["confidence"])
        groups[(s, e)].append((species, c))

    windows = []
    for (s, e), species_confs in sorted(groups.items(), key=lambda kv: kv[0]):
        # 確信度の高い順にソート
        species_confs_sorted = sorted(species_confs, key=lambda x: x[1], reverse=True)
        windows.append({"start": s, "end": e, "species_confs": species_confs_sorted})
    return windows


def build_feature_vector(confidences, top_k=TOP_K):
    """上位確信度のリストから固定長ベクトルを生成 (Phase 4b と同一規約)。"""
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
    return [float(x) for x in feats]


def is_great_tit(species_name):
    """種名が Great Tit (Parus major) かどうか判定する。"""
    name_lower = species_name.lower()
    return ("great tit" in name_lower) or ("parus major" in name_lower)


def main():
    print("=== Phase 5g 統合 段階B: 音声分離 + BirdNET ===")

    if not check_ffmpeg():
        print("エラー: ffmpeg が見つかりません")
        return
    print("ffmpeg: OK")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)

    # ラベル読み込み
    labels = {}
    with open(os.path.join(DATA_DIR, "labels.csv"), encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["video_id"]] = row

    # manifest 読み込み
    with open(os.path.join(DATA_DIR, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    print(f"対象動画: {len(manifest)} 本")

    # BirdNET モデルのロード
    print("\nBirdNET モデルをロード中...")
    from birdnet import load
    model = load("acoustic", "2.4", "tf")
    print("  モデルロード完了")

    all_audio = {}
    start_total = time.time()

    for i, entry in enumerate(manifest):
        video_id = entry["video_id"]
        filename = entry.get("filename", f"{video_id}.mp4")
        video_path = os.path.join(VIDEO_DIR, filename)
        label = labels.get(video_id, {})

        if not os.path.exists(video_path):
            print(f"  警告: {video_path} が見つかりません")
            continue

        start_sec = parse_float(label.get("start_sec"), 0.0)
        end_sec = parse_float(label.get("end_sec"), 0.0)

        # 音声抽出（時間範囲付き）
        wav_path = os.path.join(AUDIO_DIR, f"{video_id}.wav")
        if not (os.path.exists(wav_path) and os.path.getsize(wav_path) > 0):
            ok, err = extract_audio(video_path, wav_path,
                                    start_sec=start_sec, end_sec=end_sec)
            if not ok:
                print(f"  {video_id}: 音声抽出失敗: {err[:100]}")
                all_audio[video_id] = {
                    "category": label.get("category", ""),
                    "singing_matches_video": label.get("singing_matches_video", ""),
                    "n_windows": 0,
                    "windows": [],
                    "great_tit_mean_conf": 0.0,
                    "great_tit_max_conf": 0.0,
                    "overall_max_conf": 0.0,
                    "error": "audio_extract_failed",
                }
                continue

        # BirdNET 予測
        windows_raw = predict_windows(model, wav_path)

        windows = []
        gt_confs = []
        max_confs = []

        for w in windows_raw:
            species_confs = w["species_confs"]  # 確信度降順

            # 全 top_k から特徴ベクトル
            confs_only = [c for _, c in species_confs]
            feats = build_feature_vector(confs_only, TOP_K)

            # Great Tit 確信度（任意の上位候補から検索）
            gt_conf = 0.0
            for species, conf in species_confs:
                if is_great_tit(species):
                    gt_conf = conf
                    break

            top5 = [{"species": s, "confidence": float(c)}
                    for s, c in species_confs[:5]]
            max_conf = float(species_confs[0][1]) if species_confs else 0.0

            windows.append({
                "start": w["start"],
                "end": w["end"],
                "top_species": top5,
                "great_tit_confidence": float(gt_conf),
                "max_confidence": max_conf,
                "features": feats,
            })

            gt_confs.append(gt_conf)
            max_confs.append(max_conf)

        all_audio[video_id] = {
            "category": label.get("category", ""),
            "singing_matches_video": label.get("singing_matches_video", ""),
            "start_sec": start_sec,
            "end_sec": end_sec,
            "n_windows": len(windows),
            "windows": windows,
            "great_tit_mean_conf": float(np.mean(gt_confs)) if gt_confs else 0.0,
            "great_tit_max_conf": float(np.max(gt_confs)) if gt_confs else 0.0,
            "overall_max_conf": float(np.mean(max_confs)) if max_confs else 0.0,
        }

        print(f"  {i+1}. {video_id}: {len(windows)}窓, "
              f"GT確信度 平均{all_audio[video_id]['great_tit_mean_conf']:.3f} "
              f"最大{all_audio[video_id]['great_tit_max_conf']:.3f}, "
              f"ラベル: {label.get('singing_matches_video', '?')}",
              flush=True)

    elapsed = time.time() - start_total

    print(f"\n=== サマリー ===")
    print(f"処理動画: {len(all_audio)} 本")
    print(f"処理時間: {elapsed:.1f} 秒")

    out_path = os.path.join(OUT_DIR, "audio_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "n_videos": len(all_audio),
                "total_windows": sum(v.get("n_windows", 0) for v in all_audio.values()),
                "elapsed_sec": elapsed,
                "feat_dim": FEAT_DIM,
                "window_sec": 3.0,
                "hop_sec": 1.0,
            },
            "videos": all_audio,
        }, f, indent=2, ensure_ascii=False)
    print(f"保存: {out_path}")


if __name__ == "__main__":
    main()
