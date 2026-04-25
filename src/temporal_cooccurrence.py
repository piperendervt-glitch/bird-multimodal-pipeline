"""音声パイプライン改善: 映像と音声の時間的共起検出。

Phase 5g 統合で生成済みの YOLO 検出結果（フレーム単位）と
BirdNET 結果（3秒窓単位）を時間軸で照合し、
「鳥が映っている時に BirdNET 確信度が高い」パターンを自動検出する。

追加の YOLO/BirdNET 推論は不要。既存の JSON のみ使用。
"""

import csv
import json
import os
import sys
from collections import Counter

import numpy as np


def load_json(path):
    """JSON を読み込み、{"videos": {...}} のラッパーがあれば剥がす。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "videos" in data:
        return data["videos"]
    return data


def load_data():
    """Phase 5g 統合の結果と人間ラベルを読み込む。"""
    frame_results = load_json("../results/phase5g_youtube/frame_results.json")
    audio_results = load_json("../results/phase5g_youtube/audio_results.json")

    labels = {}
    with open("../data/youtube_greattit/labels.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["video_id"]] = row

    return frame_results, audio_results, labels


def compute_cooccurrence(frame_result, audio_result):
    """1 本の動画について、映像と音声の時間的共起を計算する。

    各時間窓（3秒）について:
    - その窓に含まれるフレームで YOLO が鳥を検出したか
    - BirdNET の確信度はどれくらいか
    - 両方が高ければ「共起」
    """
    crops = frame_result.get("crops", [])
    windows = audio_result.get("windows", [])

    if not crops or not windows:
        return None

    window_analysis = []

    for w in windows:
        w_start = w["start"]
        w_end = w["end"]
        gt_conf = w.get("great_tit_confidence", 0)
        max_conf = w.get("max_confidence", 0)

        # この窓に含まれるフレームを取得
        frames_in_window = [
            c for c in crops
            if w_start <= c.get("timestamp", -1) <= w_end
        ]

        # YOLO 検出の統計（n_birds が無い場合は confidence>0 かつ 非フォールバックで判定）
        n_frames = len(frames_in_window)
        n_detected = sum(
            1 for f in frames_in_window
            if f.get("n_birds", 0) > 0 or
            (f.get("confidence", 0) > 0 and not f.get("is_fallback", False))
        )

        yolo_detection_rate = n_detected / max(n_frames, 1)
        yolo_max_conf = max(
            (f.get("confidence", 0) for f in frames_in_window), default=0
        )

        # 共起スコア: YOLO検出率 × BirdNET確信度
        cooccurrence_score = yolo_detection_rate * gt_conf

        # 共起の分類
        yolo_threshold = 0.5      # フレームの50%以上で鳥検出
        birdnet_threshold = 0.3   # Great Tit 確信度

        if (yolo_detection_rate >= yolo_threshold
                and gt_conf >= birdnet_threshold):
            pattern = "cooccur"        # 共起
        elif (yolo_detection_rate >= yolo_threshold
              and gt_conf < birdnet_threshold):
            pattern = "visual_only"    # 映像のみ
        elif (yolo_detection_rate < yolo_threshold
              and gt_conf >= birdnet_threshold):
            pattern = "audio_only"     # 音声のみ
        else:
            pattern = "neither"        # どちらもなし

        window_analysis.append({
            "start": w_start,
            "end": w_end,
            "n_frames": n_frames,
            "n_detected": n_detected,
            "yolo_detection_rate": yolo_detection_rate,
            "yolo_max_conf": yolo_max_conf,
            "birdnet_gt_conf": gt_conf,
            "birdnet_max_conf": max_conf,
            "cooccurrence_score": cooccurrence_score,
            "pattern": pattern,
            "top_species": w.get("top_species", [])[:3],
        })

    return window_analysis


def analyze_video(video_id, window_analysis, human_label):
    """1 本の動画の共起パターンを集計する。"""
    if not window_analysis:
        return None

    n_windows = len(window_analysis)
    patterns = [w["pattern"] for w in window_analysis]
    pattern_counts = Counter(patterns)

    scores = [w["cooccurrence_score"] for w in window_analysis]
    gt_confs = [w["birdnet_gt_conf"] for w in window_analysis]
    yolo_rates = [w["yolo_detection_rate"] for w in window_analysis]

    # 共起区間の連続性
    cooccur_windows = [1 if w["pattern"] == "cooccur" else 0
                        for w in window_analysis]
    max_consecutive_cooccur = 0
    current_streak = 0
    for c in cooccur_windows:
        if c:
            current_streak += 1
            max_consecutive_cooccur = max(max_consecutive_cooccur, current_streak)
        else:
            current_streak = 0

    # YOLO検出率と BirdNET 確信度の時間相関
    if (len(yolo_rates) >= 3 and np.std(yolo_rates) > 0
            and np.std(gt_confs) > 0):
        correlation = float(np.corrcoef(yolo_rates, gt_confs)[0, 1])
    else:
        correlation = 0.0

    return {
        "video_id": video_id,
        "human_label": human_label,
        "n_windows": n_windows,
        "pattern_counts": dict(pattern_counts),
        "cooccur_ratio": pattern_counts.get("cooccur", 0) / n_windows,
        "visual_only_ratio": pattern_counts.get("visual_only", 0) / n_windows,
        "audio_only_ratio": pattern_counts.get("audio_only", 0) / n_windows,
        "neither_ratio": pattern_counts.get("neither", 0) / n_windows,
        "mean_cooccurrence_score": float(np.mean(scores)),
        "max_cooccurrence_score": float(np.max(scores)),
        "mean_gt_conf": float(np.mean(gt_confs)),
        "max_gt_conf": float(np.max(gt_confs)),
        "mean_yolo_rate": float(np.mean(yolo_rates)),
        "temporal_correlation": correlation,
        "max_consecutive_cooccur": max_consecutive_cooccur,
    }


def main():
    # Windows コンソール対策で UTF-8 出力に切り替える
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("=== 音声パイプライン改善: 時間的共起検出 ===")

    frame_results, audio_results, labels = load_data()

    all_analyses = {}
    all_summaries = {}

    print(f"\n{'動画ID':<14} {'ラベル':<10} {'共起':>5} {'映像のみ':>8} "
          f"{'音声のみ':>8} {'なし':>5} {'相関':>6} {'連続共起':>8}")
    print("-" * 85)

    for video_id in frame_results:
        if video_id not in audio_results:
            continue

        human_label = labels.get(video_id, {}).get("singing_matches_video", "?")

        window_analysis = compute_cooccurrence(
            frame_results[video_id],
            audio_results[video_id],
        )

        if window_analysis is None:
            continue

        all_analyses[video_id] = window_analysis

        summary = analyze_video(video_id, window_analysis, human_label)
        if summary:
            all_summaries[video_id] = summary

            print(f"{video_id:<14} {human_label:<10} "
                  f"{summary['cooccur_ratio']*100:>4.0f}% "
                  f"{summary['visual_only_ratio']*100:>7.0f}% "
                  f"{summary['audio_only_ratio']*100:>7.0f}% "
                  f"{summary['neither_ratio']*100:>4.0f}% "
                  f"{summary['temporal_correlation']:>6.2f} "
                  f"{summary['max_consecutive_cooccur']:>8}")

    # ========================================
    # ラベル別の共起パターン分析
    # ========================================
    print(f"\n{'='*70}")
    print(f"ラベル別の共起パターン")
    print(f"{'='*70}")

    label_groups = {}
    for vid, summary in all_summaries.items():
        label = summary["human_label"]
        label_groups.setdefault(label, []).append(summary)

    print(f"\n{'ラベル':<12} {'動画数':>6} {'共起率':>8} {'映像のみ':>8} "
          f"{'音声のみ':>8} {'相関':>6} {'連続共起':>8}")
    print("-" * 65)

    for label in ["yes", "partial", "other", "no_sound"]:
        if label not in label_groups:
            continue
        videos = label_groups[label]
        n = len(videos)

        mean_cooccur = np.mean([v["cooccur_ratio"] for v in videos])
        mean_visual = np.mean([v["visual_only_ratio"] for v in videos])
        mean_audio = np.mean([v["audio_only_ratio"] for v in videos])
        mean_corr = np.mean([v["temporal_correlation"] for v in videos])
        mean_consec = np.mean([v["max_consecutive_cooccur"] for v in videos])

        print(f"{label:<12} {n:>6} {mean_cooccur*100:>7.1f}% "
              f"{mean_visual*100:>7.1f}% {mean_audio*100:>7.1f}% "
              f"{mean_corr:>6.2f} {mean_consec:>8.1f}")

    # ========================================
    # 共起検出の有効性評価
    # ========================================
    print(f"\n{'='*70}")
    print(f"共起検出の有効性評価")
    print(f"{'='*70}")

    print(f"\n共起率の閾値別 Precision/Recall（正例: yes）")
    print(f"{'閾値':>6} {'検出数':>6} {'TP':>4} {'FP':>4} {'Prec':>8} {'Recall':>8}")
    print("-" * 45)

    yes_videos = set(vid for vid, s in all_summaries.items()
                      if s["human_label"] == "yes")

    for threshold in [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        detected = set(vid for vid, s in all_summaries.items()
                        if s["cooccur_ratio"] >= threshold)
        tp = len(detected & yes_videos)
        fp = len(detected - yes_videos)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(len(yes_videos), 1)

        print(f"{threshold:>6.2f} {len(detected):>6} {tp:>4} {fp:>4} "
              f"{precision*100:>7.1f}% {recall*100:>7.1f}%")

    # ========================================
    # 時間相関の分析
    # ========================================
    print(f"\n{'='*70}")
    print(f"YOLO検出率 × BirdNET確信度 の時間相関")
    print(f"{'='*70}")

    print(f"\n仮説: singing_matches_video = 'yes' の動画では")
    print(f"      YOLO で鳥が映っている時に BirdNET 確信度も高い（正の相関）")

    for label in ["yes", "partial", "other", "no_sound"]:
        if label not in label_groups:
            continue
        correlations = [v["temporal_correlation"] for v in label_groups[label]]
        print(f"\n  {label}: 平均相関 {np.mean(correlations):.3f} "
              f"(範囲: {np.min(correlations):.3f} 〜 {np.max(correlations):.3f})")

    # ========================================
    # WetlandBirds への適用可能性
    # ========================================
    print(f"\n{'='*70}")
    print(f"WetlandBirds への適用可能性")
    print(f"{'='*70}")

    wb_frame_path = "../results/bird_phase4b/frame_results.json"
    wb_audio_path = "../results/bird_phase4b/audio_results.json"

    if os.path.exists(wb_frame_path) and os.path.exists(wb_audio_path):
        wb_frames = load_json(wb_frame_path)
        wb_audio = load_json(wb_audio_path)

        wb_cooccur_count = 0
        wb_total = 0

        for vid in wb_frames:
            if vid not in wb_audio:
                continue

            wa = compute_cooccurrence(wb_frames[vid], wb_audio[vid])
            if wa is None:
                continue

            wb_total += 1
            cooccur_ratio = (sum(1 for w in wa if w["pattern"] == "cooccur")
                              / max(len(wa), 1))
            if cooccur_ratio > 0.05:
                wb_cooccur_count += 1

        print(f"\nWetlandBirds {wb_total} 動画中、共起パターン検出: "
              f"{wb_cooccur_count} 本")
        print(f"  注: WetlandBirds は Great Tit を含まないため、"
              f"great_tit_confidence は常に 0。")
        print(f"  したがって共起カウントが少ないのは想定通り（"
              f"映像と Great Tit 鳴き声の共起を測っているため）。")
        print(f"  → 一般化するには種非依存の確信度（max_confidence）を"
              f"使う必要がある。")
    else:
        print(f"\nWetlandBirds のデータが見つかりません。スキップ。")

    # ========================================
    # 結論と動的適応ルール
    # ========================================
    print(f"\n{'='*70}")
    print(f"結論: 動的適応ルールの提案")
    print(f"{'='*70}")

    print(f"""
    Phase 5g 統合の知見と時間的共起検出を組み合わせた動的ルール:

    ルール1: BirdNET 確信度閾値（Phase 5g 統合で確立）
      BirdNET GT確信度 > 0.3 → 音声を信用（Precision 100%）
      BirdNET GT確信度 ≤ 0.3 → 音声を無視

    ルール2: 時間的共起（本分析で検証）
      共起率 > X% → 映っている鳥が鳴いている可能性が高い
      共起率 ≤ X% → 映っている鳥と鳴いている鳥が異なる可能性

    ルール3: 組み合わせ
      BirdNET 確信度 > 0.3 AND 共起率 > X%
        → 音声を最大の重みで統合
      BirdNET 確信度 > 0.3 AND 共起率 ≤ X%
        → 音声を使うが重みを下げる（別の鳥の可能性）
      BirdNET 確信度 ≤ 0.3
        → 音声を無視（映像のみ）
    """)

    # 保存
    os.makedirs("../results/phase5g_youtube", exist_ok=True)
    output = {
        "summaries": all_summaries,
        "label_group_stats": {
            label: {
                "n_videos": len(videos),
                "mean_cooccur_ratio": float(np.mean(
                    [v["cooccur_ratio"] for v in videos])),
                "mean_visual_only_ratio": float(np.mean(
                    [v["visual_only_ratio"] for v in videos])),
                "mean_audio_only_ratio": float(np.mean(
                    [v["audio_only_ratio"] for v in videos])),
                "mean_temporal_correlation": float(np.mean(
                    [v["temporal_correlation"] for v in videos])),
                "mean_max_consecutive_cooccur": float(np.mean(
                    [v["max_consecutive_cooccur"] for v in videos])),
            }
            for label, videos in label_groups.items()
        },
    }
    out_path = "../results/phase5g_youtube/temporal_cooccurrence.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n保存: {out_path}")


if __name__ == "__main__":
    main()
