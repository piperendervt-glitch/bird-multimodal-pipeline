"""Phase 5g 統合 段階C: 人間ラベルと BirdNET の照合分析。

- YOLO の検出率（ラベル別）
- BirdNET Great Tit 確信度（動画別、ラベル別）
- 各動画で BirdNET が検出したトップ種
- 人間ラベルと BirdNET の一致率
- WetlandBirds との比較
- 失敗条件の判定
- 結論

このスクリプトは YOLO/DINOv2/BirdNET をインポートしない。
"""

import csv
import json
import os
import sys

import numpy as np


DATA_DIR = os.path.join("..", "data", "youtube_greattit")
OUT_DIR = os.path.join("..", "results", "phase5g_youtube")


def main():
    # Windows コンソールが cp932 でも UTF-8 で出力させる
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("=== Phase 5g 統合 段階C: 人間ラベルとの照合分析 ===")

    # データ読み込み
    with open(os.path.join(OUT_DIR, "frame_results.json"), encoding="utf-8") as f:
        frame_data = json.load(f)
    frame_results = frame_data.get("videos", frame_data)

    with open(os.path.join(OUT_DIR, "audio_results.json"), encoding="utf-8") as f:
        audio_data = json.load(f)
    audio_results = audio_data.get("videos", audio_data)

    labels = {}
    with open(os.path.join(DATA_DIR, "labels.csv"), encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["video_id"]] = row

    # ========================================
    # 分析1: YOLO 検出率（ラベル別）
    # ========================================
    print(f"\n--- 分析1: YOLO 鳥検出率（singing_matches_video 別） ---")

    categories = {}
    for vid, fr in frame_results.items():
        cat = fr.get("singing_matches_video", "?")
        if cat not in categories:
            categories[cat] = {"videos": 0, "total_frames": 0,
                               "detected_frames": 0}
        categories[cat]["videos"] += 1
        categories[cat]["total_frames"] += fr.get("n_keyframes_total", 0)
        categories[cat]["detected_frames"] += fr.get("n_detected", 0)

    print(f"{'ラベル':<12} {'動画数':>6} {'フレーム':>10} {'検出':>8} {'検出率':>8}")
    print("-" * 50)
    for cat, stats in sorted(categories.items()):
        rate = stats["detected_frames"] / max(stats["total_frames"], 1)
        print(f"{cat:<12} {stats['videos']:>6} {stats['total_frames']:>10} "
              f"{stats['detected_frames']:>8} {rate*100:>7.1f}%")

    # ========================================
    # 分析2: BirdNET Great Tit 確信度（動画別）
    # ========================================
    print(f"\n--- 分析2: BirdNET Great Tit 確信度（核心） ---")

    label_groups = {"yes": [], "partial": [], "other": [], "no_sound": []}

    print(f"\n{'動画ID':<14} {'ラベル':<10} {'GT平均':>8} {'GT最大':>8} "
          f"{'全体最大':>9} {'最頻トップ種':<40}")
    print("-" * 100)

    for vid, ar in audio_results.items():
        match_label = ar.get("singing_matches_video", "?")
        gt_mean = ar.get("great_tit_mean_conf", 0.0)
        gt_max = ar.get("great_tit_max_conf", 0.0)
        overall = ar.get("overall_max_conf", 0.0)

        # 各窓のトップ1種を集計し、最頻種を抽出
        top1_counts = {}
        for w in ar.get("windows", []):
            top_species = w.get("top_species", [])
            if top_species:
                name = top_species[0].get("species", "?")
                top1_counts[name] = top1_counts.get(name, 0) + 1
        top_species_name = max(top1_counts, key=top1_counts.get) if top1_counts else "N/A"
        # "学名_英名" → "英名" のみに整形
        if "_" in top_species_name:
            top_species_disp = top_species_name.split("_", 1)[1]
        else:
            top_species_disp = top_species_name

        print(f"{vid:<14} {match_label:<10} {gt_mean:>8.4f} {gt_max:>8.4f} "
              f"{overall:>9.4f} {top_species_disp[:40]:<40}")

        if match_label in label_groups:
            label_groups[match_label].append({
                "video_id": vid,
                "gt_mean": gt_mean,
                "gt_max": gt_max,
                "overall_max": overall,
                "top_species": top_species_name,
            })

    # ラベル別の統計
    print(f"\n--- ラベル別 BirdNET Great Tit 確信度の統計 ---")
    print(f"{'ラベル':<12} {'動画数':>6} {'GT平均':>10} {'GT最大':>10} {'全体最大':>10}")
    print("-" * 55)
    for label_val, videos in label_groups.items():
        if not videos:
            continue
        gt_means = [v["gt_mean"] for v in videos]
        gt_maxes = [v["gt_max"] for v in videos]
        overall_maxes = [v["overall_max"] for v in videos]
        print(f"{label_val:<12} {len(videos):>6} "
              f"{np.mean(gt_means):>10.4f} {np.mean(gt_maxes):>10.4f} "
              f"{np.mean(overall_maxes):>10.4f}")

    # ========================================
    # 分析3: BirdNET が検出したトップ種（Great Tit 以外も）
    # ========================================
    print(f"\n--- 分析3: 動画ごとの最頻トップ種（窓単位） ---")
    print(f"{'動画ID':<14} {'ラベル':<10} {'最頻トップ種':<50} {'出現窓数':>8}")
    print("-" * 90)
    for vid, ar in audio_results.items():
        match_label = ar.get("singing_matches_video", "?")
        top1_counts = {}
        for w in ar.get("windows", []):
            top_species = w.get("top_species", [])
            if top_species:
                name = top_species[0].get("species", "?")
                top1_counts[name] = top1_counts.get(name, 0) + 1
        if top1_counts:
            top_name = max(top1_counts, key=top1_counts.get)
            n = top1_counts[top_name]
            disp = top_name.split("_", 1)[1] if "_" in top_name else top_name
        else:
            top_name = "N/A"
            n = 0
            disp = "N/A"
        print(f"{vid:<14} {match_label:<10} {disp[:50]:<50} {n:>8}")

    # ========================================
    # 分析4: 人間ラベルと BirdNET の一致評価
    # ========================================
    print(f"\n--- 分析4: 音声の信頼性判定（閾値による検出/未検出） ---")

    threshold = 0.3
    birdnet_detected = []
    birdnet_missed = []

    for vid, ar in audio_results.items():
        gt_max = ar.get("great_tit_max_conf", 0.0)
        match_label = ar.get("singing_matches_video", "?")

        if gt_max > threshold:
            birdnet_detected.append({"video_id": vid, "label": match_label,
                                      "conf": gt_max})
        else:
            birdnet_missed.append({"video_id": vid, "label": match_label,
                                    "conf": gt_max})

    print(f"\nBirdNET Great Tit 検出閾値: {threshold}")
    print(f"  検出: {len(birdnet_detected)} 本")
    print(f"  未検出: {len(birdnet_missed)} 本")

    print(f"\n  検出されたものの内訳:")
    for d in birdnet_detected:
        print(f"    {d['video_id']:<14} ラベル={d['label']:<10} 確信度={d['conf']:.4f}")

    print(f"\n  未検出（GT最大≦{threshold}）の内訳:")
    for d in birdnet_missed:
        print(f"    {d['video_id']:<14} ラベル={d['label']:<10} 確信度={d['conf']:.4f}")

    # 一致率
    human_yes_videos = label_groups.get("yes", [])
    human_yes_detected = [v for v in human_yes_videos if v["gt_max"] > threshold]
    human_yes_total = len(human_yes_videos)

    human_other_videos = label_groups.get("other", [])
    human_other_missed = [v for v in human_other_videos
                          if v["gt_max"] <= threshold]
    human_other_total = len(human_other_videos)

    human_no_sound_videos = label_groups.get("no_sound", [])
    human_no_sound_missed = [v for v in human_no_sound_videos
                              if v["gt_max"] <= threshold]
    human_no_sound_total = len(human_no_sound_videos)

    print(f"\n  人間 'yes'（鳴き声一致）で BirdNET 検出: "
          f"{len(human_yes_detected)}/{human_yes_total}")
    print(f"  人間 'other'（別の鳥）で BirdNET 未検出: "
          f"{len(human_other_missed)}/{human_other_total}")
    print(f"  人間 'no_sound'（鳴き声なし）で BirdNET 未検出: "
          f"{len(human_no_sound_missed)}/{human_no_sound_total}")

    # ========================================
    # 分析5: WetlandBirds との比較
    # ========================================
    print(f"\n--- 分析5: WetlandBirds (Phase 4b) との比較 ---")

    wb_audio_path = os.path.join("..", "results", "bird_phase4b",
                                  "audio_results.json")
    if os.path.exists(wb_audio_path):
        with open(wb_audio_path, encoding="utf-8") as f:
            wb_audio = json.load(f)
        wb_videos = wb_audio.get("videos", wb_audio)

        wb_max_confs = [v.get("mean_max_confidence", 0)
                        for v in wb_videos.values()]
        yt_max_confs = [ar.get("overall_max_conf", 0)
                        for ar in audio_results.values()]

        print(f"  WetlandBirds 平均最大確信度: {np.mean(wb_max_confs):.4f} "
              f"({len(wb_max_confs)} 動画)")
        print(f"  YouTube GT   平均最大確信度: {np.mean(yt_max_confs):.4f} "
              f"({len(yt_max_confs)} 動画)")

        yes_confs = [ar["overall_max_conf"]
                     for ar in audio_results.values()
                     if ar.get("singing_matches_video") == "yes"]
        if yes_confs:
            print(f"  YouTube GT (yes のみ) 平均: {np.mean(yes_confs):.4f} "
                  f"({len(yes_confs)} 動画)")
    else:
        print(f"  {wb_audio_path} が見つかりません（比較スキップ）")

    # ========================================
    # 分析6: 失敗条件の判定
    # ========================================
    print(f"\n--- 失敗条件の判定 ---")

    yes_gt_confs = [v["gt_max"] for v in label_groups.get("yes", [])]
    other_gt_confs = [v["gt_max"] for v in label_groups.get("other", [])]
    partial_gt_confs = [v["gt_max"] for v in label_groups.get("partial", [])]

    if yes_gt_confs:
        mean_yes = np.mean(yes_gt_confs)
        if mean_yes < 0.3:
            print(f"  NG: yes の BirdNET GT最大確信度の平均 ({mean_yes:.4f}) < 0.3")
        else:
            print(f"  OK: yes の BirdNET GT最大確信度の平均 ({mean_yes:.4f}) >= 0.3")

    if yes_gt_confs and other_gt_confs:
        diff = np.mean(yes_gt_confs) - np.mean(other_gt_confs)
        if diff <= 0:
            print(f"  NG: yes ({np.mean(yes_gt_confs):.4f}) "
                  f"<= other ({np.mean(other_gt_confs):.4f})")
        else:
            print(f"  OK: yes ({np.mean(yes_gt_confs):.4f}) "
                  f"> other ({np.mean(other_gt_confs):.4f}): 差 {diff:.4f}")

    # ========================================
    # 結論
    # ========================================
    print(f"\n{'='*60}")
    print(f"Phase 5g 統合 結論")
    print(f"{'='*60}")

    if yes_gt_confs and other_gt_confs:
        if np.mean(yes_gt_confs) > np.mean(other_gt_confs):
            print(f"  BirdNET の確信度は人間ラベル (singing_matches_video) と")
            print(f"  正の相関を持つ。音声の信頼性判定に使用可能。")
            print(f"  → 動的適応で「BirdNET 確信度が高い時だけ音声を使う」")
            print(f"    ルールが有効である証拠。")
        else:
            print(f"  BirdNET の確信度は人間ラベルと相関しない。")
            print(f"  → 音声の信頼性判定に BirdNET の確信度は使えない。")

    # 保存
    output = {
        "yolo_stats": categories,
        "birdnet_by_label": {
            k: [{"video_id": v["video_id"],
                 "gt_mean": v["gt_mean"],
                 "gt_max": v["gt_max"],
                 "overall_max": v["overall_max"],
                 "top_species": v["top_species"]} for v in vs]
            for k, vs in label_groups.items()
        },
        "threshold": threshold,
        "birdnet_detected_count": len(birdnet_detected),
        "birdnet_missed_count": len(birdnet_missed),
        "human_yes_detected": len(human_yes_detected),
        "human_yes_total": human_yes_total,
        "human_other_missed": len(human_other_missed),
        "human_other_total": human_other_total,
    }

    out_path = os.path.join(OUT_DIR, "phase5g_analysis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n保存: {out_path}")


if __name__ == "__main__":
    main()
