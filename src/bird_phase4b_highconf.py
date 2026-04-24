import numpy as np
import json
import os
import sys
import io
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Windows cp932 対策: 標準出力を UTF-8 にする
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def main():
    print("=== Phase 4b-E: BirdNET 高確信度動画の再分析 ===")

    # 結果ファイル読み込み
    with open("../results/bird_phase4b/audio_results.json", encoding="utf-8") as f:
        audio_results_raw = json.load(f)

    with open("../results/bird_phase4b/frame_results.json", encoding="utf-8") as f:
        frame_results_raw = json.load(f)

    with open("../results/bird_phase4b/species_mapping.json", encoding="utf-8") as f:
        mapping = json.load(f)

    with open("../data/wetlandbirds/splits.json", encoding="utf-8") as f:
        splits = json.load(f)

    # audio_results は { "summary": ..., "videos": { video_name: {...}} } の形式
    audio_videos = audio_results_raw["videos"]
    frame_videos = frame_results_raw["videos"] if "videos" in frame_results_raw else frame_results_raw

    n_classes = mapping["n_classes"]
    id_to_species = mapping["id_to_species"]
    species_to_id = mapping["species_to_id"]

    # =========================================
    # 分析1: 高確信度 19 動画の特定と詳細
    # =========================================
    print("\n--- 分析1: BirdNET 確信度 > 0.5 の動画一覧 ---")

    high_conf_videos = []
    for video_name, ainfo in audio_videos.items():
        mean_max = ainfo.get("mean_max_confidence", 0) or 0
        if mean_max > 0.5:
            species = ainfo.get("species", "unknown")
            high_conf_videos.append({
                "video_name": video_name,
                "species": species,
                "mean_max_confidence": mean_max,
            })

    high_conf_videos.sort(key=lambda x: x["mean_max_confidence"], reverse=True)

    print(f"\n高確信度動画: {len(high_conf_videos)} 本")
    print(f"{'動画名':<40} {'種':<25} {'BirdNET確信度':>12}")
    print("-" * 80)
    for v in high_conf_videos:
        print(f"{v['video_name']:<40} {v['species']:<25} {v['mean_max_confidence']:>12.4f}")

    # 種の分布
    from collections import Counter
    species_counts = Counter(v["species"] for v in high_conf_videos)
    print(f"\n高確信度動画の種分布:")
    for sp, cnt in species_counts.most_common():
        print(f"  {sp}: {cnt} 本")

    # =========================================
    # 分析2: BirdNET の確信度分布（上位3種の平均スコア）
    # =========================================
    print("\n--- 分析2: BirdNET 確信度分布（窓平均 上位3位） ---")

    print("\n各高確信度動画の BirdNET 確信度分布:")
    print(f"{'動画名':<40} {'正解種':<22} {'top1':>6} {'top2':>6} {'top3':>6}")
    print("-" * 85)

    for v in high_conf_videos:
        video_name = v["video_name"]
        ainfo = audio_videos[video_name]
        windows = ainfo.get("windows", [])

        if not windows:
            continue

        all_features = np.array([w["features"] for w in windows])
        mean_features = np.mean(all_features, axis=0)

        top1 = mean_features[0] if len(mean_features) > 0 else 0
        top2 = mean_features[1] if len(mean_features) > 1 else 0
        top3 = mean_features[2] if len(mean_features) > 2 else 0

        print(f"{video_name:<40} {v['species']:<22} {top1:>6.3f} {top2:>6.3f} {top3:>6.3f}")

    # =========================================
    # 分析3: テストセット内の高確信度動画での精度比較
    # =========================================
    print("\n--- 分析3: テストセット内の高確信度動画の特定 ---")

    test_key = "test_set"
    # splits の値は ".mp4" なしの "124-black_headed_gull" のような文字列
    test_video_names = set()
    for v_id in splits[test_key]:
        v_str = str(v_id)
        # ".mp4" を付けて動画名に揃える
        candidate = v_str if v_str.endswith(".mp4") else f"{v_str}.mp4"
        test_video_names.add(candidate)

    print(f"テストセット動画数: {len(test_video_names)}")

    high_conf_names = set(v["video_name"] for v in high_conf_videos)
    high_conf_in_test = high_conf_names & test_video_names

    print(f"テストセット内の高確信度動画: {len(high_conf_in_test)} 本")

    if high_conf_in_test:
        for vn in sorted(high_conf_in_test):
            species = audio_videos[vn]["species"]
            conf = audio_videos[vn]["mean_max_confidence"]
            print(f"  {vn}: {species} (確信度 {conf:.4f})")

    # =========================================
    # 分析4: Phase 4b の学習済み分類器で予測比較
    # =========================================
    print("\n--- 分析4: 学習済み分類器での予測比較 ---")

    train_data = np.load("../results/bird_phase4b/features_train_set.npz", allow_pickle=True)
    test_data = np.load("../results/bird_phase4b/features_test_set.npz", allow_pickle=True)

    val_path = "../results/bird_phase4b/features_val_set.npz"
    if os.path.exists(val_path):
        val_data = np.load(val_path, allow_pickle=True)
        X_train_v = np.concatenate([train_data["X_visual"], val_data["X_visual"]])
        X_train_c = np.concatenate([train_data["X_combined"], val_data["X_combined"]])
        y_train = np.concatenate([train_data["y"], val_data["y"]])
    else:
        X_train_v = train_data["X_visual"]
        X_train_c = train_data["X_combined"]
        y_train = train_data["y"]

    X_test_v = test_data["X_visual"]
    X_test_c = test_data["X_combined"]
    y_test = test_data["y"]
    test_names = [str(v) for v in test_data["video_names"]]

    # 映像のみの分類器を学習
    scaler_v = StandardScaler()
    X_tr_v = scaler_v.fit_transform(X_train_v)
    X_te_v = scaler_v.transform(X_test_v)
    lr_v = LogisticRegression(max_iter=2000, C=1.0, random_state=42,
                              class_weight="balanced")
    lr_v.fit(X_tr_v, y_train)
    preds_v = lr_v.predict(X_te_v)
    probs_v = lr_v.predict_proba(X_te_v)

    # 統合の分類器を学習
    scaler_c = StandardScaler()
    X_tr_c = scaler_c.fit_transform(X_train_c)
    X_te_c = scaler_c.transform(X_test_c)
    lr_c = LogisticRegression(max_iter=2000, C=1.0, random_state=42,
                              class_weight="balanced")
    lr_c.fit(X_tr_c, y_train)
    preds_c = lr_c.predict(X_te_c)
    probs_c = lr_c.predict_proba(X_te_c)

    # テスト動画ごとの詳細
    print(f"\n{'動画名':<40} {'正解':<20} {'映像予測':<22} {'統合予測':<22} {'高確信':>6}")
    print("-" * 120)

    for idx, vname in enumerate(test_names):
        true_label = id_to_species[str(y_test[idx])]
        pred_v_label = id_to_species[str(preds_v[idx])]
        pred_c_label = id_to_species[str(preds_c[idx])]
        is_high = "★" if vname in high_conf_names else ""

        mark_v = "○" if pred_v_label == true_label else "×"
        mark_c = "○" if pred_c_label == true_label else "×"

        print(f"{vname:<40} {true_label:<20} {mark_v} {pred_v_label:<20} "
              f"{mark_c} {pred_c_label:<20} {is_high:>6}")

    # =========================================
    # 分析5: 高確信度 vs 低確信度の精度差
    # =========================================
    print("\n--- 分析5: 高確信度 vs 低確信度の精度差 ---")

    acc_v_all = accuracy_score(y_test, preds_v)
    acc_c_all = accuracy_score(y_test, preds_c)

    print(f"テスト全体 ({len(y_test)} 動画):")
    print(f"  映像のみ:  {acc_v_all*100:.2f}%")
    print(f"  統合:      {acc_c_all*100:.2f}%")
    print(f"  差:        {(acc_c_all - acc_v_all)*100:+.2f}pp")

    high_idx = [i for i, vn in enumerate(test_names) if vn in high_conf_names]
    acc_v_high = None
    acc_c_high = None
    if high_idx:
        y_high = y_test[high_idx]
        preds_v_high = preds_v[high_idx]
        preds_c_high = preds_c[high_idx]

        acc_v_high = accuracy_score(y_high, preds_v_high)
        acc_c_high = accuracy_score(y_high, preds_c_high)

        print(f"\n高確信度動画のみ ({len(high_idx)} 動画):")
        print(f"  映像のみ:  {acc_v_high*100:.2f}%")
        print(f"  統合:      {acc_c_high*100:.2f}%")
        print(f"  差:        {(acc_c_high - acc_v_high)*100:+.2f}pp")
    else:
        print("\nテストセットに高確信度動画がありません。")

    low_idx = [i for i, vn in enumerate(test_names) if vn not in high_conf_names]
    if low_idx:
        y_low = y_test[low_idx]
        preds_v_low = preds_v[low_idx]
        preds_c_low = preds_c[low_idx]

        acc_v_low = accuracy_score(y_low, preds_v_low)
        acc_c_low = accuracy_score(y_low, preds_c_low)

        print(f"\n低確信度動画のみ ({len(low_idx)} 動画):")
        print(f"  映像のみ:  {acc_v_low*100:.2f}%")
        print(f"  統合:      {acc_c_low*100:.2f}%")
        print(f"  差:        {(acc_c_low - acc_v_low)*100:+.2f}pp")

    # =========================================
    # 分析6: mallard 誤分類の詳細
    # =========================================
    print("\n--- 分析6: mallard 誤分類の詳細 ---")

    mallard_idx = [i for i, vn in enumerate(test_names)
                   if "mallard" in vn.lower()]

    if mallard_idx:
        for idx in mallard_idx:
            vname = test_names[idx]
            true_label = id_to_species[str(y_test[idx])]

            top3_v = np.argsort(probs_v[idx])[::-1][:3]
            top3_c = np.argsort(probs_c[idx])[::-1][:3]

            print(f"\n{vname}:")
            print(f"  正解: {true_label}")
            print(f"  映像のみ Top-3:")
            for rank, ci in enumerate(top3_v):
                print(f"    {rank+1}. {id_to_species[str(ci)]}: {probs_v[idx][ci]:.4f}")
            print(f"  統合 Top-3:")
            for rank, ci in enumerate(top3_c):
                print(f"    {rank+1}. {id_to_species[str(ci)]}: {probs_c[idx][ci]:.4f}")

            is_high = vname in high_conf_names
            audio_conf = audio_videos.get(vname, {}).get("mean_max_confidence", 0)
            print(f"  BirdNET 確信度: {audio_conf:.4f} ({'高確信度' if is_high else '低確信度'})")

    # =========================================
    # サマリー
    # =========================================
    print(f"\n{'='*60}")
    print(f"Phase 4b-E サマリー")
    print(f"{'='*60}")
    print(f"高確信度動画数: {len(high_conf_videos)} / 178")
    print(f"テストセット内の高確信度動画: {len(high_conf_in_test)}")
    print(f"BirdNET 高確信度動画の種分布: {dict(species_counts)}")

    print(f"\n--- 結論 ---")
    if high_conf_in_test:
        if len(high_idx) > 0:
            if acc_c_high > acc_v_high:
                print("  鳴き声がある動画では音声統合が映像単独を上回った。")
                print("  → 音声モダリティは鳴き声がある条件で有効。")
            elif acc_c_high == acc_v_high:
                print("  鳴き声がある動画でも音声統合と映像単独は同等。")
                print("  → 映像が十分に強く、音声の追加情報が不要な可能性。")
            else:
                print("  鳴き声がある動画で音声統合が映像単独より悪化。")
                print("  → BirdNET の検出種が正解と異なり、ノイズになっている可能性。")
    else:
        print("  テストセットに高確信度動画が含まれず、直接比較できない。")
        print("  → 学習セットでの傾向分析のみ可能。")

    # 保存
    output = {
        "high_conf_videos": high_conf_videos,
        "n_high_conf": len(high_conf_videos),
        "n_high_conf_in_test": len(high_conf_in_test),
        "species_distribution": dict(species_counts),
    }
    with open("../results/bird_phase4b/phase4b_highconf_analysis.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n保存: results/bird_phase4b/phase4b_highconf_analysis.json")


if __name__ == "__main__":
    main()
