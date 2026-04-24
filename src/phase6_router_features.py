"""
Phase 6-A: ルーティング特徴量の構築

各動画のフレーム予測から「ルーティング特徴量」を構築する。
この特徴量は、段階 B の AdaptiveFeatureRouter が
「この動画に対してどの集約方法が最適か」を判断するための入力となる。

制約:
- DINOv2, YOLO, BirdNET はインポートしない
- Phase 5e の frame_predictions.json と npz のみを使用
- frame_predictions.json は Phase 5e でテストセットのみ生成済み
  → 段階 B では K-fold CV でルーターの学習/評価を分離する
"""

import numpy as np
import json
import os
from collections import Counter


def compute_routing_features(frame_predictions):
    """
    フレーム単位の予測結果から、ルーティング特徴量（20次元）を計算する。

    この特徴量は「この動画に対してどの集約方法が最適か」を
    判断するための入力として使用される。
    """
    # 空入力ガード：特徴量は全て 0
    if not frame_predictions:
        return np.zeros(20, dtype=np.float32)

    n_frames = len(frame_predictions)
    predicted_labels = [fp["predicted_label"] for fp in frame_predictions]
    confidences = [fp["confidence"] for fp in frame_predictions]
    prob_distributions = [fp["prob_distribution"] for fp in frame_predictions]
    yolo_confs = [fp.get("yolo_confidence", 0) for fp in frame_predictions]
    is_fallbacks = [fp.get("is_fallback", False) for fp in frame_predictions]

    features = []

    # === グループ1: 予測の一致度（多数決 vs フレーム平均の選択に関連） ===

    # フレーム間の予測一致率（最多クラスの比率）
    counter = Counter(predicted_labels)
    most_common_count = counter.most_common(1)[0][1]
    agreement_rate = most_common_count / n_frames
    features.append(agreement_rate)

    # ユニークな予測クラス数の比率
    n_unique = len(counter)
    features.append(n_unique / max(n_frames, 1))

    # 上位2クラスの票差
    sorted_counts = sorted(counter.values(), reverse=True)
    if len(sorted_counts) >= 2:
        margin = (sorted_counts[0] - sorted_counts[1]) / n_frames
    else:
        margin = 1.0
    features.append(margin)

    # 予測分布のエントロピー（不確実性）
    probs_pred = np.array(sorted_counts) / n_frames
    entropy = -np.sum(probs_pred * np.log(probs_pred + 1e-9))
    features.append(entropy)

    # === グループ2: 確信度の統計（高確信度誤予測の検出に関連） ===

    conf_array = np.array(confidences)
    features.append(conf_array.mean())          # 平均確信度
    features.append(conf_array.std())           # 確信度の標準偏差
    features.append(conf_array.min())           # 最小確信度
    features.append(conf_array.max())           # 最大確信度
    features.append(np.median(conf_array))      # 中央確信度

    # 確信度の歪度（高確信度外れ値の検出）
    if conf_array.std() > 1e-9:
        skewness = ((conf_array - conf_array.mean()) ** 3).mean() / (conf_array.std() ** 3)
    else:
        skewness = 0.0
    features.append(skewness)

    # === グループ3: 確率分布の特性 ===

    prob_matrix = np.array(prob_distributions)  # (n_frames, n_classes)

    # フレーム平均確率分布のエントロピー
    mean_probs = prob_matrix.mean(axis=0)
    mean_entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-9))
    features.append(mean_entropy)

    # フレーム平均確率の最大値
    features.append(mean_probs.max())

    # フレーム平均確率の上位2クラスの差
    sorted_mean = np.sort(mean_probs)[::-1]
    features.append(sorted_mean[0] - sorted_mean[1] if len(sorted_mean) > 1 else 1.0)

    # 確率分布のフレーム間分散（平均）
    prob_variance = prob_matrix.var(axis=0).mean()
    features.append(prob_variance)

    # === グループ4: YOLO 検出の品質 ===

    yolo_array = np.array(yolo_confs)
    features.append(yolo_array.mean())                # YOLO 平均確信度
    features.append(sum(is_fallbacks) / n_frames)    # フォールバック率

    # === グループ5: 時間的パターン ===

    # 予測の切り替わり回数（時間的安定性の逆数）
    switches = sum(
        1 for i in range(1, len(predicted_labels))
        if predicted_labels[i] != predicted_labels[i - 1]
    )
    features.append(switches / max(n_frames - 1, 1))

    # フレーム数
    features.append(float(n_frames))

    # 確信度のトレンド（後半平均 - 前半平均）
    if n_frames >= 4:
        half = n_frames // 2
        trend = conf_array[half:].mean() - conf_array[:half].mean()
    else:
        trend = 0.0
    features.append(trend)

    # 20 次元になるよう、残り枠を 0 で埋める（予備スロット）
    while len(features) < 20:
        features.append(0.0)

    return np.array(features[:20], dtype=np.float32)


def compute_best_method(frame_predictions, true_label):
    """
    各集約方法の正誤を計算し、最適な方法を返す。

    返り値:
        best: 正解した方法の中から「優先順位」に従い1つ選択
              （多数決 → フレーム平均 → 確信度加重 → YOLO 加重 → スライディング窓）
        results: 各手法の 0/1 正誤辞書
    """
    labels = [fp["predicted_label"] for fp in frame_predictions]
    probs = [fp["prob_distribution"] for fp in frame_predictions]

    if not labels:
        return None, {}

    results = {}

    # 1. 多数決
    counter = Counter(labels)
    majority = counter.most_common(1)[0][0]
    results["majority"] = int(majority == true_label)

    # 2. フレーム平均（確率）
    prob_matrix = np.array(probs)
    mean_probs = prob_matrix.mean(axis=0)
    frame_avg = int(mean_probs.argmax())
    results["frame_avg"] = int(frame_avg == true_label)

    # 3. 確信度加重（各フレームの確信度で重み付けして確率を合成）
    n_classes = len(probs[0])
    weighted = np.zeros(n_classes)
    for fp in frame_predictions:
        p = np.array(fp["prob_distribution"])
        w = fp.get("confidence", 0.0)
        weighted += w * p
    results["conf_weighted"] = int(weighted.argmax() == true_label)

    # 4. YOLO 加重（YOLO の確信度で重み付け、フォールバック時は 0.1）
    yolo_weighted = np.zeros(n_classes)
    for fp in frame_predictions:
        p = np.array(fp["prob_distribution"])
        yc = fp.get("yolo_confidence", 0)
        is_fb = fp.get("is_fallback", False)
        w = yc if (not is_fb and yc > 0) else 0.1
        yolo_weighted += w * p
    results["yolo_weighted"] = int(yolo_weighted.argmax() == true_label)

    # 5. スライディング窓 (size=5)
    if len(labels) >= 5:
        window_labels = []
        for i in range(len(labels) - 5 + 1):
            window = labels[i:i + 5]
            wc = Counter(window)
            window_labels.append(wc.most_common(1)[0][0])
        sw_counter = Counter(window_labels)
        sw_result = sw_counter.most_common(1)[0][0]
        results["sliding_window"] = int(sw_result == true_label)
    else:
        results["sliding_window"] = results["majority"]

    # 最良の方法を優先順位付きで選択
    priority = ["majority", "frame_avg", "conf_weighted",
                "yolo_weighted", "sliding_window"]
    best = None
    for m in priority:
        if results.get(m) == 1:
            best = m
            break
    if best is None:
        best = "frame_avg"   # どれも不正解なら安全策として frame_avg

    return best, results


def process_dataset(frame_predictions_path, dataset_name):
    """1つのデータセットのルーティング特徴量を構築"""
    print(f"\n--- {dataset_name} ---")

    with open(frame_predictions_path, encoding="utf-8") as f:
        video_preds = json.load(f)

    routing_features = []
    best_methods = []
    method_results = []
    video_names = []
    true_labels = []

    for video_name, vpred in video_preds.items():
        fps = vpred.get("frame_predictions", [])
        true_label = vpred.get("true_label", -1)

        # 有効な動画のみ処理
        if not fps or true_label == -1:
            continue

        # ルーティング特徴量
        rf = compute_routing_features(fps)
        routing_features.append(rf)

        # 各集約方法の正誤
        best, results = compute_best_method(fps, true_label)
        best_methods.append(best)
        method_results.append(results)
        video_names.append(video_name)
        true_labels.append(int(true_label))

    X_routing = np.array(routing_features)

    print(f"  動画数: {len(video_names)}")
    print(f"  ルーティング特徴量: {X_routing.shape}")

    # 各手法の正解数（全動画に対して）
    method_names = list(method_results[0].keys()) if method_results else []
    for method in method_names:
        correct = sum(r[method] for r in method_results)
        acc = correct / len(method_results) * 100
        print(f"  {method:<20}: {correct}/{len(method_results)} 正解 ({acc:.2f}%)")

    # 最適手法の分布（どれか1つでも正解した方法の優先上位）
    print(f"  最適手法の分布（優先順位ベース）:")
    dist = Counter(best_methods)
    for m, c in dist.most_common():
        print(f"    {m:<20}: {c} ({c/len(best_methods)*100:.1f}%)")

    return {
        "X_routing": X_routing,
        "best_methods": best_methods,
        "method_results": method_results,
        "video_names": video_names,
        "true_labels": true_labels,
    }


def main():
    print("=== Phase 6-A: ルーティング特徴量の構築 ===")

    # 出力ディレクトリ（src/ から実行する前提）
    out_dir = "../results/phase6"
    os.makedirs(out_dir, exist_ok=True)

    # ========== WetlandBirds ==========
    wb_path = "../results/bird_phase5e/frame_predictions.json"
    if os.path.exists(wb_path):
        wb_data = process_dataset(wb_path, "WetlandBirds (13種)")
        np.savez(
            os.path.join(out_dir, "routing_wetlandbirds.npz"),
            X_routing=wb_data["X_routing"],
            video_names=np.array(wb_data["video_names"]),
            true_labels=np.array(wb_data["true_labels"]),
        )
        with open(os.path.join(out_dir, "methods_wetlandbirds.json"),
                  "w", encoding="utf-8") as f:
            json.dump({
                "video_names": wb_data["video_names"],
                "best_methods": wb_data["best_methods"],
                "method_results": wb_data["method_results"],
            }, f, indent=2, ensure_ascii=False)
    else:
        print(f"  警告: {wb_path} が見つかりません")

    # ========== VB100 ==========
    vb_path = "../results/vb100_phase5e/frame_predictions.json"
    if os.path.exists(vb_path):
        vb_data = process_dataset(vb_path, "VB100 (100種)")
        np.savez(
            os.path.join(out_dir, "routing_vb100.npz"),
            X_routing=vb_data["X_routing"],
            video_names=np.array(vb_data["video_names"]),
            true_labels=np.array(vb_data["true_labels"]),
        )
        with open(os.path.join(out_dir, "methods_vb100.json"),
                  "w", encoding="utf-8") as f:
            json.dump({
                "video_names": vb_data["video_names"],
                "best_methods": vb_data["best_methods"],
                "method_results": vb_data["method_results"],
            }, f, indent=2, ensure_ascii=False)
    else:
        print(f"  警告: {vb_path} が見つかりません")

    print(f"\n保存先: {out_dir}/")


if __name__ == "__main__":
    main()
