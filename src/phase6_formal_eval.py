"""
Phase 6 正式評価 段階B: 独立した学習/テスト分割での AdaptiveFeatureRouter 評価

段階 A で生成した train/val フレーム予測を用いてルーターを学習し、
Phase 5e の test フレーム予測で評価する。
Phase 6 の K-fold CV と異なり、学習/テスト分割が完全に独立している。

制約:
  - DINOv2, YOLO, BirdNET はインポートしない（JSON のみ使用）
  - test 側のフレーム予測は Phase 5e の既存データをそのまま使う
"""

import numpy as np
import json
import os
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score


# ================================================================
# 集約関数（Phase 6 と同一）
# ================================================================

def aggregate_majority(frame_predictions):
    """多数決"""
    labels = [fp["predicted_label"] for fp in frame_predictions]
    counter = Counter(labels)
    return counter.most_common(1)[0][0]


def aggregate_frame_avg(frame_predictions):
    """フレーム平均（確率平均）"""
    probs = np.array([fp["prob_distribution"] for fp in frame_predictions])
    return int(probs.mean(axis=0).argmax())


def aggregate_conf_weighted(frame_predictions):
    """確信度加重"""
    probs = np.array([fp["prob_distribution"] for fp in frame_predictions])
    confs = np.array([fp["confidence"] for fp in frame_predictions])
    weighted = (probs.T * confs).T
    return int(weighted.sum(axis=0).argmax())


def aggregate_yolo_weighted(frame_predictions):
    """YOLO 確信度加重"""
    n_classes = len(frame_predictions[0]["prob_distribution"])
    weighted = np.zeros(n_classes)
    for fp in frame_predictions:
        p = np.array(fp["prob_distribution"])
        yc = fp.get("yolo_confidence", 0)
        is_fb = fp.get("is_fallback", False)
        w = yc if (not is_fb and yc > 0) else 0.1
        weighted += w * p
    return int(weighted.argmax())


def aggregate_sliding_window(frame_predictions, size=5):
    """スライディング窓多数決"""
    labels = [fp["predicted_label"] for fp in frame_predictions]
    if len(labels) < size:
        return aggregate_majority(frame_predictions)
    window_labels = []
    for i in range(len(labels) - size + 1):
        wc = Counter(labels[i:i + size])
        window_labels.append(wc.most_common(1)[0][0])
    counter = Counter(window_labels)
    return counter.most_common(1)[0][0]


AGGREGATION_METHODS = {
    "majority": aggregate_majority,
    "frame_avg": aggregate_frame_avg,
    "conf_weighted": aggregate_conf_weighted,
    "yolo_weighted": aggregate_yolo_weighted,
    "sliding_window": aggregate_sliding_window,
}


# ================================================================
# ルーティング特徴量（Phase 6 と同一、20次元）
# ================================================================

def compute_routing_features(frame_predictions):
    """フレーム予測からルーティング特徴量を計算する"""
    if not frame_predictions:
        return np.zeros(20, dtype=np.float32)

    n_frames = len(frame_predictions)
    predicted_labels = [fp["predicted_label"] for fp in frame_predictions]
    confidences = [fp["confidence"] for fp in frame_predictions]
    prob_distributions = [fp["prob_distribution"] for fp in frame_predictions]
    yolo_confs = [fp.get("yolo_confidence", 0) for fp in frame_predictions]
    is_fallbacks = [fp.get("is_fallback", False) for fp in frame_predictions]

    features = []

    # グループ1: 予測の一致度
    counter = Counter(predicted_labels)
    features.append(counter.most_common(1)[0][1] / n_frames)
    features.append(len(counter) / max(n_frames, 1))
    sorted_counts = sorted(counter.values(), reverse=True)
    margin = ((sorted_counts[0] - sorted_counts[1]) / n_frames
              if len(sorted_counts) >= 2 else 1.0)
    features.append(margin)
    probs_pred = np.array(sorted_counts) / n_frames
    features.append(-np.sum(probs_pred * np.log(probs_pred + 1e-9)))

    # グループ2: 確信度の統計
    conf_array = np.array(confidences)
    features.append(conf_array.mean())
    features.append(conf_array.std())
    features.append(conf_array.min())
    features.append(conf_array.max())
    features.append(np.median(conf_array))
    if conf_array.std() > 1e-9:
        skewness = ((conf_array - conf_array.mean()) ** 3).mean() / (conf_array.std() ** 3)
    else:
        skewness = 0.0
    features.append(skewness)

    # グループ3: 確率分布の特性
    prob_matrix = np.array(prob_distributions)
    mean_probs = prob_matrix.mean(axis=0)
    features.append(-np.sum(mean_probs * np.log(mean_probs + 1e-9)))
    features.append(mean_probs.max())
    sorted_mean = np.sort(mean_probs)[::-1]
    features.append(sorted_mean[0] - sorted_mean[1] if len(sorted_mean) > 1 else 1.0)
    features.append(prob_matrix.var(axis=0).mean())

    # グループ4: YOLO 検出の品質
    yolo_array = np.array(yolo_confs)
    features.append(yolo_array.mean())
    features.append(sum(is_fallbacks) / n_frames)

    # グループ5: 時間的パターン
    switches = sum(
        1 for i in range(1, len(predicted_labels))
        if predicted_labels[i] != predicted_labels[i - 1]
    )
    features.append(switches / max(n_frames - 1, 1))
    features.append(float(n_frames))
    if n_frames >= 4:
        half = n_frames // 2
        trend = conf_array[half:].mean() - conf_array[:half].mean()
    else:
        trend = 0.0
    features.append(trend)

    while len(features) < 20:
        features.append(0.0)

    return np.array(features[:20], dtype=np.float32)


def compute_best_method(fps, true_label):
    """各集約手法の正誤と、優先順位に従った最適手法を返す"""
    results = {}
    for mname, mfunc in AGGREGATION_METHODS.items():
        pred = mfunc(fps)
        results[mname] = int(pred == true_label)

    priority = ["majority", "frame_avg", "conf_weighted",
                "yolo_weighted", "sliding_window"]
    best = None
    for m in priority:
        if results.get(m) == 1:
            best = m
            break
    if best is None:
        best = "frame_avg"
    return best, results


# ================================================================
# 評価
# ================================================================

def evaluate_formal(dataset_name, train_predictions_path, test_predictions_path):
    """独立した学習/テスト分割で各ルーターを評価"""
    print(f"\n{'=' * 70}")
    print(f"データセット: {dataset_name}（正式評価）")
    print(f"{'=' * 70}")

    with open(train_predictions_path, encoding="utf-8") as f:
        train_preds = json.load(f)
    with open(test_predictions_path, encoding="utf-8") as f:
        test_preds = json.load(f)

    print(f"学習動画: {len(train_preds)}, テスト動画: {len(test_preds)}")

    # ----- 学習側: ルーティング特徴量 + 最適手法ラベル -----
    X_train = []
    train_best_methods = []
    train_method_correct_counts = Counter()

    for video_name, vpred in train_preds.items():
        fps = vpred.get("frame_predictions", [])
        true_label = vpred.get("true_label", -1)
        if not fps or true_label == -1:
            continue
        X_train.append(compute_routing_features(fps))
        best, mc = compute_best_method(fps, true_label)
        train_best_methods.append(best)
        for m, v in mc.items():
            train_method_correct_counts[m] += v

    X_train = np.array(X_train)
    n_train = len(train_best_methods)
    print(f"\n学習データの最適手法分布（優先順位ベース）:")
    for m, c in Counter(train_best_methods).most_common():
        print(f"  {m:<20}: {c} ({c / n_train * 100:.1f}%)")

    print(f"\n学習データの各手法の単独精度:")
    for m, c in train_method_correct_counts.most_common():
        print(f"  {m:<20}: {c}/{n_train} ({c / n_train * 100:.2f}%)")

    # ----- テスト側 -----
    X_test = []
    test_fps_by_name = {}
    y_test_list = []
    test_names = []

    for video_name, vpred in test_preds.items():
        fps = vpred.get("frame_predictions", [])
        true_label = vpred.get("true_label", -1)
        if not fps or true_label == -1:
            continue
        X_test.append(compute_routing_features(fps))
        test_fps_by_name[video_name] = fps
        y_test_list.append(int(true_label))
        test_names.append(video_name)

    X_test = np.array(X_test)
    y_test = np.array(y_test_list)

    results = {}

    # ----- 固定手法ベースライン -----
    print(f"\n--- 固定手法ベースライン ---")
    for method_name, method_func in AGGREGATION_METHODS.items():
        preds = [method_func(test_fps_by_name[v]) for v in test_names]
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro", zero_division=0)
        results[f"fixed_{method_name}"] = {
            "accuracy": float(acc), "macro_f1": float(f1)
        }
        print(f"  {method_name:<20}: Acc {acc * 100:.2f}%, F1 {f1:.4f}")

    # ----- 単一クラス問題のチェック -----
    unique_methods = set(train_best_methods)

    if len(unique_methods) <= 1:
        print(f"\n  警告: 学習データの最適手法が単一クラス {unique_methods}")
        single = next(iter(unique_methods)) if unique_methods else "frame_avg"
        print(f"  ルーターは常に {single} を選択します")
        preds = [AGGREGATION_METHODS[single](test_fps_by_name[v]) for v in test_names]
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro", zero_division=0)
        results["adaptive_single_class"] = {
            "accuracy": float(acc), "macro_f1": float(f1),
            "forced_method": single,
        }
        print(f"  適応（単一クラス）: Acc {acc * 100:.2f}%, F1 {f1:.4f}")
    else:
        # ----- AdaptiveFeatureRouter (LogReg) -----
        print(f"\n--- AdaptiveFeatureRouter (LogReg) ---")
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)

        le = LabelEncoder()
        y_tr = le.fit_transform(train_best_methods)

        router_lr = LogisticRegression(
            max_iter=2000, C=1.0, random_state=42, solver="lbfgs"
        )
        router_lr.fit(X_tr, y_tr)
        train_acc_lr = router_lr.score(X_tr, y_tr)
        print(f"  ルーター学習精度: {train_acc_lr * 100:.2f}%")

        chosen_lr = le.inverse_transform(router_lr.predict(X_te))
        preds_lr = [
            AGGREGATION_METHODS[chosen_lr[i]](test_fps_by_name[v])
            for i, v in enumerate(test_names)
        ]
        acc_lr = accuracy_score(y_test, preds_lr)
        f1_lr = f1_score(y_test, preds_lr, average="macro", zero_division=0)
        results["adaptive_logistic"] = {
            "accuracy": float(acc_lr), "macro_f1": float(f1_lr),
            "router_train_acc": float(train_acc_lr),
            "chosen_distribution": dict(Counter(chosen_lr.tolist())),
        }
        print(f"  Accuracy: {acc_lr * 100:.2f}%, F1: {f1_lr:.4f}")
        print(f"  選択分布: {dict(Counter(chosen_lr.tolist()))}")

        # ----- AdaptiveFeatureRouter (RandomForest) -----
        print(f"\n--- AdaptiveFeatureRouter (RandomForest) ---")
        router_rf = RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=42, n_jobs=-1
        )
        router_rf.fit(X_tr, y_tr)
        train_acc_rf = router_rf.score(X_tr, y_tr)
        print(f"  ルーター学習精度: {train_acc_rf * 100:.2f}%")

        chosen_rf = le.inverse_transform(router_rf.predict(X_te))
        preds_rf = [
            AGGREGATION_METHODS[chosen_rf[i]](test_fps_by_name[v])
            for i, v in enumerate(test_names)
        ]
        acc_rf = accuracy_score(y_test, preds_rf)
        f1_rf = f1_score(y_test, preds_rf, average="macro", zero_division=0)
        results["adaptive_random_forest"] = {
            "accuracy": float(acc_rf), "macro_f1": float(f1_rf),
            "router_train_acc": float(train_acc_rf),
            "chosen_distribution": dict(Counter(chosen_rf.tolist())),
        }
        print(f"  Accuracy: {acc_rf * 100:.2f}%, F1: {f1_rf:.4f}")
        print(f"  選択分布: {dict(Counter(chosen_rf.tolist()))}")

        # 特徴量重要度
        if hasattr(router_rf, "feature_importances_"):
            importances = router_rf.feature_importances_
            feature_names = [
                "agreement_rate", "unique_ratio", "margin", "pred_entropy",
                "conf_mean", "conf_std", "conf_min", "conf_max", "conf_median",
                "conf_skewness", "mean_prob_entropy", "mean_prob_max",
                "mean_prob_margin", "prob_variance", "yolo_mean",
                "fallback_rate", "switch_rate", "n_frames", "conf_trend",
                "reserved_20",
            ]
            top_idx = importances.argsort()[::-1][:10]
            print(f"\n  特徴量重要度 (上位10):")
            importances_list = []
            for idx in top_idx:
                name = feature_names[idx] if idx < len(feature_names) else f"f_{idx}"
                print(f"    {name:<25}: {importances[idx]:.4f}")
                importances_list.append({
                    "name": name, "importance": float(importances[idx])
                })
            results["adaptive_random_forest"]["top_feature_importances"] = importances_list

        # ----- HybridRouter (RF + フォールバック) -----
        print(f"\n--- HybridRouter (RF + フォールバック) ---")
        probs_rf = router_rf.predict_proba(X_te)
        threshold = 0.6

        hybrid_methods = []
        hybrid_preds = []
        for i, v in enumerate(test_names):
            conf = probs_rf[i].max()
            method = chosen_rf[i] if conf >= threshold else "frame_avg"
            hybrid_methods.append(method)
            hybrid_preds.append(AGGREGATION_METHODS[method](test_fps_by_name[v]))

        acc_h = accuracy_score(y_test, hybrid_preds)
        f1_h = f1_score(y_test, hybrid_preds, average="macro", zero_division=0)
        fallback = sum(1 for m in hybrid_methods if m == "frame_avg")
        results["hybrid_router"] = {
            "accuracy": float(acc_h), "macro_f1": float(f1_h),
            "chosen_distribution": dict(Counter(hybrid_methods)),
            "fallback_count": int(fallback),
            "threshold": threshold,
        }
        print(f"  Accuracy: {acc_h * 100:.2f}%, F1: {f1_h:.4f}")
        print(f"  選択分布: {dict(Counter(hybrid_methods))}")
        print(f"  フォールバック: {fallback}/{len(hybrid_methods)}")

    # ----- ヒューリスティックルーター -----
    print(f"\n--- ヒューリスティックルーター ---")
    heuristic_methods = []
    heuristic_preds = []
    for i, v in enumerate(test_names):
        rf = X_test[i]
        agreement = rf[0]
        if agreement > 0.8:
            method = "majority"
        elif agreement > 0.6:
            method = "yolo_weighted"
        else:
            method = "frame_avg"
        heuristic_methods.append(method)
        heuristic_preds.append(AGGREGATION_METHODS[method](test_fps_by_name[v]))

    acc_heu = accuracy_score(y_test, heuristic_preds)
    f1_heu = f1_score(y_test, heuristic_preds, average="macro", zero_division=0)
    results["heuristic_router"] = {
        "accuracy": float(acc_heu), "macro_f1": float(f1_heu),
        "chosen_distribution": dict(Counter(heuristic_methods)),
    }
    print(f"  Accuracy: {acc_heu * 100:.2f}%, F1: {f1_heu:.4f}")
    print(f"  選択分布: {dict(Counter(heuristic_methods))}")

    return results


def compare_with_kfold(all_results, kfold_path):
    """Phase 6 の K-fold CV 結果との比較（バイアスの影響を定量化）"""
    print(f"\n--- Phase 6 K-fold CV vs 正式評価 の比較 ---")
    if not os.path.exists(kfold_path):
        print(f"  警告: K-fold 結果が見つかりません: {kfold_path}")
        return

    with open(kfold_path, encoding="utf-8") as f:
        kfold_results = json.load(f)

    for ds_name in all_results:
        if ds_name not in kfold_results:
            continue
        print(f"\n  [{ds_name}]")
        print(f"    {'手法':<30} {'K-fold':>10} {'正式':>10} {'差分':>10}")
        print(f"    {'-' * 62}")

        # 固定手法も比較
        methods_to_compare = sorted(
            set(all_results[ds_name].keys()) & set(kfold_results[ds_name].keys())
        )
        for method in methods_to_compare:
            kf = kfold_results[ds_name].get(method, {}).get("accuracy", None)
            fm = all_results[ds_name].get(method, {}).get("accuracy", None)
            if kf is None or fm is None:
                continue
            diff = fm - kf
            print(f"    {method:<30} {kf * 100:>9.2f}% {fm * 100:>9.2f}% "
                  f"{diff * 100:>+9.2f}pp")


def main():
    print("=== Phase 6 正式評価 段階B: 独立した学習/テスト分割での評価 ===")

    out_dir = "../results/phase6_formal"
    os.makedirs(out_dir, exist_ok=True)
    all_results = {}

    # ----- WetlandBirds -----
    wb_train = os.path.join(out_dir, "train_predictions_wetlandbirds.json")
    wb_test = "../results/bird_phase5e/frame_predictions.json"
    if os.path.exists(wb_train) and os.path.exists(wb_test):
        all_results["wetlandbirds"] = evaluate_formal(
            "WetlandBirds (13種)", wb_train, wb_test
        )
    else:
        print(f"警告: WetlandBirds の入力がありません")
        if not os.path.exists(wb_train):
            print(f"  {wb_train} が見つかりません（段階A が未実行？）")

    # ----- VB100 -----
    vb_train = os.path.join(out_dir, "train_predictions_vb100.json")
    vb_test = "../results/vb100_phase5e/frame_predictions.json"
    if os.path.exists(vb_train) and os.path.exists(vb_test):
        all_results["vb100"] = evaluate_formal(
            "VB100 (100種)", vb_train, vb_test
        )
    else:
        print(f"警告: VB100 の入力がありません")
        if not os.path.exists(vb_train):
            print(f"  {vb_train} が見つかりません（段階A が未実行？）")

    # ----- 統合比較表 -----
    print(f"\n{'=' * 80}")
    print(f"Phase 6 正式評価 統合比較表")
    print(f"{'=' * 80}")

    for ds_name, ds_results in all_results.items():
        print(f"\n--- {ds_name} ---")
        print(f"{'手法':<35} {'Accuracy':>10} {'F1':>8}")
        print(f"{'-' * 57}")
        sorted_items = sorted(
            ds_results.items(),
            key=lambda x: x[1]["accuracy"],
            reverse=True,
        )
        for method_name, r in sorted_items:
            print(f"{method_name:<35} {r['accuracy'] * 100:>9.2f}% "
                  f"{r['macro_f1']:>7.4f}")

    # K-fold CV との比較
    compare_with_kfold(all_results, "../results/phase6/phase6_results.json")

    # 失敗条件
    print(f"\n--- 失敗条件の判定 ---")
    for ds_key, ds_label in [("wetlandbirds", "WetlandBirds"),
                              ("vb100", "VB100")]:
        if ds_key not in all_results:
            continue
        ds = all_results[ds_key]
        fixed_accs = [v["accuracy"] for k, v in ds.items() if k.startswith("fixed_")]
        if not fixed_accs:
            continue
        best_fixed = max(fixed_accs)
        adaptive_keys = [k for k in ds.keys()
                        if "adaptive" in k or "hybrid" in k or "heuristic" in k]
        if not adaptive_keys:
            continue
        best_adaptive = max(ds[k]["accuracy"] for k in adaptive_keys)
        best_adaptive_name = max(adaptive_keys, key=lambda k: ds[k]["accuracy"])

        if best_adaptive < best_fixed:
            print(f"  NG: {ds_label} 最良ルーター {best_adaptive_name} "
                  f"({best_adaptive * 100:.2f}%) < 固定最良 "
                  f"({best_fixed * 100:.2f}%)")
        else:
            diff = (best_adaptive - best_fixed) * 100
            print(f"  OK: {ds_label} 最良ルーター {best_adaptive_name} "
                  f"({best_adaptive * 100:.2f}%) >= 固定最良 "
                  f"({best_fixed * 100:.2f}%) [+{diff:.2f}pp]")

    # 保存
    with open(os.path.join(out_dir, "phase6_formal_results.json"),
              "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n保存: {os.path.join(out_dir, 'phase6_formal_results.json')}")


if __name__ == "__main__":
    main()
