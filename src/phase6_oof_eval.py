"""
Phase 6 修正 段階B: Out-of-Fold 予測で学習したルーターのテスト評価

段階 A で生成した OOF フレーム予測を用いてルーターを学習し、
Phase 5e の test フレーム予測で評価する。

これにより、train データにも「現実的な誤分類」が含まれ、
ルーターは有意味な学習信号を得られるはず。

3種類の評価方式（K-fold CV / 正式 / OOF）を並列比較する。

制約:
  - DINOv2 等をインポートしない（JSON のみ使用）
  - test 側は Phase 5e の既存データを使用
  - 集約関数・ルーティング特徴量は Phase 6 (phase6_adaptive_router.py) と同一
"""

import numpy as np
import json
import os
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

# Phase 6 の実装を再利用（集約関数とルーティング特徴量）
from phase6_adaptive_router import (
    AGGREGATION_METHODS,
    compute_routing_features,
)


def compute_best_method(fps, true_label):
    """各手法の正誤と、優先順位に従った最適手法を返す"""
    method_correct = {}
    for mname, mfunc in AGGREGATION_METHODS.items():
        pred = mfunc(fps)
        method_correct[mname] = int(pred == true_label)

    priority = ["majority", "frame_avg", "conf_weighted",
                "yolo_weighted", "sliding_window"]
    best = None
    for m in priority:
        if method_correct.get(m) == 1:
            best = m
            break
    if best is None:
        best = "frame_avg"
    return best, method_correct


def evaluate_oof(dataset_name, oof_predictions_path, test_predictions_path):
    """OOF 予測で学習したルーターをテストセットで評価"""
    print(f"\n{'=' * 70}")
    print(f"データセット: {dataset_name}（Out-of-Fold 評価）")
    print(f"{'=' * 70}")

    with open(oof_predictions_path, encoding="utf-8") as f:
        oof_preds = json.load(f)
    with open(test_predictions_path, encoding="utf-8") as f:
        test_preds = json.load(f)

    print(f"学習（OOF）: {len(oof_preds)} 動画")
    print(f"テスト: {len(test_preds)} 動画")

    # ----- 学習側: OOF 予測 -----
    X_train = []
    train_best_methods = []
    train_method_details = []

    for vn, vpred in oof_preds.items():
        fps = vpred.get("frame_predictions", [])
        true_label = vpred.get("true_label", -1)
        if not fps or true_label == -1:
            continue
        X_train.append(compute_routing_features(fps))
        best, mc = compute_best_method(fps, true_label)
        train_best_methods.append(best)
        train_method_details.append(mc)

    X_train = np.array(X_train)
    n_train = len(train_best_methods)

    print(f"\n学習データの最適手法分布（優先順位ベース）:")
    for m, c in Counter(train_best_methods).most_common():
        print(f"  {m:<20}: {c} ({c / n_train * 100:.1f}%)")

    print(f"\n学習データ各手法の単独精度:")
    for mname in AGGREGATION_METHODS.keys():
        correct = sum(d.get(mname, 0) for d in train_method_details)
        print(f"  {mname:<20}: {correct}/{n_train} "
              f"({correct / n_train * 100:.2f}%)")

    divergent = sum(1 for d in train_method_details
                    if len(set(d.values())) > 1)
    print(f"\n手法で結果が分岐する動画: {divergent}/{n_train} "
          f"({divergent / n_train * 100:.1f}%)")

    # ----- テスト側 -----
    X_test = []
    test_fps_by_name = {}
    test_true = []
    test_names = []

    for vn, vpred in test_preds.items():
        fps = vpred.get("frame_predictions", [])
        true_label = vpred.get("true_label", -1)
        if not fps or true_label == -1:
            continue
        X_test.append(compute_routing_features(fps))
        test_fps_by_name[vn] = fps
        test_true.append(int(true_label))
        test_names.append(vn)

    X_test = np.array(X_test)
    y_test = np.array(test_true)

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

    # ----- 単一クラスチェック -----
    unique_methods = set(train_best_methods)

    if len(unique_methods) <= 1:
        single = next(iter(unique_methods)) if unique_methods else "frame_avg"
        print(f"\n  警告: OOF でも最適手法が単一クラス {unique_methods}")
        print(f"  ルーターは常に {single} を選択します")
        preds = [AGGREGATION_METHODS[single](test_fps_by_name[v])
                 for v in test_names]
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
                name = (feature_names[idx] if idx < len(feature_names)
                        else f"f_{idx}")
                print(f"    {name:<25}: {importances[idx]:.4f}")
                importances_list.append({
                    "name": name, "importance": float(importances[idx])
                })
            results["adaptive_random_forest"]["top_feature_importances"] = \
                importances_list

        # ----- HybridRouter -----
        print(f"\n--- HybridRouter (RF + フォールバック) ---")
        probs_rf = router_rf.predict_proba(X_te)
        threshold = 0.6
        hybrid_methods = []
        for i in range(len(test_names)):
            if probs_rf[i].max() >= threshold:
                hybrid_methods.append(str(chosen_rf[i]))
            else:
                hybrid_methods.append("frame_avg")

        preds_h = [
            AGGREGATION_METHODS[hybrid_methods[i]](test_fps_by_name[v])
            for i, v in enumerate(test_names)
        ]
        acc_h = accuracy_score(y_test, preds_h)
        f1_h = f1_score(y_test, preds_h, average="macro", zero_division=0)
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
    heur_methods = []
    for i in range(len(test_names)):
        agreement = X_test[i][0]
        if agreement > 0.8:
            heur_methods.append("majority")
        elif agreement > 0.6:
            heur_methods.append("yolo_weighted")
        else:
            heur_methods.append("frame_avg")

    preds_heur = [
        AGGREGATION_METHODS[heur_methods[i]](test_fps_by_name[v])
        for i, v in enumerate(test_names)
    ]
    acc_heu = accuracy_score(y_test, preds_heur)
    f1_heu = f1_score(y_test, preds_heur, average="macro", zero_division=0)
    results["heuristic_router"] = {
        "accuracy": float(acc_heu), "macro_f1": float(f1_heu),
        "chosen_distribution": dict(Counter(heur_methods)),
    }
    print(f"  Accuracy: {acc_heu * 100:.2f}%, F1: {f1_heu:.4f}")
    print(f"  選択分布: {dict(Counter(heur_methods))}")

    return results


def main():
    print("=== Phase 6 修正 段階B: Out-of-Fold ルーター評価 ===")

    out_dir = "../results/phase6_oof"
    os.makedirs(out_dir, exist_ok=True)
    all_results = {}

    # WetlandBirds
    wb_oof = os.path.join(out_dir, "oof_predictions_wetlandbirds.json")
    wb_test = "../results/bird_phase5e/frame_predictions.json"
    if os.path.exists(wb_oof) and os.path.exists(wb_test):
        all_results["wetlandbirds"] = evaluate_oof(
            "WetlandBirds (13種)", wb_oof, wb_test
        )
    else:
        print(f"警告: WetlandBirds 入力ファイルが揃いません")
        if not os.path.exists(wb_oof):
            print(f"  {wb_oof} がありません（段階A が未実行？）")

    # VB100
    vb_oof = os.path.join(out_dir, "oof_predictions_vb100.json")
    vb_test = "../results/vb100_phase5e/frame_predictions.json"
    if os.path.exists(vb_oof) and os.path.exists(vb_test):
        all_results["vb100"] = evaluate_oof(
            "VB100 (100種)", vb_oof, vb_test
        )
    else:
        print(f"警告: VB100 入力ファイルが揃いません")

    # ----- 統合比較表 -----
    print(f"\n{'=' * 80}")
    print(f"Phase 6 OOF 評価 統合比較表")
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

    # ----- 3種類の Phase 6 評価を並列比較 -----
    print(f"\n{'=' * 80}")
    print(f"Phase 6 全評価の比較（K-fold CV / 正式評価 / OOF 評価）")
    print(f"{'=' * 80}")

    kfold_path = "../results/phase6/phase6_results.json"
    formal_path = "../results/phase6_formal/phase6_formal_results.json"

    previous = {}
    if os.path.exists(kfold_path):
        with open(kfold_path, encoding="utf-8") as f:
            previous["kfold"] = json.load(f)
    if os.path.exists(formal_path):
        with open(formal_path, encoding="utf-8") as f:
            previous["formal"] = json.load(f)

    methods_to_compare = [
        "fixed_majority", "fixed_frame_avg", "fixed_yolo_weighted",
        "adaptive_logistic", "adaptive_random_forest",
        "hybrid_router", "heuristic_router", "adaptive_single_class",
    ]

    for ds_name in all_results:
        print(f"\n--- {ds_name} ---")
        print(f"  {'手法':<30} {'K-fold':>10} {'正式':>10} {'OOF':>10}")
        print(f"  {'-' * 62}")
        for method in methods_to_compare:
            kf = previous.get("kfold", {}).get(ds_name, {}).get(method, {}).get("accuracy")
            fm = previous.get("formal", {}).get(ds_name, {}).get(method, {}).get("accuracy")
            oof = all_results.get(ds_name, {}).get(method, {}).get("accuracy")

            kf_str = f"{kf * 100:.2f}%" if kf is not None else "N/A"
            fm_str = f"{fm * 100:.2f}%" if fm is not None else "N/A"
            oof_str = f"{oof * 100:.2f}%" if oof is not None else "N/A"

            # いずれかの評価で出てくる手法のみ表示
            if kf is None and fm is None and oof is None:
                continue
            print(f"  {method:<30} {kf_str:>10} {fm_str:>10} {oof_str:>10}")

    # ----- 失敗条件 -----
    print(f"\n--- 失敗条件の判定 ---")
    for ds_name, ds_results in all_results.items():
        fixed_accs = [v["accuracy"] for k, v in ds_results.items()
                      if k.startswith("fixed_")]
        if not fixed_accs:
            continue
        best_fixed = max(fixed_accs)
        adaptive_keys = [k for k in ds_results.keys()
                        if "adaptive" in k or "hybrid" in k or "heuristic" in k]
        if not adaptive_keys:
            continue
        best_adaptive = max(ds_results[k]["accuracy"] for k in adaptive_keys)
        best_name = max(adaptive_keys, key=lambda k: ds_results[k]["accuracy"])

        if best_adaptive < best_fixed:
            print(f"  NG: {ds_name} 適応 {best_name} "
                  f"({best_adaptive * 100:.2f}%) < "
                  f"固定最良 ({best_fixed * 100:.2f}%)")
        elif best_adaptive > best_fixed:
            diff = (best_adaptive - best_fixed) * 100
            print(f"  OK: {ds_name} 適応 {best_name} "
                  f"({best_adaptive * 100:.2f}%) > "
                  f"固定最良 ({best_fixed * 100:.2f}%) [+{diff:.2f}pp]")
        else:
            print(f"  OK（同点）: {ds_name} 適応 {best_name} = "
                  f"固定最良 ({best_fixed * 100:.2f}%)")

    # 保存
    with open(os.path.join(out_dir, "phase6_oof_results.json"),
              "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n保存: {os.path.join(out_dir, 'phase6_oof_results.json')}")


if __name__ == "__main__":
    main()
