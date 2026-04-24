"""
Phase 6-B: AdaptiveFeatureRouter 評価

段階 A で構築したルーティング特徴量を用いて、集約方法を動的に選ぶ
ルーターを実装・評価する。

制約:
- DINOv2, YOLO, BirdNET はインポートしない（JSON/npz の再分析のみ）
- Phase 5e の frame_predictions.json は「テストセットのみ」を含むため、
  学習セットがない → K-fold CV でルーターの学習/評価を分離する
- テストセットでの「パラメータチューニング」を避けるため、
  学習不要の HeuristicRouter も併記する

評価する手法:
1. 固定手法ベースライン（majority, frame_avg, conf_weighted,
   yolo_weighted, sliding_window）
2. AdaptiveFeatureRouter (LogReg, K-fold CV)
3. AdaptiveFeatureRouter (RandomForest, K-fold CV)
4. HybridRouter (RF + 低確信度時にフォールバック)
5. HeuristicRouter (学習不要)
"""

import numpy as np
import json
import os
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score


# ================================================================
# 集約関数（Phase 5e と同一ロジック）
# ================================================================

def aggregate_majority(frame_predictions):
    """多数決集約"""
    labels = [fp["predicted_label"] for fp in frame_predictions]
    counter = Counter(labels)
    return counter.most_common(1)[0][0]


def aggregate_frame_avg(frame_predictions):
    """フレーム平均（確率分布の平均）"""
    probs = np.array([fp["prob_distribution"] for fp in frame_predictions])
    return int(probs.mean(axis=0).argmax())


def aggregate_conf_weighted(frame_predictions):
    """確信度加重"""
    probs = np.array([fp["prob_distribution"] for fp in frame_predictions])
    confs = np.array([fp["confidence"] for fp in frame_predictions])
    weighted = (probs.T * confs).T
    return int(weighted.sum(axis=0).argmax())


def aggregate_yolo_weighted(frame_predictions):
    """YOLO 確信度加重（フォールバック時は 0.1 の重み）"""
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
# ルーティング特徴量（段階 A と同一ロジック、インライン再定義）
# ================================================================

def compute_routing_features(frame_predictions):
    """段階 A と同一のルーティング特徴量計算（20次元）"""
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
    margin = (sorted_counts[0] - sorted_counts[1]) / n_frames if len(sorted_counts) >= 2 else 1.0
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


# ================================================================
# 最適手法の計算（段階 A と同一）
# ================================================================

def compute_best_method_from_fps(fps, true_label):
    """各集約方法の正誤を算出し、優先順位で最適手法を返す"""
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
# AdaptiveFeatureRouter
# ================================================================

class AdaptiveFeatureRouter:
    """
    ルーティング特徴量から「どの集約方法が最適か」を学習するルーター。
    """

    def __init__(self, method="logistic"):
        self.method = method
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self._single_class = None  # 学習データが単一クラスの場合

        if method == "logistic":
            self.router = LogisticRegression(
                max_iter=2000, C=1.0, random_state=42,
                n_jobs=-1, solver="lbfgs"
            )
        elif method == "random_forest":
            self.router = RandomForestClassifier(
                n_estimators=100, max_depth=5, random_state=42, n_jobs=-1
            )
        else:
            raise ValueError(f"未対応の method: {method}")

    def fit(self, X_routing, best_methods):
        """ルーティング特徴量 → 最適手法のマッピングを学習"""
        unique_methods = set(best_methods)

        # 単一クラスのみ → 学習できないので固定ラベルとして保持
        if len(unique_methods) == 1:
            self._single_class = next(iter(unique_methods))
            return 1.0

        self._single_class = None
        y = self.label_encoder.fit_transform(best_methods)
        X = self.scaler.fit_transform(X_routing)
        self.router.fit(X, y)
        train_acc = self.router.score(X, y)
        return train_acc

    def predict_method(self, routing_features):
        """1動画のルーティング特徴量から最適手法を予測"""
        if self._single_class is not None:
            return self._single_class
        X = self.scaler.transform(routing_features.reshape(1, -1))
        y = self.router.predict(X)[0]
        return self.label_encoder.inverse_transform([y])[0]

    def predict_with_confidence(self, routing_features):
        """手法予測と確信度を返す"""
        if self._single_class is not None:
            return self._single_class, 1.0
        X = self.scaler.transform(routing_features.reshape(1, -1))
        probs = self.router.predict_proba(X)[0]
        y = probs.argmax()
        return self.label_encoder.inverse_transform([y])[0], float(probs.max())


class HybridRouter:
    """
    AdaptiveFeatureRouter (RF) + 低確信度時のフォールバック戦略。
    確信度が閾値未満なら安全なデフォルト手法に切り替える。
    """

    def __init__(self, default_method="frame_avg", confidence_threshold=0.6):
        self.adaptive = AdaptiveFeatureRouter(method="random_forest")
        self.default_method = default_method
        self.confidence_threshold = confidence_threshold

    def fit(self, X_routing, best_methods):
        return self.adaptive.fit(X_routing, best_methods)

    def predict_method(self, routing_features):
        method, confidence = self.adaptive.predict_with_confidence(routing_features)
        if confidence >= self.confidence_threshold:
            return method
        return self.default_method


class HeuristicRouter:
    """
    学習不要のルールベースルーター（Zero-shot）。

    ルーティング特徴量の値に対する単純なしきい値で手法を選ぶ。
    ルーティング特徴量のインデックス:
      [0] agreement_rate
      [4] conf_mean
      [11] mean_prob_max
      [14] yolo_mean
      [15] fallback_rate
    """

    def __init__(self):
        pass

    def fit(self, X_routing=None, best_methods=None):
        # 学習不要、引数は無視
        return 1.0

    def predict_method(self, routing_features):
        agreement_rate = routing_features[0]
        conf_mean = routing_features[4]
        mean_prob_max = routing_features[11]
        yolo_mean = routing_features[14]
        fallback_rate = routing_features[15]

        # ルール1: フレーム予測が高い一致度 → 多数決で十分
        if agreement_rate >= 0.8:
            return "majority"

        # ルール2: 平均確率が明確 → フレーム平均
        if mean_prob_max >= 0.4:
            return "frame_avg"

        # ルール3: YOLO 検出が信頼できる → YOLO 加重
        if yolo_mean >= 0.5 and fallback_rate <= 0.3:
            return "yolo_weighted"

        # ルール4: 確信度平均が高めなら確信度加重
        if conf_mean >= 0.5:
            return "conf_weighted"

        # デフォルト: フレーム平均（保守的）
        return "frame_avg"


# ================================================================
# K-fold 評価
# ================================================================

def evaluate_dataset(dataset_name, frame_predictions_path, n_folds=5):
    """1つのデータセットで各ルーターを K-fold CV で評価"""
    print(f"\n{'=' * 70}")
    print(f"データセット: {dataset_name}")
    print(f"{'=' * 70}")

    with open(frame_predictions_path, encoding="utf-8") as f:
        video_preds = json.load(f)

    # 各動画のルーティング特徴量・最適手法を事前計算
    video_names = []
    X_routing = []
    y_true = []
    fps_by_video = {}
    best_methods_all = []

    for video_name, vpred in video_preds.items():
        fps = vpred.get("frame_predictions", [])
        true_label = vpred.get("true_label", -1)
        if not fps or true_label == -1:
            continue

        rf = compute_routing_features(fps)
        best, _ = compute_best_method_from_fps(fps, true_label)

        video_names.append(video_name)
        X_routing.append(rf)
        y_true.append(int(true_label))
        fps_by_video[video_name] = fps
        best_methods_all.append(best)

    X_routing = np.array(X_routing)
    y_true = np.array(y_true)
    n_videos = len(video_names)
    print(f"動画数: {n_videos}")

    # 全体での最適手法分布
    print(f"全動画の最適手法分布（優先順位ベース）:")
    dist = Counter(best_methods_all)
    for m, c in dist.most_common():
        print(f"  {m:<20}: {c} ({c / n_videos * 100:.1f}%)")

    results = {}

    # =========================================
    # 固定手法のベースライン
    # =========================================
    print(f"\n--- 固定手法ベースライン ---")
    for method_name, method_func in AGGREGATION_METHODS.items():
        preds = [method_func(fps_by_video[v]) for v in video_names]
        acc = accuracy_score(y_true, preds)
        f1 = f1_score(y_true, preds, average="macro", zero_division=0)
        results[f"fixed_{method_name}"] = {
            "accuracy": float(acc),
            "macro_f1": float(f1),
        }
        print(f"  {method_name:<20}: Acc {acc * 100:.2f}%, F1 {f1:.4f}")

    # =========================================
    # K-fold CV 評価（LogReg, RF, Hybrid）
    # =========================================
    # n_folds より少ない動画数の場合は調整
    actual_folds = min(n_folds, n_videos)
    if actual_folds < 2:
        print(f"\n警告: 動画数 {n_videos} < 2 のため K-fold CV 不可")
        kf = None
    else:
        kf = KFold(n_splits=actual_folds, shuffle=True, random_state=42)

    def kfold_evaluate(router_factory, label):
        """ルーターファクトリを K-fold CV で評価"""
        if kf is None:
            return None
        all_preds = np.zeros(n_videos, dtype=np.int64)
        chosen_methods_all = []
        router_train_accs = []
        feature_importances_accum = None

        for fold_idx, (tr_idx, te_idx) in enumerate(kf.split(np.arange(n_videos))):
            router = router_factory()
            X_tr = X_routing[tr_idx]
            bm_tr = [best_methods_all[i] for i in tr_idx]
            tr_acc = router.fit(X_tr, bm_tr)
            router_train_accs.append(tr_acc)

            # RF の特徴量重要度を累積
            if hasattr(router, "adaptive"):
                underlying = router.adaptive.router
            else:
                underlying = getattr(router, "router", None)
            if (underlying is not None
                    and hasattr(underlying, "feature_importances_")
                    and getattr(router, "_single_class", None) is None
                    and getattr(getattr(router, "adaptive", None),
                                "_single_class", None) is None):
                if feature_importances_accum is None:
                    feature_importances_accum = np.zeros_like(
                        underlying.feature_importances_
                    )
                feature_importances_accum += underlying.feature_importances_

            for i in te_idx:
                method = router.predict_method(X_routing[i])
                chosen_methods_all.append((i, method))
                pred = AGGREGATION_METHODS[method](fps_by_video[video_names[i]])
                all_preds[i] = pred

        acc = accuracy_score(y_true, all_preds)
        f1 = f1_score(y_true, all_preds, average="macro", zero_division=0)
        print(f"\n--- {label} (K-fold={actual_folds}) ---")
        print(f"  ルーター学習精度（平均）: {np.mean(router_train_accs) * 100:.2f}%")
        print(f"  評価 Accuracy: {acc * 100:.2f}%, F1: {f1:.4f}")
        method_dist = Counter([m for _, m in chosen_methods_all])
        print(f"  選択手法分布: {dict(method_dist)}")

        out = {
            "accuracy": float(acc),
            "macro_f1": float(f1),
            "router_train_acc_mean": float(np.mean(router_train_accs)),
            "chosen_methods_distribution": dict(method_dist),
        }
        if feature_importances_accum is not None:
            importances_mean = feature_importances_accum / actual_folds
            out["feature_importances"] = importances_mean.tolist()
        return out

    # LogReg
    r_lr = kfold_evaluate(lambda: AdaptiveFeatureRouter(method="logistic"),
                          "AdaptiveFeatureRouter (LogReg)")
    if r_lr is not None:
        results["adaptive_logistic"] = r_lr

    # RandomForest
    r_rf = kfold_evaluate(lambda: AdaptiveFeatureRouter(method="random_forest"),
                          "AdaptiveFeatureRouter (RandomForest)")
    if r_rf is not None:
        results["adaptive_random_forest"] = r_rf

        # 特徴量重要度の表示
        if "feature_importances" in r_rf:
            importances = np.array(r_rf["feature_importances"])
            feature_names = [
                "agreement_rate", "unique_ratio", "margin", "pred_entropy",
                "conf_mean", "conf_std", "conf_min", "conf_max", "conf_median",
                "conf_skewness", "mean_prob_entropy", "mean_prob_max",
                "mean_prob_margin", "prob_variance", "yolo_mean",
                "fallback_rate", "switch_rate", "n_frames", "conf_trend",
                "reserved_20",
            ]
            top_idx = importances.argsort()[::-1][:10]
            print(f"\n  ルーティング特徴量重要度 (上位10, K-fold平均):")
            for idx in top_idx:
                name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
                print(f"    {name:<25}: {importances[idx]:.4f}")

    # Hybrid
    r_hy = kfold_evaluate(
        lambda: HybridRouter(default_method="frame_avg", confidence_threshold=0.6),
        "HybridRouter (RF + フォールバック)"
    )
    if r_hy is not None:
        results["hybrid_router"] = r_hy
        # フォールバック回数
        fb_count = r_hy["chosen_methods_distribution"].get("frame_avg", 0)
        print(f"  フォールバック相当（frame_avg 選択）: {fb_count}/{n_videos}")

    # =========================================
    # HeuristicRouter（学習不要、全動画に対して評価）
    # =========================================
    print(f"\n--- HeuristicRouter (学習不要) ---")
    heu = HeuristicRouter()
    heu_preds = []
    heu_methods = []
    for i, vname in enumerate(video_names):
        method = heu.predict_method(X_routing[i])
        heu_methods.append(method)
        heu_preds.append(AGGREGATION_METHODS[method](fps_by_video[vname]))
    acc_heu = accuracy_score(y_true, heu_preds)
    f1_heu = f1_score(y_true, heu_preds, average="macro", zero_division=0)
    results["heuristic_router"] = {
        "accuracy": float(acc_heu),
        "macro_f1": float(f1_heu),
        "chosen_methods_distribution": dict(Counter(heu_methods)),
    }
    print(f"  Accuracy: {acc_heu * 100:.2f}%, F1: {f1_heu:.4f}")
    print(f"  選択手法分布: {dict(Counter(heu_methods))}")

    return results


def main():
    print("=== Phase 6-B: AdaptiveFeatureRouter 評価 ===")
    print("（K-fold CV によりテストセット内で学習/評価を分離）")

    out_dir = "../results/phase6"
    os.makedirs(out_dir, exist_ok=True)

    all_results = {}

    # ============================
    # WetlandBirds
    # ============================
    wb_path = "../results/bird_phase5e/frame_predictions.json"
    if os.path.exists(wb_path):
        wb_results = evaluate_dataset("WetlandBirds (13種)", wb_path, n_folds=5)
        all_results["wetlandbirds"] = wb_results
    else:
        print(f"\n警告: {wb_path} が見つかりません")

    # ============================
    # VB100
    # ============================
    vb_path = "../results/vb100_phase5e/frame_predictions.json"
    if os.path.exists(vb_path):
        vb_results = evaluate_dataset("VB100 (100種)", vb_path, n_folds=5)
        all_results["vb100"] = vb_results
    else:
        print(f"\n警告: {vb_path} が見つかりません")

    # ============================
    # 統合比較表
    # ============================
    print(f"\n{'=' * 80}")
    print(f"Phase 6 統合比較表")
    print(f"{'=' * 80}")

    for ds_name, ds_results in all_results.items():
        print(f"\n--- {ds_name} ---")
        print(f"{'手法':<40} {'Accuracy':>10} {'F1':>8}")
        print(f"{'-' * 60}")
        sorted_results = sorted(
            ds_results.items(),
            key=lambda x: x[1]["accuracy"],
            reverse=True,
        )
        for method_name, r in sorted_results:
            print(f"{method_name:<40} {r['accuracy'] * 100:>9.2f}% "
                  f"{r['macro_f1']:>7.4f}")

    # ============================
    # 失敗条件の判定
    # ============================
    print(f"\n--- 失敗条件の判定 ---")

    for ds_key, ds_label in [("wetlandbirds", "WetlandBirds"),
                              ("vb100", "VB100")]:
        if ds_key not in all_results:
            continue
        ds = all_results[ds_key]
        baseline = ds.get("fixed_frame_avg", {}).get("accuracy", 0)

        adaptive_keys = [k for k in ds.keys()
                         if "adaptive" in k or "hybrid" in k or "heuristic" in k]
        if not adaptive_keys:
            continue
        best_adaptive = max(ds[k]["accuracy"] for k in adaptive_keys)
        best_adaptive_name = max(adaptive_keys, key=lambda k: ds[k]["accuracy"])

        if best_adaptive < baseline:
            print(f"  NG: {ds_label} 最良ルーター {best_adaptive_name} "
                  f"({best_adaptive * 100:.2f}%) < "
                  f"フレーム平均 ({baseline * 100:.2f}%)")
        else:
            diff = (best_adaptive - baseline) * 100
            print(f"  OK: {ds_label} 最良ルーター {best_adaptive_name} "
                  f"({best_adaptive * 100:.2f}%) >= "
                  f"フレーム平均 ({baseline * 100:.2f}%) [差分 +{diff:.2f}%]")

    # ============================
    # 保存
    # ============================
    with open(os.path.join(out_dir, "phase6_results.json"),
              "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n保存: {out_dir}/phase6_results.json")


if __name__ == "__main__":
    main()
