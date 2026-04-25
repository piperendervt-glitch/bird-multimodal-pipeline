"""OOD (Out-of-Distribution) 検出による YOLO 誤検出の除去。

4 つの手法を比較:
1. マハラノビス距離
2. k-NN 距離（コサインメトリック）
3. コサイン類似度（クラスプロトタイプとの最大類似度）
4. エネルギースコア（LogReg のロジット集約）

CUB-200 を「既知の鳥」として、YouTube Great Tit 19 動画の検出結果に
OOD スコアを付与し、誤検出（枝・岩・フォールバック）を識別する。

使い方:
  cd src
  python ood_detection.py
"""

import argparse
import json
import os
import sys
import time

import numpy as np


def load_cub200_features():
    """Phase 1 で保存した CUB-200 の DINOv2 特徴量を読み込む。"""
    candidates = [
        "../results/bird_phase1/features_dinov2_vits14.npz",
        "../results/bird_phase1/features.npz",
    ]
    for path in candidates:
        if os.path.exists(path):
            data = np.load(path, allow_pickle=True)
            print(f"CUB-200 特徴量: {path}")
            print(f"  キー: {list(data.keys())}")
            return data
    return None


def extract_youtube_features():
    """YouTube 19 動画の crop 画像から DINOv2 特徴量を抽出する。"""
    crops_root = "../results/phase5g_youtube/crops"
    frame_results_path = "../results/phase5g_youtube/frame_results.json"

    if not (os.path.exists(crops_root) and os.path.exists(frame_results_path)):
        print(f"エラー: crops または frame_results が見つかりません")
        return None, None

    import torch
    import torchvision.transforms as T
    from PIL import Image

    print("DINOv2 読み込み中...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"  device: {device}")

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    with open(frame_results_path, encoding="utf-8") as f:
        frame_data = json.load(f)
    videos = frame_data.get("videos", frame_data)

    all_features = []
    all_metadata = []
    total_crops = sum(len(v.get("crops", [])) for v in videos.values())
    print(f"YouTube crop 総数: {total_crops}")

    print("特徴量抽出中（GPU バッチ）...")
    start = time.time()
    BATCH_SIZE = 64
    batch_imgs = []
    batch_meta = []

    def flush_batch():
        if not batch_imgs:
            return
        tensor = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            feats = model(tensor).cpu().numpy()
        for f, m in zip(feats, batch_meta):
            all_features.append(f)
            all_metadata.append(m)
        batch_imgs.clear()
        batch_meta.clear()

    for video_id, vinfo in videos.items():
        for crop in vinfo.get("crops", []):
            crop_path = crop.get("crop_path", "")
            if not os.path.exists(crop_path):
                continue
            try:
                img = Image.open(crop_path).convert("RGB")
                tensor = transform(img)
            except Exception:
                continue
            batch_imgs.append(tensor)
            batch_meta.append({
                "video_id": video_id,
                "crop_path": crop_path,
                "yolo_confidence": float(crop.get("confidence", 0)),
                "is_fallback": bool(crop.get("is_fallback", False)),
                "n_birds": int(crop.get("n_birds", 0)),
                "frame_idx": int(crop.get("frame_idx", 0)),
                "timestamp": float(crop.get("timestamp", 0)),
            })
            if len(batch_imgs) >= BATCH_SIZE:
                flush_batch()
                if len(all_features) % 1000 < BATCH_SIZE:
                    elapsed = time.time() - start
                    print(f"  {len(all_features)}/{total_crops} "
                          f"({elapsed:.0f}秒)")
    flush_batch()

    elapsed = time.time() - start
    X = np.array(all_features)
    print(f"  抽出完了: {len(all_features)} 枚, {elapsed:.1f} 秒")
    print(f"  特徴量: {X.shape}")
    return X, all_metadata


# ================================================================
# OOD 検出の 4 手法
# ================================================================

class MahalanobisOOD:
    """手法 1: マハラノビス距離（クラス別共分散）。"""

    def __init__(self):
        from sklearn.covariance import EmpiricalCovariance
        self._EmpCov = EmpiricalCovariance
        self.class_models = {}

    def fit(self, X, y):
        classes = np.unique(y)
        for cls in classes:
            X_cls = X[y == cls]
            if len(X_cls) < 2:
                continue
            try:
                cov = self._EmpCov().fit(X_cls)
                self.class_models[int(cls)] = cov
            except Exception:
                continue
        print(f"  マハラノビス: {len(self.class_models)} クラスのモデル構築")

    def score(self, X):
        # 全クラスとのマハラノビス距離の最小値（最も近いクラスとの距離）
        scores = []
        for x in X:
            min_dist = float("inf")
            for cov in self.class_models.values():
                try:
                    d = cov.mahalanobis(x.reshape(1, -1))[0]
                    if d < min_dist:
                        min_dist = d
                except Exception:
                    continue
            scores.append(min_dist if min_dist != float("inf") else 0.0)
        return np.array(scores)


class KNNDistanceOOD:
    """手法 2: k-NN 距離（コサインメトリック）。"""

    def __init__(self, k=5, metric="cosine"):
        from sklearn.neighbors import NearestNeighbors
        self._NN = NearestNeighbors
        self.k = k
        self.metric = metric
        self.nn = None

    def fit(self, X, y=None):
        self.nn = self._NN(n_neighbors=self.k, metric=self.metric)
        self.nn.fit(X)
        print(f"  k-NN: {len(X)} サンプルのインデックス構築 (k={self.k})")

    def score(self, X):
        distances, _ = self.nn.kneighbors(X)
        return distances.mean(axis=1)


class CosineSimilarityOOD:
    """手法 3: クラスプロトタイプとのコサイン類似度の最大値（反転して OOD 化）。"""

    def __init__(self):
        self.proto_matrix = None

    def fit(self, X, y):
        classes = np.unique(y)
        protos = []
        for cls in classes:
            X_cls = X[y == cls]
            proto = X_cls.mean(axis=0)
            proto = proto / (np.linalg.norm(proto) + 1e-9)
            protos.append(proto)
        self.proto_matrix = np.array(protos)
        print(f"  コサイン: {len(protos)} クラスのプロトタイプ構築")

    def score(self, X):
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        sims = X_norm @ self.proto_matrix.T
        max_sim = sims.max(axis=1)
        return 1.0 - max_sim  # 高い = OOD


class EnergyOOD:
    """手法 4: エネルギースコア（LogReg のロジットを集約）。"""

    def __init__(self, temperature=1.0):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        self._LR = LogisticRegression
        self._SS = StandardScaler
        self.temperature = temperature
        self.clf = None
        self.scaler = None

    def fit(self, X, y):
        self.scaler = self._SS()
        X_s = self.scaler.fit_transform(X)
        self.clf = self._LR(max_iter=2000, C=1.0, random_state=42,
                              solver="lbfgs")
        self.clf.fit(X_s, y)
        train_acc = self.clf.score(X_s, y)
        print(f"  エネルギー: LogReg 学習精度 {train_acc*100:.2f}%")

    def score(self, X):
        X_s = self.scaler.transform(X)
        logits = self.clf.decision_function(X_s)
        # 各サンプルの logsumexp（数値安定版）
        L = logits / self.temperature
        Lmax = L.max(axis=1, keepdims=True)
        lse = Lmax.squeeze(axis=1) + np.log(
            np.sum(np.exp(L - Lmax), axis=1) + 1e-9
        )
        energy = self.temperature * lse
        return -energy  # 高い = OOD


# ================================================================
# 評価
# ================================================================

def evaluate_ood(scores, metadata, threshold):
    """閾値で ID/OOD に分けたときの内訳を返す。"""
    n_total = len(scores)
    ood_mask = scores > threshold
    id_mask = ~ood_mask

    yolo_confs = np.array([m["yolo_confidence"] for m in metadata])
    is_fallback = np.array([m.get("is_fallback", False) for m in metadata])

    if ood_mask.sum() > 0:
        ood_fb_rate = float(is_fallback[ood_mask].mean())
        ood_yolo_mean = float(yolo_confs[ood_mask].mean())
    else:
        ood_fb_rate = 0.0
        ood_yolo_mean = 0.0

    if id_mask.sum() > 0:
        id_fb_rate = float(is_fallback[id_mask].mean())
        id_yolo_mean = float(yolo_confs[id_mask].mean())
    else:
        id_fb_rate = 0.0
        id_yolo_mean = 0.0

    return {
        "n_total": int(n_total),
        "n_id": int(id_mask.sum()),
        "n_ood": int(ood_mask.sum()),
        "ood_ratio": float(ood_mask.sum() / max(n_total, 1)),
        "ood_fallback_rate": ood_fb_rate,
        "ood_yolo_mean": ood_yolo_mean,
        "id_fallback_rate": id_fb_rate,
        "id_yolo_mean": id_yolo_mean,
    }


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="OOD 検出による誤検出除去")
    parser.add_argument("--threshold-sweep", action="store_true",
                         help="（互換用、現状は常に閾値スイープを実施）")
    args = parser.parse_args()

    print("=== OOD 検出: YOLO 誤検出の除去 ===")
    os.makedirs("../results/ood_detection", exist_ok=True)

    # ========================================
    # データ準備
    # ========================================
    print("\n--- CUB-200 特徴量の読み込み ---")
    cub_data = load_cub200_features()
    if cub_data is None:
        print("CUB-200 の特徴量が見つかりません")
        return

    keys = list(cub_data.keys())
    if "X_train" in keys:
        X_train = cub_data["X_train"]
        y_train = cub_data["y_train"]
    elif "X" in keys:
        X_train = cub_data["X"]
        y_train = cub_data["y"]
    else:
        X_train = cub_data[keys[0]]
        y_train = (cub_data[keys[1]] if len(keys) > 1
                    else np.zeros(len(X_train)))

    print(f"  CUB-200 学習データ: {X_train.shape}")
    print(f"  クラス数: {len(np.unique(y_train))}")

    print("\n--- YouTube 特徴量の抽出 ---")
    feat_cache = "../results/ood_detection/youtube_features.npy"
    meta_cache = "../results/ood_detection/youtube_metadata.json"

    if os.path.exists(feat_cache) and os.path.exists(meta_cache):
        print(f"  キャッシュから読み込み: {feat_cache}")
        X_youtube = np.load(feat_cache)
        with open(meta_cache, encoding="utf-8") as f:
            metadata_youtube = json.load(f)
    else:
        result = extract_youtube_features()
        if result is None or result[0] is None:
            print("YouTube 特徴量の抽出に失敗")
            return
        X_youtube, metadata_youtube = result
        np.save(feat_cache, X_youtube)
        with open(meta_cache, "w", encoding="utf-8") as f:
            json.dump(metadata_youtube, f, ensure_ascii=False)
        print(f"  キャッシュ保存: {feat_cache}, {meta_cache}")

    print(f"  YouTube テストデータ: {X_youtube.shape}")

    # ========================================
    # 4 手法の実行
    # ========================================
    methods = {}

    print("\n--- 手法 1: マハラノビス距離 ---")
    start = time.time()
    mahal = MahalanobisOOD()
    mahal.fit(X_train, y_train)
    scores_mahal = mahal.score(X_youtube)
    t1 = time.time() - start
    methods["mahalanobis"] = {"scores": scores_mahal, "time": t1}
    print(f"  スコア範囲: {scores_mahal.min():.2f} 〜 {scores_mahal.max():.2f}")
    print(f"  処理時間: {t1:.1f} 秒")

    print("\n--- 手法 2: k-NN 距離 ---")
    start = time.time()
    knn = KNNDistanceOOD(k=5, metric="cosine")
    knn.fit(X_train)
    scores_knn = knn.score(X_youtube)
    t2 = time.time() - start
    methods["knn"] = {"scores": scores_knn, "time": t2}
    print(f"  スコア範囲: {scores_knn.min():.4f} 〜 {scores_knn.max():.4f}")
    print(f"  処理時間: {t2:.1f} 秒")

    print("\n--- 手法 3: コサイン類似度 ---")
    start = time.time()
    cos = CosineSimilarityOOD()
    cos.fit(X_train, y_train)
    scores_cos = cos.score(X_youtube)
    t3 = time.time() - start
    methods["cosine"] = {"scores": scores_cos, "time": t3}
    print(f"  スコア範囲: {scores_cos.min():.4f} 〜 {scores_cos.max():.4f}")
    print(f"  処理時間: {t3:.1f} 秒")

    print("\n--- 手法 4: エネルギースコア ---")
    start = time.time()
    energy = EnergyOOD(temperature=1.0)
    energy.fit(X_train, y_train)
    scores_eng = energy.score(X_youtube)
    t4 = time.time() - start
    methods["energy"] = {"scores": scores_eng, "time": t4}
    print(f"  スコア範囲: {scores_eng.min():.2f} 〜 {scores_eng.max():.2f}")
    print(f"  処理時間: {t4:.1f} 秒")

    # ========================================
    # 閾値別の評価
    # ========================================
    print(f"\n{'='*84}")
    print(f"閾値別の評価")
    print(f"{'='*84}")

    for name, m in methods.items():
        scores = m["scores"]
        print(f"\n--- {name} ---")
        print(f"{'%ile':>6} {'閾値':>10} {'ID':>6} {'OOD':>6} "
              f"{'OOD率':>7} {'OOD fb率':>10} {'ID YOLO平均':>12}")
        print("-" * 70)
        for pct in [50, 70, 80, 90, 95, 99]:
            th = float(np.percentile(scores, pct))
            r = evaluate_ood(scores, metadata_youtube, th)
            print(f"{pct:>5}% {th:>10.4f} {r['n_id']:>6} {r['n_ood']:>6} "
                  f"{r['ood_ratio']*100:>6.1f}% "
                  f"{r['ood_fallback_rate']*100:>9.1f}% "
                  f"{r['id_yolo_mean']:>12.4f}")

    # ========================================
    # フォールバック vs 検出の OOD スコア分布
    # ========================================
    print(f"\n{'='*84}")
    print(f"フォールバック（鳥未検出）vs 鳥検出 の OOD スコア比較")
    print(f"{'='*84}")

    is_fb = np.array([m.get("is_fallback", False) for m in metadata_youtube])
    yolo_confs = np.array([m["yolo_confidence"] for m in metadata_youtube])
    print(f"\nフォールバック: {is_fb.sum()} 枚, 鳥検出: {(~is_fb).sum()} 枚")

    for name, m in methods.items():
        scores = m["scores"]
        fb = scores[is_fb] if is_fb.sum() > 0 else np.array([])
        det = scores[~is_fb]
        if len(fb) == 0:
            continue
        print(f"\n  {name}:")
        print(f"    フォールバック: 平均 {fb.mean():.4f}, "
              f"中央値 {np.median(fb):.4f}")
        print(f"    鳥検出:         平均 {det.mean():.4f}, "
              f"中央値 {np.median(det):.4f}")
        print(f"    分離度:         {fb.mean() - det.mean():+.4f}")

    # ========================================
    # YOLO 確信度 vs OOD スコアの相関
    # ========================================
    print(f"\n{'='*84}")
    print(f"YOLO 確信度 vs OOD スコアの相関")
    print(f"{'='*84}")
    for name, m in methods.items():
        scores = m["scores"]
        valid = np.isfinite(scores) & np.isfinite(yolo_confs)
        if valid.sum() > 10:
            corr = float(np.corrcoef(yolo_confs[valid], scores[valid])[0, 1])
            print(f"  {name}: 相関 {corr:+.4f}")

    # ========================================
    # YOLO 確信度帯別の OOD スコア
    # ========================================
    print(f"\n{'='*84}")
    print(f"YOLO 確信度帯別の OOD スコア（誤検出は確信度が低い側に集中するか）")
    print(f"{'='*84}")
    bins = [(0, 0.25), (0.25, 0.40), (0.40, 0.60), (0.60, 0.80), (0.80, 1.01)]
    for name in methods:
        scores = methods[name]["scores"]
        print(f"\n  {name}:")
        print(f"  {'確信度帯':>12} {'枚数':>6} {'OOD平均':>10} {'OOD中央値':>12}")
        for low, high in bins:
            mask = (yolo_confs >= low) & (yolo_confs < high)
            if mask.sum() == 0:
                continue
            s = scores[mask]
            print(f"  {low:.2f}-{high:.2f} {mask.sum():>6} "
                  f"{s.mean():>10.4f} {np.median(s):>12.4f}")

    # ========================================
    # 統合比較表（90 パーセンタイル閾値）
    # ========================================
    print(f"\n{'='*84}")
    print(f"統合比較表（90 パーセンタイル閾値）")
    print(f"{'='*84}")
    print(f"\n{'手法':<14} {'時間':>8} {'OOD数':>6} {'OOD fb率':>10} "
          f"{'ID YOLO平均':>12} {'分離度':>10}")
    print("-" * 75)

    summary = {}
    for name, m in methods.items():
        scores = m["scores"]
        th = float(np.percentile(scores, 90))
        r = evaluate_ood(scores, metadata_youtube, th)
        fb = scores[is_fb] if is_fb.sum() > 0 else np.array([0.0])
        det = scores[~is_fb]
        sep = float(fb.mean() - det.mean()) if len(fb) > 0 else 0.0
        print(f"{name:<14} {m['time']:>7.1f}s {r['n_ood']:>6} "
              f"{r['ood_fallback_rate']*100:>9.1f}% "
              f"{r['id_yolo_mean']:>12.4f} {sep:>+10.4f}")

        summary[name] = {
            "time_sec": float(m["time"]),
            "threshold_p90": th,
            "evaluation_p90": r,
            "separation_fb_minus_det": sep,
            "score_stats": {
                "mean": float(np.nanmean(scores)),
                "std": float(np.nanstd(scores)),
                "min": float(np.nanmin(scores)),
                "max": float(np.nanmax(scores)),
                "percentiles": {
                    str(p): float(np.nanpercentile(scores, p))
                    for p in [10, 25, 50, 75, 90, 95, 99]
                },
            },
        }

    # 保存
    output = {
        "n_cub200_train": int(len(X_train)),
        "n_cub200_classes": int(len(np.unique(y_train))),
        "n_youtube_test": int(len(X_youtube)),
        "n_fallback": int(is_fb.sum()),
        "n_real_detection": int((~is_fb).sum()),
        "methods": summary,
    }
    out_path = "../results/ood_detection/ood_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n保存: {out_path}")


if __name__ == "__main__":
    main()
