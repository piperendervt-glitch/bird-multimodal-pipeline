"""
Phase 6 修正 段階A: Out-of-Fold フレーム予測生成

Phase 6 正式評価ではフレーム分類器が train データで 99.7〜100% の精度を
達成してしまい、ルーターの学習信号が消失した。この修正では K-fold (K=5)
で分類器を再学習し、各 fold の hold-out 動画で予測を生成することで、
train データにも「現実的な誤分類」を含ませる。

入力:
  - results/bird_phase4b/frame_results.json （{"summary":..., "videos":{...}} 構造）
  - results/vb100_phase5e/frame_results.json
  - 各 splits.json, species_mapping.json

出力:
  - results/phase6_oof/oof_predictions_wetlandbirds.json
  - results/phase6_oof/oof_predictions_vb100.json
  - results/phase6_oof/oof_stats.json

制約:
  - test 動画は一切含めない（Phase 5e の frame_predictions.json は別途利用）
  - DINOv2 推論は train/val 動画の crops のみ
"""

import numpy as np
import json
import os
import time
from collections import Counter
from PIL import Image
import torch
import torchvision.transforms as T
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold


# ================================================================
# DINOv2
# ================================================================

def load_dinov2():
    """DINOv2 ViT-S/14 モデルを読み込む"""
    print("DINOv2 (vits14) 読み込み中...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print(f"  デバイス: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        print(f"  デバイス: CPU")
    return model


def get_transform():
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


# ================================================================
# データ分割
# ================================================================

def resolve_video_splits(videos_dict, splits, dataset_kind):
    """
    frame_results.videos のキー（動画名）を splits.json とマッチングして
    train/val/test 分割を判定する。
    """
    all_video_names = list(videos_dict.keys())
    video_split = {}

    if dataset_kind == "wetlandbirds":
        basename_to_full = {v.replace(".mp4", ""): v for v in all_video_names}
        for split_name, vid_list in splits.items():
            for v_id in vid_list:
                v_str = str(v_id)
                if v_str in basename_to_full:
                    video_split[basename_to_full[v_str]] = split_name
    elif dataset_kind == "vb100":
        vn_set = set(all_video_names)
        for split_name, vid_list in splits.items():
            if not isinstance(vid_list, list):
                continue
            for v_id in vid_list:
                if v_id in vn_set:
                    video_split[v_id] = split_name
    else:
        raise ValueError(f"未知の dataset_kind: {dataset_kind}")

    return video_split


# ================================================================
# 特徴抽出（バッチ推論）
# ================================================================

def extract_features_for_crops(crops, model, transform, device, batch_size=32):
    """crops リストから DINOv2 特徴量と対応メタ情報を抽出する"""
    features = []
    metas = []
    batch_tensors = []
    batch_metas = []

    def flush():
        if not batch_tensors:
            return
        batch = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            feats = model(batch).cpu().numpy()
        for i, m in enumerate(batch_metas):
            features.append(feats[i])
            metas.append(m)
        batch_tensors.clear()
        batch_metas.clear()

    for crop in crops:
        crop_path = crop.get("crop_path", "")
        if not os.path.exists(crop_path):
            continue
        try:
            img = Image.open(crop_path).convert("RGB")
            tensor = transform(img)
        except Exception:
            continue
        batch_tensors.append(tensor)
        batch_metas.append(crop)
        if len(batch_tensors) >= batch_size:
            flush()
    flush()

    return features, metas


# ================================================================
# 動画ごとの予測
# ================================================================

def predict_frames(clf, scaler, feats_arr, metas, id_to_species):
    """学習済みフレーム分類器で予測を生成"""
    X_s = scaler.transform(feats_arr)
    preds = clf.predict(X_s)
    probs = clf.predict_proba(X_s)

    frame_predictions = []
    for i, crop in enumerate(metas):
        frame_predictions.append({
            "timestamp": crop.get("timestamp", 0),
            "frame_idx": crop.get("frame_idx", 0),
            "predicted_label": int(preds[i]),
            "predicted_species": id_to_species[str(int(preds[i]))],
            "confidence": float(probs[i].max()),
            "prob_distribution": probs[i].tolist(),
            "is_fallback": crop.get("is_fallback", False),
            "yolo_confidence": crop.get("confidence", 0),
        })
    return frame_predictions


# ================================================================
# 集約関数（分岐統計のみに使用）
# ================================================================

def _agg_majority(fps):
    return Counter([f["predicted_label"] for f in fps]).most_common(1)[0][0]


def _agg_frame_avg(fps):
    probs = np.array([f["prob_distribution"] for f in fps])
    return int(probs.mean(axis=0).argmax())


def _agg_conf_weighted(fps):
    probs = np.array([f["prob_distribution"] for f in fps])
    confs = np.array([f["confidence"] for f in fps])
    return int((probs.T * confs).T.sum(axis=0).argmax())


def _agg_yolo_weighted(fps):
    n_classes = len(fps[0]["prob_distribution"])
    w = np.zeros(n_classes)
    for f in fps:
        p = np.array(f["prob_distribution"])
        yc = f.get("yolo_confidence", 0)
        is_fb = f.get("is_fallback", False)
        weight = yc if (not is_fb and yc > 0) else 0.1
        w += weight * p
    return int(w.argmax())


def _agg_sliding_window(fps, size=5):
    labels = [f["predicted_label"] for f in fps]
    if len(labels) < size:
        return _agg_majority(fps)
    window_labels = []
    for i in range(len(labels) - size + 1):
        window_labels.append(Counter(labels[i:i + size]).most_common(1)[0][0])
    return Counter(window_labels).most_common(1)[0][0]


AGG_METHODS = {
    "majority": _agg_majority,
    "frame_avg": _agg_frame_avg,
    "conf_weighted": _agg_conf_weighted,
    "yolo_weighted": _agg_yolo_weighted,
    "sliding_window": _agg_sliding_window,
}


# ================================================================
# データセット処理
# ================================================================

def process_dataset_oof(dataset_kind, dataset_name, frame_results_path,
                        splits_path, species_mapping_path, output_path,
                        model, transform, device, n_folds=5):
    """1データセットの out-of-fold 予測を生成"""
    print(f"\n{'=' * 70}")
    print(f"データセット: {dataset_name} (Out-of-Fold, K={n_folds})")
    print(f"{'=' * 70}")

    with open(frame_results_path, encoding="utf-8") as f:
        fdata = json.load(f)
    videos_dict = fdata["videos"]

    with open(splits_path, encoding="utf-8") as f:
        splits = json.load(f)

    with open(species_mapping_path, encoding="utf-8") as f:
        mapping = json.load(f)

    species_to_id = mapping["species_to_id"]
    id_to_species = mapping["id_to_species"]

    video_split = resolve_video_splits(videos_dict, splits, dataset_kind)
    train_videos = [
        v for v, s in video_split.items()
        if "train" in s.lower() or "val" in s.lower()
    ]
    print(f"train/val 動画: {len(train_videos)}")

    # === 全 train/val 動画の特徴量を事前抽出 ===
    print(f"\n全 train/val 動画の DINOv2 特徴量抽出...")
    start = time.time()

    # video_data: {video_name: (features_array, label, metas)}
    video_data = {}
    for i, vn in enumerate(train_videos):
        vinfo = videos_dict[vn]
        species = vinfo["species"]
        label = species_to_id.get(species, -1)
        if label == -1:
            continue
        crops = vinfo.get("crops", [])
        feats, metas = extract_features_for_crops(
            crops, model, transform, device
        )
        if feats:
            video_data[vn] = (np.array(feats, dtype=np.float32), int(label), metas)

        if (i + 1) % 50 == 0 or (i + 1) == len(train_videos):
            elapsed = time.time() - start
            print(f"  {i + 1}/{len(train_videos)} 動画処理済み "
                  f"(累計 {elapsed:.1f}秒)")

    elapsed_feat = time.time() - start
    total_frames = sum(d[0].shape[0] for d in video_data.values())
    print(f"  完了: {len(video_data)} 動画, {total_frames} フレーム "
          f"({elapsed_feat:.1f}秒)")

    # === K-fold で out-of-fold 予測を生成 ===
    print(f"\nOut-of-fold 予測生成 (K={n_folds})...")
    start = time.time()

    video_names_list = list(video_data.keys())
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    oof_predictions = {}
    fold_train_accs = []

    for fold_idx, (tr_idx, ho_idx) in enumerate(kf.split(video_names_list)):
        fold_train_vids = [video_names_list[i] for i in tr_idx]
        fold_holdout_vids = [video_names_list[i] for i in ho_idx]

        # Fold の学習フレームを連結
        X_parts = []
        y_parts = []
        for vn in fold_train_vids:
            feats, label, _ = video_data[vn]
            X_parts.append(feats)
            y_parts.extend([label] * feats.shape[0])
        X_fold = np.concatenate(X_parts, axis=0)
        y_fold = np.array(y_parts, dtype=np.int64)

        scaler = StandardScaler()
        X_fold_s = scaler.fit_transform(X_fold)

        clf = LogisticRegression(
            max_iter=2000, C=1.0, random_state=42, solver="lbfgs"
        )
        clf.fit(X_fold_s, y_fold)
        fold_acc = clf.score(X_fold_s, y_fold)
        fold_train_accs.append(fold_acc)

        # Holdout の動画でフレーム予測
        for vn in fold_holdout_vids:
            feats, label, metas = video_data[vn]
            species = videos_dict[vn]["species"]
            frame_preds = predict_frames(clf, scaler, feats, metas, id_to_species)
            oof_predictions[vn] = {
                "species": species,
                "true_label": int(label),
                "n_frames": len(frame_preds),
                "frame_predictions": frame_preds,
                "fold": fold_idx,
            }

        print(f"  Fold {fold_idx + 1}: train {len(fold_train_vids)} 動画, "
              f"holdout {len(fold_holdout_vids)} 動画, "
              f"train精度 {fold_acc * 100:.2f}%")

    elapsed_oof = time.time() - start
    print(f"  OOF 予測完了 ({elapsed_oof:.1f}秒)")

    # === 統計 ===
    print(f"\n--- Out-of-fold 予測の統計 ---")
    print(f"予測動画数: {len(oof_predictions)}")

    agreements = []
    majority_correct = 0
    for vn, vpred in oof_predictions.items():
        fps = vpred["frame_predictions"]
        labels = [fp["predicted_label"] for fp in fps]
        true_label = vpred["true_label"]
        counter = Counter(labels)
        agreement = counter.most_common(1)[0][1] / len(labels)
        agreements.append(agreement)
        majority_pred = counter.most_common(1)[0][0]
        if majority_pred == true_label:
            majority_correct += 1

    print(f"多数決正解率: {majority_correct}/{len(oof_predictions)} "
          f"({majority_correct / len(oof_predictions) * 100:.2f}%)")
    print(f"フレーム一致率: 平均 {np.mean(agreements) * 100:.1f}%, "
          f"最小 {np.min(agreements) * 100:.1f}%, "
          f"最大 {np.max(agreements) * 100:.1f}%")

    # 手法間の分岐
    method_divergent = 0
    all_correct = 0
    all_wrong = 0
    for vn, vpred in oof_predictions.items():
        fps = vpred["frame_predictions"]
        true_label = vpred["true_label"]
        results = {
            m: int(fn(fps) == true_label) for m, fn in AGG_METHODS.items()
        }
        values = set(results.values())
        if values == {1}:
            all_correct += 1
        elif values == {0}:
            all_wrong += 1
        else:
            method_divergent += 1

    print(f"\n手法間の分岐:")
    n_total = len(oof_predictions)
    print(f"  全手法正解: {all_correct} ({all_correct / n_total * 100:.1f}%)")
    print(f"  全手法不正解: {all_wrong} ({all_wrong / n_total * 100:.1f}%)")
    print(f"  手法で分岐: {method_divergent} ({method_divergent / n_total * 100:.1f}%)")

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(oof_predictions, f, indent=2, ensure_ascii=False)
    print(f"\n保存: {output_path}")

    return {
        "n_videos": len(oof_predictions),
        "n_total_frames": int(total_frames),
        "fold_train_acc_mean": float(np.mean(fold_train_accs)),
        "fold_train_acc_min": float(np.min(fold_train_accs)),
        "fold_train_acc_max": float(np.max(fold_train_accs)),
        "majority_accuracy": majority_correct / n_total,
        "mean_agreement": float(np.mean(agreements)),
        "all_correct": all_correct,
        "all_wrong": all_wrong,
        "method_divergent": method_divergent,
        "elapsed_feat_sec": float(elapsed_feat),
        "elapsed_oof_sec": float(elapsed_oof),
    }


def main():
    print("=== Phase 6 修正 段階A: Out-of-Fold フレーム予測 ===")
    total_start = time.time()

    model = load_dinov2()
    transform = get_transform()
    device = next(model.parameters()).device

    out_dir = "../results/phase6_oof"
    os.makedirs(out_dir, exist_ok=True)

    stats = {}

    # ----- WetlandBirds -----
    wb = {
        "dataset_kind": "wetlandbirds",
        "dataset_name": "WetlandBirds (13種)",
        "frame_results_path": "../results/bird_phase4b/frame_results.json",
        "splits_path": "../data/wetlandbirds/splits.json",
        "species_mapping_path": "../results/bird_phase4b/species_mapping.json",
        "output_path": os.path.join(out_dir, "oof_predictions_wetlandbirds.json"),
    }
    if all(os.path.exists(p) for p in [wb["frame_results_path"],
                                        wb["splits_path"],
                                        wb["species_mapping_path"]]):
        stats["wetlandbirds"] = process_dataset_oof(
            **wb, model=model, transform=transform, device=device
        )
    else:
        print("警告: WetlandBirds の入力ファイルが揃いません")

    # ----- VB100 -----
    vb = {
        "dataset_kind": "vb100",
        "dataset_name": "VB100 (100種)",
        "frame_results_path": "../results/vb100_phase5e/frame_results.json",
        "splits_path": "../results/vb100_phase5e/splits.json",
        "species_mapping_path": "../results/vb100_phase5e/species_mapping.json",
        "output_path": os.path.join(out_dir, "oof_predictions_vb100.json"),
    }
    if all(os.path.exists(p) for p in [vb["frame_results_path"],
                                        vb["splits_path"],
                                        vb["species_mapping_path"]]):
        stats["vb100"] = process_dataset_oof(
            **vb, model=model, transform=transform, device=device
        )
    else:
        print("警告: VB100 の入力ファイルが揃いません")

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"段階A 完了（合計 {total_elapsed:.1f}秒 = {total_elapsed / 60:.1f}分）")
    print(f"{'=' * 70}")

    with open(os.path.join(out_dir, "oof_stats.json"),
              "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # 学習信号確保の判定
    print(f"\n--- 失敗条件: 学習信号は確保されたか ---")
    for ds, s in stats.items():
        n_total = s["n_videos"]
        if n_total == 0:
            continue
        divergent_ratio = s["method_divergent"] / n_total
        if s["all_correct"] / n_total > 0.95:
            print(f"  {ds}: 全手法正解 {s['all_correct'] / n_total * 100:.1f}% > 95%")
            print(f"    → 学習信号がまだ不足している可能性")
        else:
            print(f"  {ds}: 手法で分岐 {s['method_divergent']}/{n_total} "
                  f"({divergent_ratio * 100:.1f}%)")
            print(f"    → ルーターの学習信号が確保された")


if __name__ == "__main__":
    main()
