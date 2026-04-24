"""VB100 Phase 5e 段階B: DINOv2 特徴抽出 + フレーム単位分類器 + 予測保存。

Phase 5e (bird_phase5e_frame_predict.py) のロジックを VB100 向けに移植。

- 種ごとの層化分割 (80% 学習 / 20% テスト、seed=42) を自前で作成
- DINOv2 ViT-S/14 で切り抜き画像の特徴量 (384 次元) を抽出
- 学習セット全フレームで LogReg (class_weight="balanced") を学習
- テストセットのフレーム単位で予測・確信度・確率分布を保存
- 加えて動画特徴量平均 + LogReg（Phase 4b 方式）のベースラインも計算し保存
"""

import numpy as np
import json
import os
import sys
import io
import time
from collections import Counter, defaultdict
import random

from PIL import Image
import torch
import torchvision.transforms as T
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

# Windows cp932 対策
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


FRAME_RESULTS_PATH = os.path.join("..", "results", "vb100_phase5e", "frame_results.json")
OUT_DIR = os.path.join("..", "results", "vb100_phase5e")

SEED = 42
TRAIN_RATIO = 0.80
DINOV2_DIM = 384


def load_dinov2():
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def get_transform():
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def extract_frame_features(model, transform, crops, device, batch_size=32):
    """crops リストからフレーム単位の特徴量を抽出（crop_path 存在フレームのみ）。
    戻り値: (features ndarray [n, D], valid_crops list)"""
    valid_crops = []
    tensors = []

    for crop in crops:
        crop_path = crop.get("crop_path", "")
        if not os.path.exists(crop_path):
            continue
        try:
            img = Image.open(crop_path).convert("RGB")
            tensors.append(transform(img))
            valid_crops.append(crop)
        except Exception:
            continue

    if not tensors:
        return np.zeros((0, DINOV2_DIM), dtype=np.float32), []

    feats = []
    for i in range(0, len(tensors), batch_size):
        batch = torch.stack(tensors[i:i + batch_size]).to(device)
        with torch.no_grad():
            f = model(batch).cpu().numpy()
        feats.append(f)
    feats = np.concatenate(feats, axis=0)
    return feats, valid_crops


def stratified_split(videos_by_species, train_ratio=TRAIN_RATIO, seed=SEED):
    """種ごとに動画を 80/20 に層化分割。
    各種で少なくとも 1 本はテストに回す（総数 >= 2 の場合）。
    """
    rng = random.Random(seed)
    train_keys, test_keys = [], []
    for species in sorted(videos_by_species.keys()):
        keys = sorted(videos_by_species[species])
        rng.shuffle(keys)
        n = len(keys)
        if n <= 1:
            # 1 本しかない種は学習のみに割り当てる
            train_keys.extend(keys)
            continue
        n_train = int(round(n * train_ratio))
        # 端数調整: 学習が 0 にならないように最低 1、テストも最低 1 確保
        n_train = max(1, min(n - 1, n_train))
        train_keys.extend(keys[:n_train])
        test_keys.extend(keys[n_train:])
    return sorted(train_keys), sorted(test_keys)


def main():
    print("=== VB100 Phase 5e 段階B: DINOv2 特徴抽出 + 分類 ===")

    with open(FRAME_RESULTS_PATH, encoding="utf-8") as f:
        frame_results_raw = json.load(f)
    frame_videos = frame_results_raw["videos"]
    print(f"読み込んだ動画数: {len(frame_videos)}")

    # 種 ID マッピング (アルファベット順で決定的に)
    species_set = sorted({v["species"] for v in frame_videos.values()})
    species_to_id = {s: i for i, s in enumerate(species_set)}
    id_to_species = {str(i): s for i, s in enumerate(species_set)}
    n_classes = len(species_set)
    print(f"種数 (クラス数): {n_classes}")

    # 種ごとに動画キーをまとめる
    videos_by_species = defaultdict(list)
    for key, v in frame_videos.items():
        videos_by_species[v["species"]].append(key)

    # 層化分割
    train_keys, test_keys = stratified_split(videos_by_species,
                                             train_ratio=TRAIN_RATIO, seed=SEED)
    print(f"学習動画: {len(train_keys)} 本")
    print(f"テスト動画: {len(test_keys)} 本")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "splits.json"), "w", encoding="utf-8") as f:
        json.dump({
            "train": train_keys,
            "test": test_keys,
            "seed": SEED,
            "train_ratio": TRAIN_RATIO,
            "n_train_videos": len(train_keys),
            "n_test_videos": len(test_keys),
        }, f, indent=2, ensure_ascii=False)

    with open(os.path.join(OUT_DIR, "species_mapping.json"), "w",
              encoding="utf-8") as f:
        json.dump({
            "species_to_id": species_to_id,
            "id_to_species": id_to_species,
            "n_classes": n_classes,
        }, f, indent=2, ensure_ascii=False)

    # DINOv2 読み込み
    print("\nDINOv2 読み込み中...")
    model = load_dinov2()
    transform = get_transform()
    device = next(model.parameters()).device
    print(f"  device: {device}")

    # ========================================
    # 学習セット: 全フレームの特徴量とラベル
    # ========================================
    print("\n--- 学習セットのフレーム特徴抽出 ---")
    train_frame_features = []
    train_frame_labels = []
    # 動画ごとの特徴も保持（Phase 4b ベースライン用）
    train_video_mean_features = []
    train_video_labels = []

    start = time.time()
    for i, key in enumerate(train_keys):
        vinfo = frame_videos[key]
        species = vinfo["species"]
        label = species_to_id.get(species, -1)
        if label == -1:
            continue

        crops = sorted(vinfo.get("crops", []),
                       key=lambda c: c.get("frame_idx", 0))
        feats, _ = extract_frame_features(model, transform, crops, device)
        if feats.shape[0] == 0:
            continue
        for j in range(feats.shape[0]):
            train_frame_features.append(feats[j])
            train_frame_labels.append(label)

        # 動画単位の平均
        train_video_mean_features.append(feats.mean(axis=0))
        train_video_labels.append(label)

        if (i + 1) % 50 == 0 or (i + 1) == len(train_keys):
            print(f"  {i+1}/{len(train_keys)} 動画処理完了 "
                  f"(累計フレーム: {len(train_frame_features)}, "
                  f"経過: {time.time()-start:.1f}秒)")

    elapsed = time.time() - start
    print(f"  学習フレーム数: {len(train_frame_features)} ({elapsed:.1f}秒)")

    X_train = np.array(train_frame_features, dtype=np.float32)
    y_train = np.array(train_frame_labels, dtype=np.int64)

    # ==========
    # フレーム分類器
    # ==========
    print("\nフレーム分類器 (LogReg) を学習中...")
    t0 = time.time()
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED,
                             class_weight="balanced")
    clf.fit(X_train_s, y_train)
    train_acc = clf.score(X_train_s, y_train)
    print(f"  学習セット精度 (フレーム単位): {train_acc*100:.2f}% "
          f"({time.time()-t0:.1f}秒)")

    # ==========
    # Phase 4b ベースライン: 動画単位の平均特徴 + 別 LogReg
    # ==========
    print("\nPhase 4b ベースライン (動画特徴量平均 + LogReg) を学習中...")
    t0 = time.time()
    X_vtrain = np.array(train_video_mean_features, dtype=np.float32)
    y_vtrain = np.array(train_video_labels, dtype=np.int64)
    vscaler = StandardScaler()
    X_vtrain_s = vscaler.fit_transform(X_vtrain)
    vclf = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED,
                              class_weight="balanced")
    vclf.fit(X_vtrain_s, y_vtrain)
    vtrain_acc = vclf.score(X_vtrain_s, y_vtrain)
    print(f"  学習セット精度 (動画単位): {vtrain_acc*100:.2f}% "
          f"({time.time()-t0:.1f}秒)")

    # ==========
    # テストセット: フレーム単位予測 + 動画単位ベースライン
    # ==========
    print("\n--- テストセットの予測 ---")
    video_frame_predictions = {}
    total_frames = 0

    # Phase 4b ベースライン評価
    y_baseline_true = []
    y_baseline_pred = []

    start = time.time()
    for idx, key in enumerate(test_keys):
        vinfo = frame_videos[key]
        species = vinfo["species"]
        true_label = species_to_id.get(species, -1)

        crops = sorted(vinfo.get("crops", []),
                       key=lambda c: c.get("frame_idx", 0))
        feats, valid_crops = extract_frame_features(model, transform, crops,
                                                    device)

        frame_preds = []
        if feats.shape[0] > 0:
            feats_s = scaler.transform(feats)
            preds = clf.predict(feats_s)
            probs = clf.predict_proba(feats_s)
            for j, crop in enumerate(valid_crops):
                frame_preds.append({
                    "timestamp": crop.get("timestamp", 0),
                    "frame_idx": crop.get("frame_idx", 0),
                    "predicted_label": int(preds[j]),
                    "predicted_species": id_to_species[str(int(preds[j]))],
                    "confidence": float(probs[j].max()),
                    "prob_distribution": probs[j].tolist(),
                    "is_fallback": bool(crop.get("is_fallback", False)),
                    "yolo_confidence": float(crop.get("confidence", 0)),
                })

            # Phase 4b ベースライン: 動画平均特徴で予測
            v_feat = feats.mean(axis=0, keepdims=True)
            v_feat_s = vscaler.transform(v_feat)
            base_pred = int(vclf.predict(v_feat_s)[0])
            y_baseline_true.append(true_label)
            y_baseline_pred.append(base_pred)

        total_frames += len(frame_preds)
        video_frame_predictions[key] = {
            "species": species,
            "true_label": int(true_label),
            "n_frames": len(frame_preds),
            "frame_predictions": frame_preds,
        }

        if (idx + 1) % 50 == 0 or (idx + 1) == len(test_keys):
            print(f"  {idx+1}/{len(test_keys)} テスト動画処理完了 "
                  f"(累計フレーム: {total_frames}, "
                  f"経過: {time.time()-start:.1f}秒)")

    elapsed = time.time() - start
    print(f"  テスト動画: {len(test_keys)} 本")
    print(f"  テストフレーム数: {total_frames}")
    print(f"  処理時間: {elapsed:.1f}秒")

    # Phase 4b ベースライン精度
    baseline_acc = accuracy_score(y_baseline_true, y_baseline_pred)
    baseline_f1 = f1_score(y_baseline_true, y_baseline_pred,
                           average="macro", zero_division=0)
    baseline_correct = sum(1 for t, p in zip(y_baseline_true, y_baseline_pred)
                           if t == p)
    print(f"\n[ベースライン] 動画特徴量平均 + LogReg (Phase 4b 方式)")
    print(f"  Accuracy: {baseline_acc*100:.2f}%  "
          f"F1: {baseline_f1:.4f}  "
          f"正解数: {baseline_correct}/{len(y_baseline_true)}")

    # 保存
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "frame_predictions.json"), "w",
              encoding="utf-8") as f:
        json.dump(video_frame_predictions, f, indent=2, ensure_ascii=False)
    print(f"\n保存: results/vb100_phase5e/frame_predictions.json")

    with open(os.path.join(OUT_DIR, "phase4b_baseline.json"), "w",
              encoding="utf-8") as f:
        json.dump({
            "description": "動画特徴量平均 + LogReg (Phase 4b 方式)",
            "accuracy": float(baseline_acc),
            "macro_f1": float(baseline_f1),
            "n_correct": int(baseline_correct),
            "n_total": int(len(y_baseline_true)),
            "train_accuracy_frame_level": float(train_acc),
            "train_accuracy_video_level": float(vtrain_acc),
            "n_train_videos": int(len(train_keys)),
            "n_test_videos": int(len(test_keys)),
            "n_train_frames": int(len(train_frame_features)),
            "n_test_frames": int(total_frames),
            "seed": SEED,
            "train_ratio": TRAIN_RATIO,
        }, f, indent=2, ensure_ascii=False)
    print(f"保存: results/vb100_phase5e/phase4b_baseline.json")


if __name__ == "__main__":
    main()
