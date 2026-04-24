"""
Phase 6 正式評価 段階A: train/val 動画のフレーム単位予測生成

Phase 5e の bird_phase5e_frame_predict.py と同等のロジックで、
train/val 動画のフレーム単位予測を生成する。

入力:
  - results/bird_phase4b/frame_results.json
    （WetlandBirds 178動画の crops 情報、{"summary":..., "videos":{...}} 構造）
  - results/vb100_phase5e/frame_results.json
    （VB100 1416動画の crops 情報、{"summary":..., "videos":{...}} 構造）
  - 各 species_mapping.json, splits.json

出力:
  - results/phase6_formal/train_predictions_wetlandbirds.json
  - results/phase6_formal/train_predictions_vb100.json

  （Phase 5e の frame_predictions.json と同じスキーマ）

制約:
  - DINOv2 推論は train/val 動画の切り抜きに対してのみ実行
  - テスト動画の crops は触らない（Phase 5e のまま）
  - フレーム分類器は train/val フレームの crops 特徴量で学習
    （Phase 5e で test 用に学習されたものは直接再利用できないため、
      同等ロジックで train データから再学習する）
"""

import numpy as np
import json
import os
import time
from PIL import Image
import torch
import torchvision.transforms as T
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


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
    """DINOv2 用の入力変換"""
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


def resolve_video_splits(videos_dict, splits, dataset_kind):
    """
    frame_results.videos のキー（動画名）を splits.json のエントリと
    マッチングして train/val/test を判定する。

    dataset_kind="wetlandbirds":
      splits.json: {"train_set":[...], "val_set":[...], "test_set":[...]}
      各要素: "124-black_headed_gull" （.mp4 なし）
      videos のキー: "124-black_headed_gull.mp4"

    dataset_kind="vb100":
      splits.json: {"train":[...], "test":[...]}
      各要素: "Acorn_Woodpecker/Acorn_Woodpecker_00001.mp4" （直接マッチ）
    """
    video_split = {}
    all_video_names = list(videos_dict.keys())

    if dataset_kind == "wetlandbirds":
        # 動画名の基底（.mp4 抜き）でマッチング
        basename_to_full = {v.replace(".mp4", ""): v for v in all_video_names}
        for split_name, vid_list in splits.items():
            for v_id in vid_list:
                v_str = str(v_id)
                if v_str in basename_to_full:
                    video_split[basename_to_full[v_str]] = split_name
    elif dataset_kind == "vb100":
        # 直接マッチ
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


def extract_features_from_crops(crops, model, transform, device, batch_size=32):
    """
    crops 配列から DINOv2 特徴量と crop メタ情報を抽出する。
    バッチ推論で高速化。
    """
    features = []
    metas = []
    batch_tensors = []
    batch_metas = []

    def flush_batch():
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
            flush_batch()

    flush_batch()

    return features, metas


def process_dataset(dataset_kind, dataset_name, frame_results_path,
                    splits_path, species_mapping_path, output_path,
                    model, transform, device):
    """1つのデータセットの train/val フレーム単位予測を生成"""
    print(f"\n{'=' * 70}")
    print(f"データセット: {dataset_name}")
    print(f"{'=' * 70}")

    with open(frame_results_path, encoding="utf-8") as f:
        frame_data = json.load(f)
    videos_dict = frame_data["videos"]

    with open(splits_path, encoding="utf-8") as f:
        splits = json.load(f)

    with open(species_mapping_path, encoding="utf-8") as f:
        mapping = json.load(f)

    species_to_id = mapping["species_to_id"]
    id_to_species = mapping["id_to_species"]
    n_classes = mapping["n_classes"]

    # 動画と分割のマッピング
    video_split = resolve_video_splits(videos_dict, splits, dataset_kind)

    # train + val を「学習側」として扱う
    train_videos = [
        v for v, s in video_split.items()
        if "train" in s.lower() or "val" in s.lower()
    ]
    test_videos = [
        v for v, s in video_split.items()
        if "test" in s.lower()
    ]

    print(f"train/val 動画: {len(train_videos)}")
    print(f"test 動画: {len(test_videos)}")

    # ========= ステップ1: train/val フレームの特徴抽出 =========
    print(f"\n--- ステップ1: train/val フレームの DINOv2 特徴抽出 ---")
    train_frame_features = []
    train_frame_labels = []

    start = time.time()
    for i, video_name in enumerate(train_videos):
        vinfo = videos_dict[video_name]
        species = vinfo["species"]
        label = species_to_id.get(species, -1)
        if label == -1:
            continue

        crops = vinfo.get("crops", [])
        feats, _ = extract_features_from_crops(crops, model, transform, device)
        for f in feats:
            train_frame_features.append(f)
            train_frame_labels.append(label)

        if (i + 1) % 50 == 0 or (i + 1) == len(train_videos):
            elapsed = time.time() - start
            print(f"  {i + 1}/{len(train_videos)} 動画処理済み "
                  f"(累計 {elapsed:.1f}秒, フレーム={len(train_frame_features)})")

    elapsed_feat = time.time() - start
    X_train_frames = np.array(train_frame_features, dtype=np.float32)
    y_train_frames = np.array(train_frame_labels, dtype=np.int64)
    print(f"  train/val フレーム総数: {X_train_frames.shape[0]} "
          f"(次元 {X_train_frames.shape[1] if X_train_frames.ndim > 1 else 'N/A'})")
    print(f"  特徴抽出時間: {elapsed_feat:.1f}秒")

    # ========= ステップ2: フレーム分類器の学習 =========
    print(f"\n--- ステップ2: フレーム分類器 (LogReg) 学習 ---")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_frames)

    clf = LogisticRegression(
        max_iter=2000, C=1.0, random_state=42, solver="lbfgs"
    )
    clf.fit(X_train_s, y_train_frames)
    train_acc = clf.score(X_train_s, y_train_frames)
    print(f"  フレーム単位の学習精度: {train_acc * 100:.2f}% "
          f"(n_samples={len(y_train_frames)}, n_classes={n_classes})")

    # ========= ステップ3: train/val 動画のフレーム予測生成 =========
    print(f"\n--- ステップ3: train/val 動画のフレーム単位予測 ---")
    train_predictions = {}

    start = time.time()
    for i, video_name in enumerate(train_videos):
        vinfo = videos_dict[video_name]
        species = vinfo["species"]
        true_label = species_to_id.get(species, -1)
        if true_label == -1:
            continue

        crops = vinfo.get("crops", [])
        feats, metas = extract_features_from_crops(crops, model, transform, device)
        if not feats:
            continue

        feats_arr = np.array(feats, dtype=np.float32)
        feats_s = scaler.transform(feats_arr)
        preds = clf.predict(feats_s)
        probs = clf.predict_proba(feats_s)

        frame_preds = []
        for j, meta in enumerate(metas):
            frame_preds.append({
                "timestamp": meta.get("timestamp", 0),
                "frame_idx": meta.get("frame_idx", 0),
                "predicted_label": int(preds[j]),
                "predicted_species": id_to_species[str(int(preds[j]))],
                "confidence": float(probs[j].max()),
                "prob_distribution": probs[j].tolist(),
                "is_fallback": meta.get("is_fallback", False),
                "yolo_confidence": meta.get("confidence", 0),
            })

        train_predictions[video_name] = {
            "species": species,
            "true_label": int(true_label),
            "n_frames": len(frame_preds),
            "frame_predictions": frame_preds,
        }

        if (i + 1) % 50 == 0 or (i + 1) == len(train_videos):
            elapsed = time.time() - start
            print(f"  {i + 1}/{len(train_videos)} 動画予測済み "
                  f"(累計 {elapsed:.1f}秒)")

    elapsed_pred = time.time() - start
    print(f"  train/val 予測完了: {len(train_predictions)} 動画 "
          f"({elapsed_pred:.1f}秒)")

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(train_predictions, f, indent=2, ensure_ascii=False)
    print(f"  保存: {output_path}")

    return {
        "n_train_videos": len(train_videos),
        "n_test_videos": len(test_videos),
        "n_train_predictions": len(train_predictions),
        "train_frame_count": X_train_frames.shape[0],
        "frame_classifier_train_acc": float(train_acc),
        "elapsed_feat_sec": elapsed_feat,
        "elapsed_pred_sec": elapsed_pred,
    }


def main():
    print("=== Phase 6 正式評価 段階A: train/val フレーム予測生成 ===")
    total_start = time.time()

    # DINOv2 読み込み
    model = load_dinov2()
    transform = get_transform()
    device = next(model.parameters()).device

    out_dir = "../results/phase6_formal"
    os.makedirs(out_dir, exist_ok=True)

    stats = {}

    # ===== WetlandBirds =====
    wb_frame = "../results/bird_phase4b/frame_results.json"
    wb_splits = "../data/wetlandbirds/splits.json"
    wb_mapping = "../results/bird_phase4b/species_mapping.json"

    if all(os.path.exists(p) for p in [wb_frame, wb_splits, wb_mapping]):
        stats["wetlandbirds"] = process_dataset(
            dataset_kind="wetlandbirds",
            dataset_name="WetlandBirds (13種)",
            frame_results_path=wb_frame,
            splits_path=wb_splits,
            species_mapping_path=wb_mapping,
            output_path=os.path.join(
                out_dir, "train_predictions_wetlandbirds.json"
            ),
            model=model, transform=transform, device=device,
        )
    else:
        print("  警告: WetlandBirds の入力ファイルが揃いません")

    # ===== VB100 =====
    vb_frame = "../results/vb100_phase5e/frame_results.json"
    vb_splits = "../results/vb100_phase5e/splits.json"
    vb_mapping = "../results/vb100_phase5e/species_mapping.json"

    if all(os.path.exists(p) for p in [vb_frame, vb_splits, vb_mapping]):
        stats["vb100"] = process_dataset(
            dataset_kind="vb100",
            dataset_name="VB100 (100種)",
            frame_results_path=vb_frame,
            splits_path=vb_splits,
            species_mapping_path=vb_mapping,
            output_path=os.path.join(
                out_dir, "train_predictions_vb100.json"
            ),
            model=model, transform=transform, device=device,
        )
    else:
        print("  警告: VB100 の入力ファイルが揃いません")

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"段階A 完了（合計 {total_elapsed:.1f}秒 = {total_elapsed / 60:.1f}分）")
    print(f"{'=' * 70}")
    for ds, s in stats.items():
        print(f"\n[{ds}]")
        for k, v in s.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.3f}")
            else:
                print(f"  {k}: {v}")

    # 統計を保存
    with open(os.path.join(out_dir, "phase6_train_predict_stats.json"),
              "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
