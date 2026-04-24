import numpy as np
import json
import os
import sys
import io
import time
from collections import Counter

from PIL import Image
import torch
import torchvision.transforms as T
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Windows cp932 対策
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


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
        return np.zeros((0, 384), dtype=np.float32), []

    feats = []
    for i in range(0, len(tensors), batch_size):
        batch = torch.stack(tensors[i:i + batch_size]).to(device)
        with torch.no_grad():
            f = model(batch).cpu().numpy()
        feats.append(f)
    feats = np.concatenate(feats, axis=0)
    return feats, valid_crops


def main():
    print("=== Phase 5e-A: フレーム単位の予測 ===")

    with open("../results/bird_phase4b/frame_results.json", encoding="utf-8") as f:
        frame_results_raw = json.load(f)
    with open("../results/bird_phase4b/species_mapping.json", encoding="utf-8") as f:
        mapping = json.load(f)
    with open("../data/wetlandbirds/splits.json", encoding="utf-8") as f:
        splits = json.load(f)

    frame_videos = frame_results_raw["videos"]
    species_to_id = mapping["species_to_id"]
    id_to_species = mapping["id_to_species"]
    n_classes = mapping["n_classes"]

    # splits は ".mp4" なし文字列のリスト → 動画名に揃える
    video_split = {}
    for split_name, video_list in splits.items():
        for v_id in video_list:
            v_str = str(v_id)
            candidate = v_str if v_str.endswith(".mp4") else f"{v_str}.mp4"
            if candidate in frame_videos:
                video_split[candidate] = split_name

    train_videos = [v for v, s in video_split.items()
                    if s in ("train_set", "val_set")]
    test_videos = [v for v, s in video_split.items() if s == "test_set"]
    print(f"学習動画(train+val): {len(train_videos)} 本")
    print(f"テスト動画: {len(test_videos)} 本")

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

    start = time.time()
    for i, video_name in enumerate(train_videos):
        vinfo = frame_videos[video_name]
        species = vinfo["species"]
        label = species_to_id.get(species, -1)
        if label == -1:
            continue

        crops = sorted(vinfo.get("crops", []),
                       key=lambda c: c.get("frame_idx", 0))
        feats, _ = extract_frame_features(model, transform, crops, device)
        for j in range(feats.shape[0]):
            train_frame_features.append(feats[j])
            train_frame_labels.append(label)

        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{len(train_videos)} 動画処理完了 "
                  f"(累計フレーム: {len(train_frame_features)}, "
                  f"経過: {time.time()-start:.1f}秒)")

    elapsed = time.time() - start
    print(f"  学習フレーム数: {len(train_frame_features)} ({elapsed:.1f}秒)")

    X_train = np.array(train_frame_features, dtype=np.float32)
    y_train = np.array(train_frame_labels, dtype=np.int64)

    # フレーム分類器
    print("\nフレーム分類器 (LogReg) を学習中...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42,
                             class_weight="balanced")
    clf.fit(X_train_s, y_train)
    train_acc = clf.score(X_train_s, y_train)
    print(f"  学習セット精度 (フレーム単位): {train_acc*100:.2f}%")

    # ========================================
    # テストセット: フレーム単位予測
    # ========================================
    print("\n--- テストセットのフレーム単位予測 ---")
    video_frame_predictions = {}
    total_frames = 0

    start = time.time()
    for video_name in test_videos:
        vinfo = frame_videos[video_name]
        species = vinfo["species"]
        true_label = species_to_id.get(species, -1)

        crops = sorted(vinfo.get("crops", []),
                       key=lambda c: c.get("frame_idx", 0))
        feats, valid_crops = extract_frame_features(model, transform, crops, device)

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

        total_frames += len(frame_preds)
        video_frame_predictions[video_name] = {
            "species": species,
            "true_label": int(true_label),
            "n_frames": len(frame_preds),
            "frame_predictions": frame_preds,
        }

    elapsed = time.time() - start
    print(f"  テスト動画: {len(test_videos)} 本")
    print(f"  テストフレーム数: {total_frames}")
    print(f"  処理時間: {elapsed:.1f}秒")

    # フレーム単位の統計
    print("\n--- フレーム単位予測の統計（動画内一致率）---")
    for video_name in sorted(video_frame_predictions.keys()):
        vpred = video_frame_predictions[video_name]
        preds = [fp["predicted_label"] for fp in vpred["frame_predictions"]]
        if not preds:
            print(f"  {video_name:<40} (フレームなし)")
            continue

        pred_counts = Counter(preds)
        most_common = pred_counts.most_common(1)[0]
        agreement = most_common[1] / len(preds)

        true_sp = vpred["species"]
        pred_sp = id_to_species[str(most_common[0])]
        correct = "○" if pred_sp == true_sp else "×"

        mark = ""
        if "142-mallard" in video_name:
            mark = " ← Phase 4b での誤分類動画"

        print(f"  {video_name:<40} 正解:{true_sp:<20} "
              f"最頻:{pred_sp:<20} 一致率:{agreement:.0%} {correct}{mark}")

    os.makedirs("../results/bird_phase5e", exist_ok=True)
    with open("../results/bird_phase5e/frame_predictions.json", "w", encoding="utf-8") as f:
        json.dump(video_frame_predictions, f, indent=2, ensure_ascii=False)
    print(f"\n保存: results/bird_phase5e/frame_predictions.json")


if __name__ == "__main__":
    main()
