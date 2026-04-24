"""Phase 4a 段階B: YOLO 検出結果で画像を切り抜き、DINOv2 で特徴抽出する。

3 パターンの特徴を保存:
- X_original: 元画像そのまま（Phase 1 相当）
- X_yolo_crop: YOLO の bbox で切り抜き（未検出時は元画像フォールバック）
- X_gt_crop: 正解 bbox で切り抜き（YOLO の上限）
"""

import json
import os
import time

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


PHASE1_DIR = os.path.join("..", "results", "bird_phase1")
PHASE4A_DIR = os.path.join("..", "results", "bird_phase4a")
CUB_DIR = os.path.join("..", "data", "cub200", "CUB_200_2011")


def load_dinov2():
    print("DINOv2 読み込み中...")
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
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


def crop_image(img, bbox, padding_ratio=0.1):
    """bbox で切り抜き。鳥全体が入るよう padding を加える。"""
    w_img, h_img = img.size
    x1, y1, x2, y2 = bbox
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad_x = bw * padding_ratio
    pad_y = bh * padding_ratio
    x1 = max(0.0, x1 - pad_x)
    y1 = max(0.0, y1 - pad_y)
    x2 = min(float(w_img), x2 + pad_x)
    y2 = min(float(h_img), y2 + pad_y)
    x1_i, y1_i = int(x1), int(y1)
    x2_i, y2_i = max(x1_i + 1, int(x2)), max(y1_i + 1, int(y2))
    return img.crop((x1_i, y1_i, x2_i, y2_i))


def extract_features_batch(model, images, transform, batch_size=32):
    """画像リストを DINOv2 で特徴抽出（バッチ処理）"""
    device = next(model.parameters()).device
    feats = []
    total = len(images)
    for i in range(0, total, batch_size):
        batch = images[i:i + batch_size]
        tensors = []
        for img in batch:
            try:
                tensors.append(transform(img))
            except Exception:
                tensors.append(torch.zeros(3, 224, 224))
        batch_tensor = torch.stack(tensors).to(device)
        with torch.no_grad():
            feat = model(batch_tensor)
        feats.append(feat.cpu().numpy())
        if ((i // batch_size) + 1) % 20 == 0:
            print(f"    {i+len(batch)}/{total} バッチ処理済")
    return np.concatenate(feats, axis=0)


def main():
    print("=== Phase 4a-B: 切り抜き + DINOv2 特徴抽出 ===")

    with open(os.path.join(PHASE4A_DIR, "detection_results.json"),
              encoding="utf-8") as f:
        det_data = json.load(f)
    det_map = {r["img_id"]: r for r in det_data["results"]}
    print(f"検出結果: {len(det_map)} 件読み込み")

    with open(os.path.join(PHASE1_DIR, "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)

    images_map = {}
    with open(os.path.join(CUB_DIR, "images.txt")) as f:
        for line in f:
            img_id, path = line.strip().split()
            images_map[int(img_id)] = os.path.join(CUB_DIR, "images", path)

    labels = {}
    with open(os.path.join(CUB_DIR, "image_class_labels.txt")) as f:
        for line in f:
            img_id, label = line.strip().split()
            labels[int(img_id)] = int(label) - 1

    model = load_dinov2()
    transform = get_transform()

    os.makedirs(PHASE4A_DIR, exist_ok=True)

    for split_name, id_list in [("train", meta["train_ids"]),
                                 ("test", meta["test_ids"])]:
        print(f"\n--- {split_name} ({len(id_list)} 枚) ---")

        original_imgs = []
        yolo_crop_imgs = []
        gt_crop_imgs = []
        y_labels = []
        fallback_count = 0

        for img_id in id_list:
            img = Image.open(images_map[img_id]).convert("RGB")
            det = det_map.get(img_id, {})

            # パターン1: 元画像
            original_imgs.append(img)

            # パターン2: YOLO 切り抜き（未検出時は元画像フォールバック）
            if det.get("detected") and det.get("best_detection"):
                d = det["best_detection"]
                yolo_crop_imgs.append(
                    crop_image(img, (d["x1"], d["y1"], d["x2"], d["y2"]))
                )
            else:
                yolo_crop_imgs.append(img)
                fallback_count += 1

            # パターン3: 正解 bbox 切り抜き
            gt = det.get("gt_bbox")
            if gt:
                gt_crop_imgs.append(
                    crop_image(img, (gt["x"], gt["y"],
                                     gt["x"] + gt["w"], gt["y"] + gt["h"]))
                )
            else:
                gt_crop_imgs.append(img)

            y_labels.append(labels[img_id])

        print(f"  YOLO フォールバック (元画像使用): {fallback_count} 枚")
        y = np.array(y_labels)

        print(f"  元画像の特徴抽出中...")
        t0 = time.time()
        X_original = extract_features_batch(model, original_imgs, transform)
        print(f"  完了: {time.time()-t0:.1f} 秒, shape={X_original.shape}")

        print(f"  YOLO 切り抜きの特徴抽出中...")
        t0 = time.time()
        X_yolo = extract_features_batch(model, yolo_crop_imgs, transform)
        print(f"  完了: {time.time()-t0:.1f} 秒, shape={X_yolo.shape}")

        print(f"  正解 bbox 切り抜きの特徴抽出中...")
        t0 = time.time()
        X_gt = extract_features_batch(model, gt_crop_imgs, transform)
        print(f"  完了: {time.time()-t0:.1f} 秒, shape={X_gt.shape}")

        out_path = os.path.join(PHASE4A_DIR, f"features_{split_name}.npz")
        np.savez(out_path,
                 X_original=X_original,
                 X_yolo_crop=X_yolo,
                 X_gt_crop=X_gt,
                 y=y)
        print(f"  保存: {out_path}")


if __name__ == "__main__":
    main()
