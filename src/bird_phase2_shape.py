"""Phase 2 段階B: 形態特徴の抽出。

CUB-200-2011 の各画像から OpenCV を用いて形態特徴を抽出する。
DINOv2 や MLP はインポートしない（OpenCV のみ使用）。

特徴量内訳（合計18次元）:
  - 面積(1) + 周長(1) + 円形度(1) + アスペクト比(1) + 充填率(1) + 凸性(1) = 6
  - Hu モーメント = 7
  - エッジ密度 = 1
  - エッジ方向ヒストグラム(4方向) = 4
"""

import os
import json
import time

import cv2
import numpy as np


PHASE1_DIR = os.path.join("..", "results", "bird_phase1")
PHASE2_DIR = os.path.join("..", "results", "bird_phase2")
CUB_DIR = os.path.join("..", "data", "cub200", "CUB_200_2011")

SHAPE_FEAT_DIM = 18


def extract_shape_features(image_path):
    """1枚の画像から形態特徴（18次元）を抽出"""
    img = cv2.imread(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_area = gray.shape[0] * gray.shape[1]

    # エッジ検出
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        # 輪郭が見つからない場合はゼロベクトルで返す
        return np.zeros(SHAPE_FEAT_DIM, dtype=np.float32)

    features = []

    # 最大輪郭（鳥の外形と仮定）
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, True)

    # 特徴1: 正規化面積
    features.append(area / (img_area + 1e-9))
    # 特徴2: 正規化周長
    features.append(perimeter / (2 * (gray.shape[0] + gray.shape[1]) + 1e-9))

    # 特徴3: 円形度
    circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-9)
    features.append(min(circularity, 1.0))

    # 特徴4: バウンディングボックスのアスペクト比
    x, y, w, h = cv2.boundingRect(largest)
    aspect_ratio = w / (h + 1e-9)
    features.append(aspect_ratio)

    # 特徴5: 充填率（extent）
    extent = area / (w * h + 1e-9)
    features.append(min(extent, 1.0))

    # 特徴6: 凸性（solidity）
    hull = cv2.convexHull(largest)
    hull_area = cv2.contourArea(hull)
    solidity = area / (hull_area + 1e-9)
    features.append(min(solidity, 1.0))

    # 特徴7: Hu モーメント (7次元、対数スケール化)
    moments = cv2.moments(largest)
    hu = cv2.HuMoments(moments).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    features.extend(hu_log)

    # 特徴8: エッジ密度
    edge_density = edges.sum() / (255.0 * img_area + 1e-9)
    features.append(edge_density)

    # 特徴9: エッジ方向ヒストグラム（Sobel、4方向、勾配強度で重み付け）
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    angles = np.arctan2(gy, gx)
    magnitudes = np.sqrt(gx ** 2 + gy ** 2)

    angle_bins = np.digitize(angles, [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]) - 1
    angle_bins = np.clip(angle_bins, 0, 3)
    dir_hist = np.zeros(4, dtype=np.float32)
    for b in range(4):
        dir_hist[b] = magnitudes[angle_bins == b].sum()
    dir_hist = dir_hist / (dir_hist.sum() + 1e-9)
    features.extend(dir_hist)

    return np.array(features, dtype=np.float32)


def extract_batch(image_paths, label, feat_dim):
    print(f"\n{label} ({len(image_paths)} 枚) の形態特徴抽出中...")
    start = time.time()
    features = []
    for i, path in enumerate(image_paths):
        feat = extract_shape_features(path)
        if feat is None:
            feat = np.zeros(feat_dim, dtype=np.float32)
        features.append(feat)
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(image_paths)} 完了")
    features = np.array(features)
    elapsed = time.time() - start
    print(f"完了: {elapsed:.1f} 秒 ({len(image_paths)/elapsed:.1f} 枚/秒)")
    return features


def main():
    print("=== Phase 2B: 形態特徴抽出 ===")

    with open(os.path.join(PHASE1_DIR, "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)

    images = {}
    with open(os.path.join(CUB_DIR, "images.txt"), encoding="utf-8") as f:
        for line in f:
            img_id, path = line.strip().split()
            images[int(img_id)] = os.path.join(CUB_DIR, "images", path)

    train_ids = meta["train_ids"]
    test_ids = meta["test_ids"]
    train_paths = [images[i] for i in train_ids]
    test_paths = [images[i] for i in test_ids]

    # 次元確認
    sample_feat = extract_shape_features(train_paths[0])
    feat_dim = len(sample_feat)
    print(f"形態特徴量次元: {feat_dim}")
    assert feat_dim == SHAPE_FEAT_DIM, f"期待次元 {SHAPE_FEAT_DIM} と不一致"

    train_features = extract_batch(train_paths, "学習セット", feat_dim)
    test_features = extract_batch(test_paths, "テストセット", feat_dim)

    os.makedirs(PHASE2_DIR, exist_ok=True)
    output_path = os.path.join(PHASE2_DIR, "features_shape.npz")
    np.savez(output_path,
             X_train=train_features.astype(np.float32),
             X_test=test_features.astype(np.float32))
    print(f"\n保存: {output_path}")
    print(f"  X_train: {train_features.shape}")
    print(f"  X_test:  {test_features.shape}")


if __name__ == "__main__":
    main()
