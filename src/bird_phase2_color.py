"""Phase 2 段階A: 色彩特徴の抽出。

CUB-200-2011 の各画像から OpenCV を用いて色彩特徴を抽出する。
DINOv2 や MLP はインポートしない（OpenCV のみ使用）。
"""

import os
import json
import time

import cv2
import numpy as np


PHASE1_DIR = os.path.join("..", "results", "bird_phase1")
PHASE2_DIR = os.path.join("..", "results", "bird_phase2")
CUB_DIR = os.path.join("..", "data", "cub200", "CUB_200_2011")

# 特徴量内訳: H(18) + S(8) + V(8) + 支配色BGR(9) + 面積比率(3) + 統計量(4) = 50
COLOR_FEAT_DIM = 50


def extract_color_features(image_path):
    """1枚の画像から色彩特徴（50次元）を抽出"""
    img = cv2.imread(image_path)
    if img is None:
        return None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    features = []

    # 特徴1: HSV ヒストグラム (H:18bin, S:8bin, V:8bin, 合計34次元)
    # OpenCV の H は 0-179
    h_hist = cv2.calcHist([hsv], [0], None, [18], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256])

    h_hist = h_hist.flatten() / (h_hist.sum() + 1e-9)
    s_hist = s_hist.flatten() / (s_hist.sum() + 1e-9)
    v_hist = v_hist.flatten() / (v_hist.sum() + 1e-9)

    features.extend(h_hist)
    features.extend(s_hist)
    features.extend(v_hist)

    # 特徴2: 支配色3色（K-means、面積比率の降順）
    pixels = img.reshape(-1, 3).astype(np.float32)
    if len(pixels) > 5000:
        idx = np.random.RandomState(42).choice(len(pixels), 5000, replace=False)
        pixels = pixels[idx]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, 3, None, criteria, 3, cv2.KMEANS_PP_CENTERS)

    counts = np.bincount(labels.flatten(), minlength=3)
    order = counts.argsort()[::-1]
    for i in order:
        features.extend(centers[i] / 255.0)  # BGR を 0-1 に正規化
    ratios = counts[order] / (counts.sum() + 1e-9)
    features.extend(ratios)

    # 特徴3: 彩度・明度の統計量（平均と標準偏差）
    s_channel = hsv[:, :, 1].astype(np.float32)
    v_channel = hsv[:, :, 2].astype(np.float32)
    features.append(s_channel.mean() / 255.0)
    features.append(s_channel.std() / 255.0)
    features.append(v_channel.mean() / 255.0)
    features.append(v_channel.std() / 255.0)

    return np.array(features, dtype=np.float32)


def extract_batch(image_paths, label):
    """画像パスのリストから特徴量を抽出"""
    print(f"\n{label} ({len(image_paths)} 枚) の色彩特徴抽出中...")
    start = time.time()
    features = []
    for i, path in enumerate(image_paths):
        feat = extract_color_features(path)
        if feat is None:
            feat = np.zeros(COLOR_FEAT_DIM, dtype=np.float32)
        features.append(feat)
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(image_paths)} 完了")
    features = np.array(features)
    elapsed = time.time() - start
    print(f"完了: {elapsed:.1f} 秒 ({len(image_paths)/elapsed:.1f} 枚/秒)")
    return features


def main():
    print("=== Phase 2A: 色彩特徴抽出 ===")

    # メタデータ読み込み
    with open(os.path.join(PHASE1_DIR, "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)

    # 画像パス読み込み
    images = {}
    with open(os.path.join(CUB_DIR, "images.txt"), encoding="utf-8") as f:
        for line in f:
            img_id, path = line.strip().split()
            images[int(img_id)] = os.path.join(CUB_DIR, "images", path)

    train_ids = meta["train_ids"]
    test_ids = meta["test_ids"]

    train_paths = [images[i] for i in train_ids]
    test_paths = [images[i] for i in test_ids]

    train_features = extract_batch(train_paths, "学習セット")
    test_features = extract_batch(test_paths, "テストセット")

    # 保存
    os.makedirs(PHASE2_DIR, exist_ok=True)
    output_path = os.path.join(PHASE2_DIR, "features_color.npz")
    np.savez(output_path,
             X_train=train_features.astype(np.float32),
             X_test=test_features.astype(np.float32))
    print(f"\n保存: {output_path}")
    print(f"  X_train: {train_features.shape}")
    print(f"  X_test:  {test_features.shape}")
    print(f"  特徴量次元: {train_features.shape[1]}")


if __name__ == "__main__":
    main()
