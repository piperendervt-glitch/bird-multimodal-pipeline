"""Phase 1 段階B: DINOv2 による特徴抽出。

全画像の特徴ベクトルを抽出し .npz に保存する。
このスクリプトは sklearn をインポートしない（特徴抽出のみ）。
"""

import os
import json
import time
import hashlib

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


RESULTS_DIR = os.path.join("..", "results", "bird_phase1")
CUB_DIR = os.path.join("..", "data", "cub200", "CUB_200_2011")


def load_dinov2(model_name="dinov2_vits14"):
    """DINOv2 ViT-S/14 を torch.hub 経由で読み込む"""
    try:
        model = torch.hub.load("facebookresearch/dinov2", model_name)
    except Exception as e:
        print(f"DINOv2 の読み込みに失敗: {e}")
        print("\nDINOv2 のダウンロードに失敗しました。")
        print("ネットワーク設定で github.com と dl.fbaipublicfiles.com へのアクセスを許可してください。")
        raise
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def get_transform():
    """DINOv2 の標準前処理（ImageNet 統計量）"""
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])


def extract_features(model, image_paths, transform, batch_size=32):
    """全画像の特徴ベクトルを抽出"""
    device = next(model.parameters()).device
    features = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_imgs = []

        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                img = transform(img)
                batch_imgs.append(img)
            except Exception as e:
                print(f"  警告: {path} の読み込みに失敗: {e}")
                # 失敗した画像はゼロベクトルで埋める（インデックス整合性のため）
                batch_imgs.append(torch.zeros(3, 224, 224))

        batch_tensor = torch.stack(batch_imgs).to(device)

        with torch.no_grad():
            feat = model(batch_tensor)  # ViT-S は (batch, 384)

        features.append(feat.cpu().numpy())

        if (i // batch_size + 1) % 10 == 0:
            print(f"  {i + len(batch_paths)}/{len(image_paths)} 枚処理完了")

    return np.concatenate(features, axis=0)


def main():
    print("=== 段階B: DINOv2 特徴抽出 ===")

    # メタデータ読み込み
    meta_path = os.path.join(RESULTS_DIR, "metadata.json")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    # 画像パスとラベルの読み込み（段階A と同じロジック）
    images = {}
    with open(os.path.join(CUB_DIR, "images.txt"), encoding="utf-8") as f:
        for line in f:
            img_id, path = line.strip().split()
            images[int(img_id)] = os.path.join(CUB_DIR, "images", path)

    labels = {}
    with open(os.path.join(CUB_DIR, "image_class_labels.txt"), encoding="utf-8") as f:
        for line in f:
            img_id, label = line.strip().split()
            labels[int(img_id)] = int(label) - 1

    train_ids = meta["train_ids"]
    test_ids = meta["test_ids"]

    train_paths = [images[i] for i in train_ids]
    train_labels = np.array([labels[i] for i in train_ids], dtype=np.int64)
    test_paths = [images[i] for i in test_ids]
    test_labels = np.array([labels[i] for i in test_ids], dtype=np.int64)

    # DINOv2 読み込み
    print("\nDINOv2 (ViT-S/14) を読み込み中...")
    model = load_dinov2("dinov2_vits14")

    device = next(model.parameters()).device
    print(f"デバイス: {device}")

    transform = get_transform()

    # 特徴量の次元を確認（ダミー入力で推論）
    dummy = torch.zeros(1, 3, 224, 224).to(device)
    with torch.no_grad():
        feat_dim = model(dummy).shape[1]
    print(f"特徴量次元: {feat_dim}")

    # 学習セットの特徴抽出
    print(f"\n学習セット ({len(train_paths)} 枚) の特徴抽出中...")
    start = time.time()
    X_train = extract_features(model, train_paths, transform)
    elapsed_tr = time.time() - start
    print(f"完了: {elapsed_tr:.1f} 秒 ({len(train_paths)/elapsed_tr:.1f} 枚/秒)")

    # テストセットの特徴抽出
    print(f"\nテストセット ({len(test_paths)} 枚) の特徴抽出中...")
    start = time.time()
    X_test = extract_features(model, test_paths, transform)
    elapsed_te = time.time() - start
    print(f"完了: {elapsed_te:.1f} 秒 ({len(test_paths)/elapsed_te:.1f} 枚/秒)")

    # 保存
    output_path = os.path.join(RESULTS_DIR, "features_dinov2_vits14.npz")
    np.savez(
        output_path,
        X_train=X_train.astype(np.float32),
        y_train=train_labels,
        X_test=X_test.astype(np.float32),
        y_test=test_labels,
        feat_dim=np.int64(feat_dim),
        model_name=np.array("dinov2_vits14"),
    )
    print(f"\n保存: {output_path}")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test:  {X_test.shape}")

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  ファイルサイズ: {size_mb:.2f} MB")

    # SHA256
    with open(output_path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    print(f"  SHA256: {sha}")


if __name__ == "__main__":
    main()
