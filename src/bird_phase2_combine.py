"""Phase 2 段階C: 特徴量の結合。

Phase 1 の DINOv2 特徴量と Phase 2 の色彩・形態特徴量を結合する。
DINOv2 モデルや OpenCV はインポートしない。
"""

import os
import json
import hashlib

import numpy as np
from sklearn.preprocessing import StandardScaler


PHASE1_DIR = os.path.join("..", "results", "bird_phase1")
PHASE2_DIR = os.path.join("..", "results", "bird_phase2")


def _sanitize(arr, name, label):
    """NaN/Inf の検出と 0 への置換"""
    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())
    if n_nan > 0 or n_inf > 0:
        print(f"  警告: {name} ({label}) に NaN={n_nan}, Inf={n_inf} を検出。0 に置換。")
        arr[np.isnan(arr)] = 0
        arr[np.isinf(arr)] = 0
    return n_nan, n_inf


def main():
    print("=== Phase 2C: 特徴量結合 ===")

    # Phase 1: DINOv2
    dino = np.load(os.path.join(PHASE1_DIR, "features_dinov2_vits14.npz"))
    X_train_dino = dino["X_train"]
    X_test_dino = dino["X_test"]
    y_train = dino["y_train"]
    y_test = dino["y_test"]

    # Phase 2: 色彩
    color = np.load(os.path.join(PHASE2_DIR, "features_color.npz"))
    X_train_color = color["X_train"]
    X_test_color = color["X_test"]

    # Phase 2: 形態
    shape = np.load(os.path.join(PHASE2_DIR, "features_shape.npz"))
    X_train_shape = shape["X_train"]
    X_test_shape = shape["X_test"]

    print(f"DINOv2: {X_train_dino.shape[1]} 次元")
    print(f"色彩:   {X_train_color.shape[1]} 次元")
    print(f"形態:   {X_train_shape.shape[1]} 次元")

    # サンプル数整合性
    assert X_train_dino.shape[0] == X_train_color.shape[0] == X_train_shape.shape[0], \
        "学習セットのサンプル数が不一致"
    assert X_test_dino.shape[0] == X_test_color.shape[0] == X_test_shape.shape[0], \
        "テストセットのサンプル数が不一致"

    # 標準化（学習セットの統計量を使用）
    scaler_dino = StandardScaler()
    X_train_dino_n = scaler_dino.fit_transform(X_train_dino)
    X_test_dino_n = scaler_dino.transform(X_test_dino)

    scaler_color = StandardScaler()
    X_train_color_n = scaler_color.fit_transform(X_train_color)
    X_test_color_n = scaler_color.transform(X_test_color)

    scaler_shape = StandardScaler()
    X_train_shape_n = scaler_shape.fit_transform(X_train_shape)
    X_test_shape_n = scaler_shape.transform(X_test_shape)

    # NaN/Inf のチェック
    total_nan_inf = 0
    for name, arr, label in [
        ("dino", X_train_dino_n, "train"), ("dino", X_test_dino_n, "test"),
        ("color", X_train_color_n, "train"), ("color", X_test_color_n, "test"),
        ("shape", X_train_shape_n, "train"), ("shape", X_test_shape_n, "test"),
    ]:
        n_nan, n_inf = _sanitize(arr, name, label)
        total_nan_inf += n_nan + n_inf
    if total_nan_inf == 0:
        print("  NaN/Inf なし（全特徴量）")

    # 結合パターン
    combinations = {
        "dino_only": (X_train_dino_n, X_test_dino_n),
        "dino_color": (
            np.concatenate([X_train_dino_n, X_train_color_n], axis=1),
            np.concatenate([X_test_dino_n, X_test_color_n], axis=1),
        ),
        "dino_shape": (
            np.concatenate([X_train_dino_n, X_train_shape_n], axis=1),
            np.concatenate([X_test_dino_n, X_test_shape_n], axis=1),
        ),
        "dino_color_shape": (
            np.concatenate([X_train_dino_n, X_train_color_n, X_train_shape_n], axis=1),
            np.concatenate([X_test_dino_n, X_test_color_n, X_test_shape_n], axis=1),
        ),
        "color_shape": (
            np.concatenate([X_train_color_n, X_train_shape_n], axis=1),
            np.concatenate([X_test_color_n, X_test_shape_n], axis=1),
        ),
    }

    print(f"\n結合パターン:")
    for name, (tr, te) in combinations.items():
        print(f"  {name}: {tr.shape[1]} 次元 (train={tr.shape[0]}, test={te.shape[0]})")

    # 保存
    output_path = os.path.join(PHASE2_DIR, "features_combined.npz")
    save_dict = {"y_train": y_train, "y_test": y_test}
    for name, (tr, te) in combinations.items():
        save_dict[f"X_train_{name}"] = tr.astype(np.float32)
        save_dict[f"X_test_{name}"] = te.astype(np.float32)
    np.savez(output_path, **save_dict)

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"\n保存: {output_path} ({size_mb:.2f} MB)")

    with open(output_path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    print(f"SHA256: {sha}")


if __name__ == "__main__":
    main()
