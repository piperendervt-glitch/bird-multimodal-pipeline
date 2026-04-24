"""Phase 1 段階A: CUB-200-2011 データセットの取得と公式分割の読み込み。

このスクリプトは torch をインポートしない（データ準備のみ）。
"""

import os
import sys
import json
import hashlib
import tarfile
import urllib.request
from collections import Counter


DATA_DIR = os.path.join("..", "data", "cub200")
CUB_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
CUB_TGZ = os.path.join(DATA_DIR, "CUB_200_2011.tgz")
RESULTS_DIR = os.path.join("..", "results", "bird_phase1")


def download_cub():
    """CUB-200-2011 のアーカイブをダウンロード"""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(CUB_TGZ):
        size_mb = os.path.getsize(CUB_TGZ) / 1024 / 1024
        print(f"既にダウンロード済み: {CUB_TGZ} ({size_mb:.1f} MB)")
        return True

    print(f"CUB-200-2011 をダウンロード中... (約1.1GB)")
    print(f"URL: {CUB_URL}")
    try:
        urllib.request.urlretrieve(CUB_URL, CUB_TGZ)
        print("ダウンロード完了")
        return True
    except Exception as e:
        print(f"ダウンロード失敗: {e}")
        print("\nCUB-200-2011 のダウンロードに失敗しました。")
        print("手動でダウンロードしてください:")
        print(f"  URL: {CUB_URL}")
        print(f"  保存先: {CUB_TGZ}")
        return False


def extract_cub():
    """アーカイブを展開"""
    cub_dir = os.path.join(DATA_DIR, "CUB_200_2011")
    if os.path.exists(cub_dir):
        print(f"既に展開済み: {cub_dir}")
        return cub_dir

    print("展開中...")
    with tarfile.open(CUB_TGZ, "r:gz") as tar:
        tar.extractall(DATA_DIR)
    print(f"展開完了: {cub_dir}")
    return cub_dir


def load_split(cub_dir):
    """CUB-200-2011 の公式学習/テスト分割を読み込む"""
    # 画像IDからファイルパス
    images = {}
    with open(os.path.join(cub_dir, "images.txt"), encoding="utf-8") as f:
        for line in f:
            img_id, path = line.strip().split()
            images[int(img_id)] = path

    # 画像IDからクラスラベル（1-indexed → 0-indexed に変換）
    labels = {}
    with open(os.path.join(cub_dir, "image_class_labels.txt"), encoding="utf-8") as f:
        for line in f:
            img_id, label = line.strip().split()
            labels[int(img_id)] = int(label) - 1

    # 学習(1) / テスト(0) の分割
    split = {}
    with open(os.path.join(cub_dir, "train_test_split.txt"), encoding="utf-8") as f:
        for line in f:
            img_id, is_train = line.strip().split()
            split[int(img_id)] = int(is_train)

    # クラスID → クラス名
    classes = {}
    with open(os.path.join(cub_dir, "classes.txt"), encoding="utf-8") as f:
        for line in f:
            cls_id, cls_name = line.strip().split()
            classes[int(cls_id) - 1] = cls_name

    train_data, test_data = [], []
    for img_id in sorted(images.keys()):
        entry = {
            "img_id": img_id,
            "path": os.path.join(cub_dir, "images", images[img_id]),
            "label": labels[img_id],
            "class_name": classes[labels[img_id]],
        }
        if split[img_id] == 1:
            train_data.append(entry)
        else:
            test_data.append(entry)

    return train_data, test_data, classes


def main():
    print("=== 段階A: CUB-200-2011 データセットの準備 ===")

    # ダウンロード
    if not download_cub():
        sys.exit(1)

    # 展開
    cub_dir = extract_cub()

    # 公式分割の読み込み
    train_data, test_data, classes = load_split(cub_dir)

    # 統計表示
    print(f"\n=== CUB-200-2011 データセット統計 ===")
    print(f"クラス数: {len(classes)}")
    print(f"学習セット: {len(train_data)} 枚")
    print(f"テストセット: {len(test_data)} 枚")

    train_counts = Counter(d["label"] for d in train_data)
    test_counts = Counter(d["label"] for d in test_data)
    print(f"学習: 1クラスあたり {min(train_counts.values())}〜{max(train_counts.values())} 枚")
    print(f"テスト: 1クラスあたり {min(test_counts.values())}〜{max(test_counts.values())} 枚")

    # 画像の存在確認（先頭5枚）
    print(f"\n画像ファイルの存在確認:")
    missing = 0
    for d in train_data[:5]:
        exists = os.path.exists(d["path"])
        print(f"  {os.path.basename(d['path'])}: {'OK' if exists else 'MISSING'}")
        if not exists:
            missing += 1

    # 全体での欠損チェック（パスのみ、I/Oは最小限）
    all_missing = sum(1 for d in train_data + test_data if not os.path.exists(d["path"]))
    if all_missing > 0:
        print(f"  警告: 合計 {all_missing} 枚の画像ファイルが見つかりません")
    else:
        print(f"  全 {len(train_data) + len(test_data)} 枚の存在を確認")

    # メタデータ保存
    os.makedirs(RESULTS_DIR, exist_ok=True)
    metadata = {
        "dataset": "CUB-200-2011",
        "source_url": CUB_URL,
        "n_classes": len(classes),
        "n_train": len(train_data),
        "n_test": len(test_data),
        "classes": {str(k): v for k, v in classes.items()},
        "train_ids": [d["img_id"] for d in train_data],
        "test_ids": [d["img_id"] for d in test_data],
    }

    meta_path = os.path.join(RESULTS_DIR, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"\nメタデータを保存: {meta_path}")

    # 分割データの SHA256 を計算（改竄検知用）
    split_str = json.dumps(metadata["train_ids"] + metadata["test_ids"])
    sha = hashlib.sha256(split_str.encode()).hexdigest()
    print(f"分割データの SHA256: {sha}")


if __name__ == "__main__":
    main()
