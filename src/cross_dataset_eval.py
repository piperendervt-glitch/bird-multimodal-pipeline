"""クロスデータセット評価: CUB-200 ⇄ VB100（共通16種）。

- CUB-200: Phase 1 の DINOv2 特徴量 (features_dinov2_vits14.npz) を再利用
- VB100:   crop 画像から DINOv2 ViT-S/14 で新規抽出し、動画単位でフレーム平均
- 双方向（CUB→VB / VB→CUB）と各データセット内ベースラインを評価
"""

import csv
import json
import os
import sys
import time
from collections import Counter

import numpy as np


COMMON_SPECIES = [
    "Brown_Pelican", "California_Gull", "Elegant_Tern",
    "Hooded_Merganser", "Hooded_Oriole", "Northern_Waterthrush",
    "Orchard_Oriole", "Pied_billed_Grebe", "Red_breasted_Merganser",
    "Rose_breasted_Grosbeak", "Rufous_Hummingbird", "Song_Sparrow",
    "Summer_Tanager", "Western_Grebe", "Western_Meadowlark",
    "White_crowned_Sparrow",
]


def normalize_name(name):
    """種名を正規化して比較可能にする（_ や - を消し小文字化）。"""
    if not name:
        return ""
    return name.lower().replace("_", "").replace("-", "").strip()


COMMON_NORMALIZED = {normalize_name(s): s for s in COMMON_SPECIES}


def load_cub200_common():
    """CUB-200 の Phase 1 特徴量から共通種のサンプルを抽出する。"""
    print(f"\n--- CUB-200 共通種の抽出 ---")

    feats_path = "../results/bird_phase1/features_dinov2_vits14.npz"
    meta_path = "../results/bird_phase1/metadata.json"

    if not (os.path.exists(feats_path) and os.path.exists(meta_path)):
        print(f"  CUB-200 特徴量またはメタが見つかりません")
        return None

    data = np.load(feats_path, allow_pickle=True)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    classes = meta["classes"]   # {"0": "001.Black_footed_Albatross", ...}

    # ラベル ID → 正規化種名 → 正規化済み共通名
    common_id_to_name = {}
    for id_str, full_name in classes.items():
        # "100.Brown_Pelican" → "Brown_Pelican"
        sp = full_name.split(".", 1)[-1] if "." in full_name else full_name
        if normalize_name(sp) in COMMON_NORMALIZED:
            common_id_to_name[int(id_str)] = COMMON_NORMALIZED[normalize_name(sp)]

    print(f"  共通種: {len(common_id_to_name)} / {len(COMMON_SPECIES)}")

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    train_mask = np.isin(y_train, list(common_id_to_name.keys()))
    test_mask = np.isin(y_test, list(common_id_to_name.keys()))

    X_train_c = X_train[train_mask]
    X_test_c = X_test[test_mask]
    y_train_c = np.array([common_id_to_name[int(y)] for y in y_train[train_mask]])
    y_test_c = np.array([common_id_to_name[int(y)] for y in y_test[test_mask]])

    print(f"  学習: {len(X_train_c)} 枚, テスト: {len(X_test_c)} 枚")
    return {"X_train": X_train_c, "y_train": y_train_c,
            "X_test": X_test_c, "y_test": y_test_c}


def extract_vb100_common():
    """VB100 の共通種動画について crop 画像を DINOv2 で特徴抽出し動画単位で平均する。"""
    print(f"\n--- VB100 共通種の抽出 ---")

    fr_path = "../results/vb100_phase5e/frame_results.json"
    sp_path = "../results/vb100_phase5e/splits.json"

    with open(fr_path, encoding="utf-8") as f:
        fr_data = json.load(f)
    with open(sp_path, encoding="utf-8") as f:
        sp_data = json.load(f)

    videos = fr_data.get("videos", fr_data)
    train_set = set(sp_data["train"])
    test_set = set(sp_data["test"])

    # 共通種の動画を split 別に集約
    common_videos = {"train": [], "test": []}
    for vname, vinfo in videos.items():
        sp_name = vinfo.get("species", "")
        if normalize_name(sp_name) not in COMMON_NORMALIZED:
            continue
        canonical = COMMON_NORMALIZED[normalize_name(sp_name)]
        if vname in train_set:
            common_videos["train"].append((vname, canonical, vinfo))
        elif vname in test_set:
            common_videos["test"].append((vname, canonical, vinfo))

    print(f"  共通種 train 動画: {len(common_videos['train'])}")
    print(f"  共通種 test 動画:  {len(common_videos['test'])}")
    total_frames = sum(len(v[2].get("crops", []))
                       for split in common_videos.values() for v in split)
    print(f"  抽出予定フレーム: {total_frames}")

    # DINOv2 ViT-S/14 をロード
    import torch
    import torchvision.transforms as T
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval()
    model = model.to(device)

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    def extract_video_feature(crops, batch_size=32):
        """動画の crop 画像を DINOv2 でバッチ推論しフレーム平均を返す。"""
        feats = []
        batch_imgs = []
        for crop in crops:
            crop_path = crop.get("crop_path", "")
            if not os.path.exists(crop_path):
                continue
            try:
                img = Image.open(crop_path).convert("RGB")
                batch_imgs.append(transform(img))
            except Exception:
                continue
            if len(batch_imgs) >= batch_size:
                tensor = torch.stack(batch_imgs).to(device)
                with torch.no_grad():
                    out = model(tensor).cpu().numpy()
                feats.extend(out)
                batch_imgs = []
        if batch_imgs:
            tensor = torch.stack(batch_imgs).to(device)
            with torch.no_grad():
                out = model(tensor).cpu().numpy()
            feats.extend(out)
        if not feats:
            return None
        return np.mean(np.array(feats), axis=0).astype(np.float32)

    results = {"train": {"X": [], "y": []},
               "test": {"X": [], "y": []}}

    start = time.time()
    for split in ["train", "test"]:
        n_videos = len(common_videos[split])
        for i, (vname, species, vinfo) in enumerate(common_videos[split]):
            crops = vinfo.get("crops", [])
            feat = extract_video_feature(crops)
            if feat is not None:
                results[split]["X"].append(feat)
                results[split]["y"].append(species)
            if (i + 1) % 20 == 0 or (i + 1) == n_videos:
                elapsed = time.time() - start
                print(f"    {split}: {i+1}/{n_videos} ({elapsed:.1f}秒)")

    elapsed = time.time() - start
    print(f"  特徴抽出完了: {elapsed:.1f} 秒")

    out = {}
    for split in ["train", "test"]:
        X = np.array(results[split]["X"]) if results[split]["X"] else None
        y = np.array(results[split]["y"]) if results[split]["y"] else None
        out[f"X_{split}"] = X
        out[f"y_{split}"] = y
        if X is not None:
            print(f"  X_{split}: {X.shape}")
    return out


def evaluate_cross(train_data, test_data, direction_name):
    """クロスデータセット評価を実施し精度を返す。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    print(f"\n{'='*70}")
    print(f"クロスデータセット評価: {direction_name}")
    print(f"{'='*70}")

    X_train = train_data["X_train"]
    y_train = train_data["y_train"]
    X_test = test_data["X_test"]
    y_test = test_data["y_test"]

    # 学習・テストの両方に出現するラベルだけを採用
    le = LabelEncoder()
    all_labels = np.concatenate([y_train, y_test])
    le.fit(all_labels)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    n_classes = len(le.classes_)

    print(f"学習: {len(X_train)} サンプル, テスト: {len(X_test)} サンプル")
    print(f"クラス数: {n_classes}")

    train_counts = Counter(y_train.tolist())
    test_counts = Counter(y_test.tolist())

    print(f"\n種ごとのサンプル数:")
    print(f"{'種名':<28} {'学習':>6} {'テスト':>6}")
    print("-" * 45)
    for sp in sorted(le.classes_):
        print(f"{sp:<28} {train_counts.get(sp, 0):>6} "
              f"{test_counts.get(sp, 0):>6}")

    # 標準化（train で fit、test に適用）
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # LogReg
    print(f"\n--- LogReg ---")
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42,
                              solver="lbfgs", n_jobs=-1)
    clf.fit(X_train_s, y_train_enc)

    train_acc = clf.score(X_train_s, y_train_enc)
    print(f"学習セット精度: {train_acc*100:.2f}%")

    preds = clf.predict(X_test_s)
    acc = accuracy_score(y_test_enc, preds)
    f1 = f1_score(y_test_enc, preds, average="macro", zero_division=0)
    print(f"テスト精度: {acc*100:.2f}%")
    print(f"マクロ F1:  {f1:.4f}")

    # 種別の精度
    print(f"\n種ごとのテスト精度:")
    print(f"{'種名':<28} {'正解':>4} {'テスト':>6} {'精度':>8}")
    print("-" * 50)
    per_species = []
    for i, sp in enumerate(le.classes_):
        sp_mask = (y_test_enc == i)
        if sp_mask.sum() == 0:
            continue
        sp_correct = int((preds[sp_mask] == i).sum())
        sp_total = int(sp_mask.sum())
        sp_acc = sp_correct / sp_total
        per_species.append((sp, sp_acc, sp_total))
        print(f"{sp:<28} {sp_correct:>4} {sp_total:>6} {sp_acc*100:>7.1f}%")

    per_species.sort(key=lambda x: x[1])
    if per_species:
        print(f"\n最も精度が低い種:")
        for sp, sp_acc, sp_total in per_species[:5]:
            print(f"  {sp}: {sp_acc*100:.1f}% ({sp_total} サンプル)")

    return {
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "train_accuracy": float(train_acc),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_classes": n_classes,
        "per_species": {sp: {"accuracy": float(a), "n_test": int(n)}
                         for sp, a, n in per_species},
    }


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("=== クロスデータセット評価 ===")
    os.makedirs("../results/cross_dataset", exist_ok=True)

    overall_start = time.time()

    cub_common = load_cub200_common()
    if cub_common is None:
        print("CUB-200 の共通種データを抽出できません")
        return

    # VB100 の特徴量キャッシュ（再実行を高速化）
    vb_cache_path = "../results/cross_dataset/vb100_common_features.npz"
    if os.path.exists(vb_cache_path):
        print(f"\n--- VB100 特徴量キャッシュを再利用: {vb_cache_path} ---")
        vc = np.load(vb_cache_path, allow_pickle=True)
        vb100_common = {
            "X_train": vc["X_train"],
            "y_train": vc["y_train"],
            "X_test": vc["X_test"],
            "y_test": vc["y_test"],
        }
        print(f"  X_train: {vb100_common['X_train'].shape}, "
              f"X_test: {vb100_common['X_test'].shape}")
    else:
        vb_extracted = extract_vb100_common()
        if vb_extracted.get("X_train") is None:
            print("VB100 の共通種データを抽出できません")
            return
        np.savez(vb_cache_path,
                  X_train=vb_extracted["X_train"],
                  y_train=vb_extracted["y_train"],
                  X_test=vb_extracted["X_test"],
                  y_test=vb_extracted["y_test"])
        print(f"  保存: {vb_cache_path}")
        vb100_common = {
            "X_train": vb_extracted["X_train"],
            "y_train": vb_extracted["y_train"],
            "X_test": vb_extracted["X_test"],
            "y_test": vb_extracted["y_test"],
        }

    all_results = {}

    # ========================================
    # ベースライン: 同一データセット内（共通16種）
    # ========================================
    print(f"\n{'='*70}")
    print(f"ベースライン: 同一データセット内評価（共通16種のみ）")
    print(f"{'='*70}")

    all_results["cub200_internal"] = evaluate_cross(
        cub_common, cub_common, "CUB-200 内部（16種）"
    )
    all_results["vb100_internal"] = evaluate_cross(
        vb100_common, vb100_common, "VB100 内部（16種）"
    )

    # ========================================
    # クロス: CUB-200 → VB100
    # ========================================
    all_results["cub200_to_vb100"] = evaluate_cross(
        cub_common, vb100_common, "CUB-200 → VB100"
    )

    # ========================================
    # クロス: VB100 → CUB-200
    # ========================================
    all_results["vb100_to_cub200"] = evaluate_cross(
        vb100_common, cub_common, "VB100 → CUB-200"
    )

    # ========================================
    # 統合比較表
    # ========================================
    print(f"\n{'='*80}")
    print(f"統合比較表")
    print(f"{'='*80}")

    print(f"\n{'評価':<28} {'学習':>6} {'テスト':>6} {'Accuracy':>10} {'F1':>8}")
    print("-" * 65)
    for name, r in all_results.items():
        print(f"{name:<28} {r['n_train']:>6} {r['n_test']:>6} "
              f"{r['accuracy']*100:>9.2f}% {r['macro_f1']:>7.4f}")

    print(f"\n--- ドメインシフトの影響 ---")
    if "cub200_internal" in all_results and "cub200_to_vb100" in all_results:
        a = all_results["cub200_internal"]["accuracy"]
        b = all_results["cub200_to_vb100"]["accuracy"]
        print(f"CUB-200 → VB100: {a*100:.2f}% → {b*100:.2f}% "
              f"({(b-a)*100:+.2f}pp)")
    if "vb100_internal" in all_results and "vb100_to_cub200" in all_results:
        a = all_results["vb100_internal"]["accuracy"]
        b = all_results["vb100_to_cub200"]["accuracy"]
        print(f"VB100 → CUB-200: {a*100:.2f}% → {b*100:.2f}% "
              f"({(b-a)*100:+.2f}pp)")

    print(f"\n--- 全種評価との比較 ---")
    print(f"Phase 1 (CUB-200, 200種): 87.31%")
    print(f"Phase 5f (VB100, 100種):  89.05%")
    print(f"CUB-200 内部 (16種):      "
          f"{all_results['cub200_internal']['accuracy']*100:.2f}%")
    print(f"VB100 内部 (16種):        "
          f"{all_results['vb100_internal']['accuracy']*100:.2f}%")

    elapsed = time.time() - overall_start
    print(f"\n総処理時間: {elapsed:.1f} 秒")

    out_path = "../results/cross_dataset/cross_dataset_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"保存: {out_path}")


if __name__ == "__main__":
    main()
