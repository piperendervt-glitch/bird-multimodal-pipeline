"""Phase 3 段階A: 特定種検出の対象種（5 種）を選択する。

DINOv2 特徴量 + ロジスティック回帰のクラス別 F1 をもとに、
低精度 2 種・中精度 1 種・高精度 2 種を層化サンプリングで決定論的に選択する。
"""

import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


PHASE1_DIR = os.path.join("..", "results", "bird_phase1")
PHASE3_DIR = os.path.join("..", "results", "bird_phase3")


def main():
    print("=== Phase 3A: 対象種の選択 ===")

    # メタデータ読み込み（クラス ID → 種名）
    with open(os.path.join(PHASE1_DIR, "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    classes = meta["classes"]  # キーは str（JSON 由来）

    # DINOv2 特徴量の読み込み
    dino = np.load(os.path.join(PHASE1_DIR, "features_dinov2_vits14.npz"))
    X_train = dino["X_train"]
    y_train = dino["y_train"]
    X_test = dino["X_test"]
    y_test = dino["y_test"]

    print(f"学習: {X_train.shape[0]} 枚, テスト: {X_test.shape[0]} 枚")
    print(f"クラス数: {len(classes)}")

    # クラス別 F1 を得るため Phase 1 の LogReg を再実行
    print("\nロジスティック回帰を学習中（クラス別 F1 の算出）...")
    lr = LogisticRegression(max_iter=2000, C=1.0, random_state=42, n_jobs=-1)
    lr.fit(X_train, y_train)
    preds = lr.predict(X_test)

    n_classes = len(classes)
    class_f1 = {}
    for cls_id in range(n_classes):
        y_binary_true = (y_test == cls_id).astype(int)
        y_binary_pred = (preds == cls_id).astype(int)
        f1 = f1_score(y_binary_true, y_binary_pred, zero_division=0)
        class_f1[cls_id] = float(f1)

    # F1 昇順ソート（同値はクラス ID で決定論的）
    sorted_classes = sorted(class_f1.items(), key=lambda x: (x[1], x[0]))

    # 層化サンプリング用の候補
    # 下位 1/4 から低精度、中央 1/3 から中精度、上位 1/4 から高精度
    candidates_low = [(c, f) for c, f in sorted_classes[: n_classes // 4] if f > 0]
    candidates_mid = [
        (c, f) for c, f in sorted_classes[n_classes // 3: 2 * n_classes // 3]
    ]
    candidates_high = [(c, f) for c, f in sorted_classes[3 * n_classes // 4:]]

    selected = []
    # 下位 2 種（F1 > 0 の中から最も低い 2 種）
    if len(candidates_low) >= 2:
        selected.extend(candidates_low[:2])
    # 中位 1 種（中央値付近）
    if len(candidates_mid) >= 1:
        selected.append(candidates_mid[len(candidates_mid) // 2])
    # 上位 2 種（最も高い 2 種）
    if len(candidates_high) >= 2:
        selected.extend(candidates_high[-2:])

    # 5 種に満たない場合は中位から補充
    while len(selected) < 5 and candidates_mid:
        c = candidates_mid.pop(0)
        if c not in selected:
            selected.append(c)

    # 表示
    print(f"\n選択された 5 種:")
    print(f"{'クラスID':<10} {'種名':<40} {'DINOv2 F1':>10} {'難易度':<8}")
    print("-" * 70)
    for cls_id, f1 in selected:
        name = classes.get(str(cls_id), "?")
        if f1 < 0.5:
            difficulty = "低精度"
        elif f1 < 0.8:
            difficulty = "中精度"
        else:
            difficulty = "高精度"
        print(f"{cls_id:<10} {name:<40} {f1:>10.4f} {difficulty:<8}")

    # CUB-200 名を Xeno-canto 検索用クエリに変換
    # 例: "001.Black_footed_Albatross" → "Black footed Albatross"
    selected_species = []
    for cls_id, f1 in selected:
        name = classes.get(str(cls_id), "?")
        clean_name = name.split(".", 1)[-1].replace("_", " ")
        selected_species.append({
            "class_id": int(cls_id),
            "cub_name": name,
            "search_name": clean_name,
            "dinov2_f1": float(f1),
        })

    os.makedirs(PHASE3_DIR, exist_ok=True)
    output = {
        "selected_species": selected_species,
        "selection_criteria": {
            "method": "DINOv2 F1 score による層化サンプリング",
            "low_accuracy": 2,
            "mid_accuracy": 1,
            "high_accuracy": 2,
        },
        "all_class_f1": {str(k): v for k, v in class_f1.items()},
    }
    out_path = os.path.join(PHASE3_DIR, "selected_species.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n保存: {out_path}")


if __name__ == "__main__":
    main()
