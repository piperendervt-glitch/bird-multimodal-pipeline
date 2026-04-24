"""Phase 4b 段階D: 分類器学習 + 4 手法比較。

- 映像のみ (DINOv2 384次元)
- 音声のみ (BirdNET 15次元)
- 映像+音声統合 (404次元)
- 映像+ゼロ音声 (音声次元をノイズ化した対照)

DINOv2 / YOLO / BirdNET をインポートしない。特徴量ファイルのみ使用。
"""

import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report


OUT_DIR = os.path.join("..", "results", "bird_phase4b")


def evaluate(X_train, y_train, X_test, y_test, n_classes):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    lr = LogisticRegression(
        max_iter=2000, C=1.0, random_state=42,
        class_weight="balanced", n_jobs=-1,
    )
    lr.fit(X_train_s, y_train)
    preds = lr.predict(X_test_s)

    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "macro_f1": float(f1_score(y_test, preds, average="macro",
                                   zero_division=0)),
        "predictions": preds.tolist(),
    }


def main():
    print("=== Phase 4b-D: 分類器学習 + 全手法比較 ===")

    train = np.load(os.path.join(OUT_DIR, "features_train_set.npz"),
                    allow_pickle=True)
    test = np.load(os.path.join(OUT_DIR, "features_test_set.npz"),
                   allow_pickle=True)

    val_path = os.path.join(OUT_DIR, "features_val_set.npz")
    if os.path.exists(val_path):
        val = np.load(val_path, allow_pickle=True)
        X_train_visual = np.concatenate([train["X_visual"], val["X_visual"]])
        X_train_audio = np.concatenate([train["X_audio"], val["X_audio"]])
        X_train_combined = np.concatenate([train["X_combined"], val["X_combined"]])
        y_train = np.concatenate([train["y"], val["y"]])
        train_n = len(train["y"])
        val_n = len(val["y"])
        print(f"学習データ: train {train_n} + val {val_n} = {len(y_train)} 動画")
    else:
        X_train_visual = train["X_visual"]
        X_train_audio = train["X_audio"]
        X_train_combined = train["X_combined"]
        y_train = train["y"]
        print(f"学習データ: train {len(y_train)} 動画 (val は未検出)")

    X_test_visual = test["X_visual"]
    X_test_audio = test["X_audio"]
    X_test_combined = test["X_combined"]
    y_test = test["y"]
    print(f"テストデータ: {len(y_test)} 動画")

    with open(os.path.join(OUT_DIR, "species_mapping.json"), encoding="utf-8") as f:
        mapping = json.load(f)
    n_classes = mapping["n_classes"]
    print(f"クラス数: {n_classes}")

    results = {}

    print(f"\n--- 映像のみ (DINOv2, {X_train_visual.shape[1]} 次元) ---")
    r = evaluate(X_train_visual, y_train, X_test_visual, y_test, n_classes)
    results["visual_only"] = r
    print(f"  Accuracy: {r['accuracy']*100:.2f}%, F1: {r['macro_f1']:.4f}")

    print(f"\n--- 音声のみ (BirdNET, {X_train_audio.shape[1]} 次元) ---")
    r = evaluate(X_train_audio, y_train, X_test_audio, y_test, n_classes)
    results["audio_only"] = r
    print(f"  Accuracy: {r['accuracy']*100:.2f}%, F1: {r['macro_f1']:.4f}")

    print(f"\n--- 映像 + 音声 統合 ({X_train_combined.shape[1]} 次元) ---")
    r = evaluate(X_train_combined, y_train, X_test_combined, y_test, n_classes)
    results["combined"] = r
    print(f"  Accuracy: {r['accuracy']*100:.2f}%, F1: {r['macro_f1']:.4f}")

    print(f"\n--- 映像 + ゼロ音声（音声情報なし対照） ---")
    audio_dim = X_train_audio.shape[1]
    meta_dim = X_train_combined.shape[1] - X_train_visual.shape[1] - audio_dim
    X_train_zero = np.concatenate([
        X_train_visual,
        np.zeros_like(X_train_audio),
        np.zeros((len(y_train), meta_dim), dtype=np.float32),
    ], axis=1)
    X_test_zero = np.concatenate([
        X_test_visual,
        np.zeros_like(X_test_audio),
        np.zeros((len(y_test), meta_dim), dtype=np.float32),
    ], axis=1)
    r = evaluate(X_train_zero, y_train, X_test_zero, y_test, n_classes)
    results["visual_zero_audio"] = r
    print(f"  Accuracy: {r['accuracy']*100:.2f}%, F1: {r['macro_f1']:.4f}")

    print(f"\n{'='*60}")
    print(f"Phase 4b 結果サマリー")
    print(f"{'='*60}")
    print(f"{'手法':<35} {'Accuracy':>10} {'F1':>8}")
    print(f"{'-'*60}")
    labels = {
        "visual_only": "映像のみ (DINOv2)",
        "audio_only": "音声のみ (BirdNET)",
        "combined": "映像 + 音声 統合",
        "visual_zero_audio": "映像 + ゼロ音声 (対照)",
    }
    for key in ["visual_only", "audio_only", "combined", "visual_zero_audio"]:
        r = results[key]
        print(f"{labels[key]:<35} {r['accuracy']*100:>9.2f}% {r['macro_f1']:>7.4f}")

    print(f"\n--- 失敗条件の判定 ---")
    visual_acc = results["visual_only"]["accuracy"]
    combined_acc = results["combined"]["accuracy"]
    zero_acc = results["visual_zero_audio"]["accuracy"]

    # 判定 1: 映像のみ >= 50%
    if visual_acc < 0.50:
        print(f"  NG: 映像のみ正解率 {visual_acc*100:.2f}% < 50%")
    else:
        print(f"  OK: 映像のみ正解率 {visual_acc*100:.2f}% >= 50%")

    # 判定 2: 統合が映像のみを 3pp 以上下回らない
    diff = combined_acc - visual_acc
    if diff < -0.03:
        print(f"  NG: 統合が映像のみより {diff*100:.2f}pp 低下 (> 3pp)")
    else:
        print(f"  OK: 統合 vs 映像のみ: {diff*100:+.2f}pp")

    # 判定 3: ゼロ音声パディングによる影響
    zero_diff = zero_acc - visual_acc
    note = "音声次元がノイズ" if zero_diff < -0.02 else "ほぼ影響なし"
    print(f"  参考: 映像+ゼロ音声 vs 映像のみ: {zero_diff*100:+.2f}pp ({note})")

    audio_acc = results["audio_only"]["accuracy"]
    chance = 1.0 / n_classes
    print(f"\n  音声のみ正解率: {audio_acc*100:.2f}% "
          f"(チャンスレート {chance*100:.2f}%)")

    # 詳細分類レポート（映像のみ）
    print(f"\n--- 映像のみ: 種別分類結果 ---")
    id_to_species = mapping["id_to_species"]
    target_names = [id_to_species[str(i)] for i in range(n_classes)]
    present_labels = sorted(set(y_test.tolist()) | set(results["visual_only"]["predictions"]))
    present_names = [id_to_species[str(i)] for i in present_labels]
    print(classification_report(
        y_test, results["visual_only"]["predictions"],
        labels=present_labels, target_names=present_names,
        zero_division=0,
    ))

    # 保存
    out_path = os.path.join(OUT_DIR, "phase4b_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "n_classes": n_classes,
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n保存: {out_path}")


if __name__ == "__main__":
    main()
