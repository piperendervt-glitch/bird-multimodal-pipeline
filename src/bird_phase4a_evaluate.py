"""Phase 4a 段階C: 元画像 / YOLO 切り抜き / 正解 bbox 切り抜きの精度比較。

DINOv2 や YOLO はインポートしない。特徴量ファイルのみ使用。
"""

import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score


PHASE1_DIR = os.path.join("..", "results", "bird_phase1")
PHASE4A_DIR = os.path.join("..", "results", "bird_phase4a")


def evaluate(X_train, y_train, X_test, y_test, n_classes):
    lr = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
    lr.fit(X_train, y_train)
    preds = lr.predict(X_test)
    probs = lr.predict_proba(X_test)
    return {
        "top1": float(accuracy_score(y_test, preds)),
        "top5": float(top_k_accuracy_score(
            y_test, probs, k=5, labels=list(range(n_classes)))),
        "f1": float(f1_score(y_test, preds, average="macro", zero_division=0)),
    }


def main():
    print("=== Phase 4a-C: 切り抜き精度比較 ===")

    train = np.load(os.path.join(PHASE4A_DIR, "features_train.npz"))
    test = np.load(os.path.join(PHASE4A_DIR, "features_test.npz"))

    with open(os.path.join(PHASE1_DIR, "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    n_classes = meta["n_classes"]

    with open(os.path.join(PHASE4A_DIR, "detection_results.json"),
              encoding="utf-8") as f:
        det = json.load(f)

    print(f"YOLO 検出率: {det['detection_rate']*100:.2f}%")
    print(f"平均 IoU: {det['mean_iou']:.4f}")
    print(f"推論速度: {det.get('images_per_sec', 0):.2f} 枚/秒")

    patterns = [
        ("original", "X_original", "元画像（Phase 1 相当）"),
        ("yolo_crop", "X_yolo_crop", "YOLO 切り抜き"),
        ("gt_crop", "X_gt_crop", "正解 bbox 切り抜き（上限）"),
    ]

    results = {}
    for key, feat_key, label in patterns:
        print(f"\n--- {label} ---")
        r = evaluate(train[feat_key], train["y"],
                     test[feat_key], test["y"], n_classes)
        results[key] = r
        print(f"  Top-1: {r['top1']*100:.2f}%, "
              f"Top-5: {r['top5']*100:.2f}%, F1: {r['f1']:.4f}")

    print(f"\n{'='*60}")
    print(f"Phase 4a 結果サマリー")
    print(f"{'='*60}")
    print(f"{'パターン':<35} {'Top-1':>8} {'Top-5':>8} {'F1':>8}")
    print(f"{'-'*60}")
    for key, _, label in patterns:
        r = results[key]
        print(f"{label:<35} {r['top1']*100:>7.2f}% "
              f"{r['top5']*100:>7.2f}% {r['f1']:>7.4f}")

    print(f"\n--- Phase 1 との比較 ---")
    phase1_top1 = 0.8773
    yolo_top1 = results["yolo_crop"]["top1"]
    orig_top1 = results["original"]["top1"]
    gt_top1 = results["gt_crop"]["top1"]
    diff_yolo = yolo_top1 - phase1_top1
    diff_orig = orig_top1 - phase1_top1
    print(f"  Phase 1 (報告値):         {phase1_top1*100:.2f}%")
    print(f"  本実験 元画像:            {orig_top1*100:.2f}% ({diff_orig*100:+.2f}pp)")
    print(f"  YOLO 切り抜き:            {yolo_top1*100:.2f}% ({diff_yolo*100:+.2f}pp)")
    print(f"  正解 bbox 切り抜き (上限): {gt_top1*100:.2f}% "
          f"({(gt_top1-phase1_top1)*100:+.2f}pp)")

    print(f"\n--- 失敗条件の判定 ---")
    if det["detection_rate"] < 0.80:
        print(f"  NG: 検出率 {det['detection_rate']*100:.2f}% < 80%")
    else:
        print(f"  OK: 検出率 {det['detection_rate']*100:.2f}% >= 80%")

    if diff_yolo < -0.05:
        print(f"  NG: YOLO 切り抜きで精度低下 {diff_yolo*100:.2f}pp (>5pp 低下)")
    else:
        print(f"  OK: YOLO 切り抜きの精度変化 {diff_yolo*100:+.2f}pp (許容範囲)")

    # YOLO vs 元画像（本実験内で直接比較）
    diff_within = yolo_top1 - orig_top1
    print(f"\n  本実験内 YOLO vs 元画像: {diff_within*100:+.2f}pp")

    output = {
        "detection_rate": det["detection_rate"],
        "mean_iou": det["mean_iou"],
        "images_per_sec": det.get("images_per_sec", 0),
        "results": results,
        "phase1_baseline_reported": phase1_top1,
    }
    out_path = os.path.join(PHASE4A_DIR, "phase4a_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n保存: {out_path}")


if __name__ == "__main__":
    main()
