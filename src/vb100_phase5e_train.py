"""VB100 Phase 5e 段階C: 時間方向の平滑化 7 手法 + フレーム平均ベースライン比較。

Phase 5e (bird_phase5e_smoothing.py) と同じ 7 手法を VB100 に適用。
さらに動画特徴量平均 + LogReg (Phase 4b 方式) のベースラインと、
先行研究 (VB100: YOLO+ResNet-50+多数決 = 84%) および WetlandBirds の
Phase 5e 結果との比較も出力する。

DINOv2 や YOLO はインポートしない。JSON のみを使用する。
"""

import numpy as np
import json
import os
import sys
import io
from collections import Counter, defaultdict
from sklearn.metrics import accuracy_score, f1_score

# Windows cp932 対策
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


RESULTS_DIR = os.path.join("..", "results", "vb100_phase5e")

# 先行研究値
VB100_PRIOR_ACCURACY = 0.84  # YOLO + ResNet-50 + 多数決

# WetlandBirds Phase 5e 結果 (bird_phase5e/phase5e_results.json より)
WETLAND_RESULTS = {
    "frame_avg": 0.9629629629629629,        # Phase 4b baseline (visual_only)
    "majority_vote": 1.0,
    "confidence_weighted": 0.9629629629629629,
    "exponential_decay": 0.9259259259259259,
    "sliding_window_3": 1.0,
    "sliding_window_5": 1.0,
    "temporal_gate": 0.9259259259259259,
    "yolo_weighted": 1.0,
}


def method_majority_vote(frame_preds):
    """多数決: 最も多く予測された種を採用"""
    labels = [fp["predicted_label"] for fp in frame_preds]
    if not labels:
        return -1, 0
    counter = Counter(labels)
    best_label, count = counter.most_common(1)[0]
    confidence = count / len(labels)
    return best_label, confidence


def method_confidence_weighted_vote(frame_preds):
    """確信度加重投票: 各フレームの確信度分布を平均"""
    if not frame_preds:
        return -1, 0
    n_classes = len(frame_preds[0]["prob_distribution"])
    weighted = np.zeros(n_classes)
    for fp in frame_preds:
        weighted += np.array(fp["prob_distribution"])
    weighted /= len(frame_preds)
    best_label = int(weighted.argmax())
    confidence = float(weighted.max())
    return best_label, confidence


def method_exponential_decay(frame_preds, decay=0.8):
    """指数減衰: 新しいフレームほど重みが大きい"""
    if not frame_preds:
        return -1, 0
    n_classes = len(frame_preds[0]["prob_distribution"])
    weighted = np.zeros(n_classes)
    n = len(frame_preds)
    for i, fp in enumerate(frame_preds):
        weight = decay ** (n - 1 - i)
        weighted += weight * np.array(fp["prob_distribution"])
    weighted /= weighted.sum() + 1e-9
    best_label = int(weighted.argmax())
    confidence = float(weighted.max())
    return best_label, confidence


def method_sliding_window_consensus(frame_preds, window_size=5):
    """スライディング窓の多数決を集約"""
    if not frame_preds or len(frame_preds) < window_size:
        return method_majority_vote(frame_preds)
    window_results = []
    for i in range(len(frame_preds) - window_size + 1):
        window = frame_preds[i:i + window_size]
        labels = [fp["predicted_label"] for fp in window]
        counter = Counter(labels)
        best, _ = counter.most_common(1)[0]
        window_results.append(best)
    counter = Counter(window_results)
    best_label, count = counter.most_common(1)[0]
    confidence = count / len(window_results)
    return best_label, confidence


def method_temporal_gate(frame_preds, success_rate=0.1, failure_rate=0.7):
    """時間方向の Gate 更新（CAGL 類似）"""
    if not frame_preds:
        return -1, 0
    n_classes = len(frame_preds[0]["prob_distribution"])
    gate = np.full(n_classes, 0.5)
    for fp in frame_preds:
        pred = fp["predicted_label"]
        conf = fp["confidence"]
        gate[pred] += success_rate * conf * (1 - gate[pred])
        for j in range(n_classes):
            if j != pred:
                gate[j] *= failure_rate
    best_label = int(gate.argmax())
    confidence = float(gate.max())
    return best_label, confidence


def method_yolo_weighted(frame_preds):
    """YOLO 検出確信度が高いフレームを重視"""
    if not frame_preds:
        return -1, 0
    n_classes = len(frame_preds[0]["prob_distribution"])
    weighted = np.zeros(n_classes)
    total_weight = 0
    for fp in frame_preds:
        probs = np.array(fp["prob_distribution"])
        yolo_conf = fp.get("yolo_confidence", 0)
        is_fallback = fp.get("is_fallback", False)
        weight = yolo_conf if (not is_fallback and yolo_conf > 0) else 0.1
        weighted += weight * probs
        total_weight += weight
    if total_weight > 0:
        weighted /= total_weight
    best_label = int(weighted.argmax())
    confidence = float(weighted.max())
    return best_label, confidence


def main():
    print("=== VB100 Phase 5e 段階C: 時間方向の平滑化 + 比較 ===")

    with open(os.path.join(RESULTS_DIR, "frame_predictions.json"),
              encoding="utf-8") as f:
        video_preds = json.load(f)
    with open(os.path.join(RESULTS_DIR, "species_mapping.json"),
              encoding="utf-8") as f:
        mapping = json.load(f)
    with open(os.path.join(RESULTS_DIR, "phase4b_baseline.json"),
              encoding="utf-8") as f:
        baseline = json.load(f)

    id_to_species = mapping["id_to_species"]

    methods = {
        "majority_vote": {
            "func": method_majority_vote,
            "description": "多数決",
        },
        "confidence_weighted": {
            "func": method_confidence_weighted_vote,
            "description": "確信度加重投票",
        },
        "exponential_decay": {
            "func": lambda fp: method_exponential_decay(fp, decay=0.8),
            "description": "指数減衰 (decay=0.8)",
        },
        "sliding_window_3": {
            "func": lambda fp: method_sliding_window_consensus(fp, window_size=3),
            "description": "スライディング窓 (size=3)",
        },
        "sliding_window_5": {
            "func": lambda fp: method_sliding_window_consensus(fp, window_size=5),
            "description": "スライディング窓 (size=5)",
        },
        "temporal_gate": {
            "func": method_temporal_gate,
            "description": "時間 Gate (CAGL 類似)",
        },
        "yolo_weighted": {
            "func": method_yolo_weighted,
            "description": "YOLO 確信度加重",
        },
    }

    results = {}
    for method_name, info in methods.items():
        y_true, y_pred = [], []
        video_details = []
        for video_key, vpred in video_preds.items():
            true_label = vpred["true_label"]
            frame_predictions = vpred["frame_predictions"]
            if not frame_predictions:
                continue
            pred_label, confidence = info["func"](frame_predictions)
            y_true.append(true_label)
            y_pred.append(pred_label)
            video_details.append({
                "video": video_key,
                "true": id_to_species[str(true_label)],
                "pred": id_to_species[str(pred_label)] if pred_label >= 0 else "-",
                "correct": pred_label == true_label,
                "confidence": confidence,
            })
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        results[method_name] = {
            "description": info["description"],
            "accuracy": float(acc),
            "macro_f1": float(f1),
            "n_correct": sum(1 for d in video_details if d["correct"]),
            "n_total": len(video_details),
            "details": video_details,
        }

    # =========================
    # サマリー表
    # =========================
    n_total_eval = baseline["n_total"]
    print(f"\n{'='*72}")
    print("=== VB100 Phase 5e 結果サマリー ===")
    print(f"{'='*72}")
    print(f"{'手法':<35} {'Accuracy':>10} {'F1':>8} {'正解数':>10}")
    print(f"{'-'*72}")
    print(f"{'[ベースライン] フレーム平均':<35} "
          f"{baseline['accuracy']*100:>9.2f}% "
          f"{baseline['macro_f1']:>7.4f} "
          f"{baseline['n_correct']:>5}/{baseline['n_total']}")
    print(f"{'-'*72}")
    for name, r in results.items():
        print(f"{r['description']:<35} {r['accuracy']*100:>9.2f}% "
              f"{r['macro_f1']:>7.4f} {r['n_correct']:>5}/{r['n_total']}")

    # 最良手法
    best_method_name, best_r = max(results.items(),
                                   key=lambda x: x[1]["accuracy"])
    best_acc = best_r["accuracy"]
    # ベースラインも候補に入れる（8手法比較）
    if baseline["accuracy"] > best_acc:
        best_overall_name = "frame_avg"
        best_overall_desc = "[ベースライン] フレーム平均"
        best_overall_acc = baseline["accuracy"]
    else:
        best_overall_name = best_method_name
        best_overall_desc = best_r["description"]
        best_overall_acc = best_acc

    # =========================
    # 先行研究比較
    # =========================
    print(f"\n{'='*72}")
    print("先行研究比較:")
    print(f"  VB100 先行研究 (YOLO+ResNet-50+多数決): "
          f"{VB100_PRIOR_ACCURACY*100:.2f}%")
    print(f"  本パイプライン (YOLO+DINOv2+{best_overall_desc}): "
          f"{best_overall_acc*100:.2f}%")
    diff = best_overall_acc - VB100_PRIOR_ACCURACY
    sign = "+" if diff >= 0 else ""
    print(f"  差分: {sign}{diff*100:.2f}pp")

    # =========================
    # WetlandBirds 比較表
    # =========================
    print(f"\n{'='*72}")
    print("=== WetlandBirds vs VB100 比較 ===")
    print(f"{'='*72}")
    print(f"{'手法':<28} {'WetlandBirds (13種)':<22} {'VB100 (100種)':<15}")
    print(f"{'-'*72}")
    rows = [
        ("フレーム平均", WETLAND_RESULTS["frame_avg"], baseline["accuracy"]),
        ("多数決", WETLAND_RESULTS["majority_vote"],
         results["majority_vote"]["accuracy"]),
        ("確信度加重投票", WETLAND_RESULTS["confidence_weighted"],
         results["confidence_weighted"]["accuracy"]),
        ("指数減衰 (decay=0.8)", WETLAND_RESULTS["exponential_decay"],
         results["exponential_decay"]["accuracy"]),
        ("スライディング窓 (size=3)", WETLAND_RESULTS["sliding_window_3"],
         results["sliding_window_3"]["accuracy"]),
        ("スライディング窓 (size=5)", WETLAND_RESULTS["sliding_window_5"],
         results["sliding_window_5"]["accuracy"]),
        ("時間 Gate (CAGL 類似)", WETLAND_RESULTS["temporal_gate"],
         results["temporal_gate"]["accuracy"]),
        ("YOLO 確信度加重", WETLAND_RESULTS["yolo_weighted"],
         results["yolo_weighted"]["accuracy"]),
    ]
    for desc, w, v in rows:
        print(f"{desc:<28} {w*100:>8.2f}%{'':<13} {v*100:>8.2f}%")

    # =========================
    # 最も精度が悪かった種 Top-5 (最良手法ベース)
    # =========================
    print(f"\n{'='*72}")
    print(f"最も精度が悪かった種 Top-5 (基準手法: {best_overall_desc}, "
          f"テストセット)")
    print(f"{'='*72}")
    # 最良手法の詳細から種ごとの正解率を集計
    if best_overall_name == "frame_avg":
        # ベースラインは per-video の詳細を保存していないので、
        # 代わりに最良の平滑化手法 (confidence_weighted 等) の details を使う
        top_details = best_r["details"]
        print(f"  (フレーム平均の動画別詳細は未保存のため、"
              f"平滑化最良手法 [{best_r['description']}] で代替表示)")
    else:
        top_details = results[best_overall_name]["details"]

    species_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for d in top_details:
        species_stats[d["true"]]["total"] += 1
        if d["correct"]:
            species_stats[d["true"]]["correct"] += 1

    species_acc = []
    for sp, st in species_stats.items():
        if st["total"] == 0:
            continue
        acc = st["correct"] / st["total"]
        species_acc.append((sp, acc, st["correct"], st["total"]))

    # 精度昇順、次に動画数降順で並べて Top-5
    species_acc.sort(key=lambda x: (x[1], -x[3]))
    print(f"{'順位':<4} {'種名':<35} {'精度':>8} {'正解/全体':<10}")
    print(f"{'-'*72}")
    for rank, (sp, acc, c, t) in enumerate(species_acc[:5], 1):
        print(f"{rank:<4} {sp:<35} {acc*100:>7.2f}%  {c}/{t}")

    # =========================
    # 処理時間サマリー (ベースライン JSON に含まれる値)
    # =========================
    print(f"\n{'='*72}")
    print("データ / 分割サマリー")
    print(f"{'='*72}")
    print(f"  種数 (クラス数):       {len(mapping['species_to_id'])}")
    print(f"  学習動画数:            {baseline['n_train_videos']}")
    print(f"  テスト動画数:          {baseline['n_test_videos']}")
    print(f"  学習フレーム数:        {baseline['n_train_frames']}")
    print(f"  テストフレーム数:      {baseline['n_test_frames']}")
    print(f"  フレーム分類器学習精度 (学習セット): "
          f"{baseline['train_accuracy_frame_level']*100:.2f}%")
    print(f"  動画単位分類器学習精度 (学習セット): "
          f"{baseline['train_accuracy_video_level']*100:.2f}%")
    print(f"  seed: {baseline['seed']}, train_ratio: {baseline['train_ratio']}")

    # =========================
    # 保存
    # =========================
    save_results = {k: {kk: vv for kk, vv in v.items() if kk != "details"}
                    for k, v in results.items()}
    save_results["frame_avg_baseline"] = {
        "description": "[ベースライン] フレーム平均 + LogReg (Phase 4b 方式)",
        "accuracy": baseline["accuracy"],
        "macro_f1": baseline["macro_f1"],
        "n_correct": baseline["n_correct"],
        "n_total": baseline["n_total"],
    }
    save_results["vb100_prior_work"] = {
        "description": "先行研究 VB100 (YOLO+ResNet-50+多数決)",
        "accuracy": VB100_PRIOR_ACCURACY,
    }
    save_results["best_method"] = {
        "name": best_overall_name,
        "description": best_overall_desc,
        "accuracy": best_overall_acc,
    }
    save_results["wetland_comparison"] = {
        desc: {"wetlandbirds": w, "vb100": v}
        for desc, w, v in rows
    }
    save_results["worst_species_top5"] = [
        {"species": sp, "accuracy": a, "n_correct": c, "n_total": t}
        for sp, a, c, t in species_acc[:5]
    ]
    save_results["data_summary"] = {
        "n_classes": len(mapping["species_to_id"]),
        "n_train_videos": baseline["n_train_videos"],
        "n_test_videos": baseline["n_test_videos"],
        "n_train_frames": baseline["n_train_frames"],
        "n_test_frames": baseline["n_test_frames"],
        "train_accuracy_frame_level": baseline["train_accuracy_frame_level"],
        "train_accuracy_video_level": baseline["train_accuracy_video_level"],
    }

    with open(os.path.join(RESULTS_DIR, "vb100_results.json"), "w",
              encoding="utf-8") as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    print(f"\n保存: results/vb100_phase5e/vb100_results.json")

    with open(os.path.join(RESULTS_DIR, "vb100_details.json"), "w",
              encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"保存: results/vb100_phase5e/vb100_details.json")

    # =========================
    # 失敗条件の判定
    # =========================
    print(f"\n--- 判定 ---")
    if best_overall_acc >= VB100_PRIOR_ACCURACY:
        diff = best_overall_acc - VB100_PRIOR_ACCURACY
        print(f"  OK: 本パイプライン ({best_overall_desc}: "
              f"{best_overall_acc*100:.2f}%) "
              f">= 先行研究 ({VB100_PRIOR_ACCURACY*100:.2f}%), "
              f"+{diff*100:.2f}pp")
    else:
        diff = VB100_PRIOR_ACCURACY - best_overall_acc
        print(f"  NG: 本パイプライン ({best_overall_desc}: "
              f"{best_overall_acc*100:.2f}%) "
              f"< 先行研究 ({VB100_PRIOR_ACCURACY*100:.2f}%), "
              f"-{diff*100:.2f}pp")


if __name__ == "__main__":
    main()
