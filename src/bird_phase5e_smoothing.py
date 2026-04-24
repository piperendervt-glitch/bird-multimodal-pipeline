import numpy as np
import json
import os
import sys
import io
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score

# Windows cp932 対策
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


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
    print("=== Phase 5e-B: 時間方向の平滑化 ===")

    with open("../results/bird_phase5e/frame_predictions.json", encoding="utf-8") as f:
        video_preds = json.load(f)
    with open("../results/bird_phase4b/species_mapping.json", encoding="utf-8") as f:
        mapping = json.load(f)
    with open("../results/bird_phase4b/phase4b_results.json", encoding="utf-8") as f:
        phase4b_raw = json.load(f)

    id_to_species = mapping["id_to_species"]
    p4b_acc = phase4b_raw["results"]["visual_only"]["accuracy"]

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
        for video_name, vpred in video_preds.items():
            true_label = vpred["true_label"]
            frame_predictions = vpred["frame_predictions"]
            if not frame_predictions:
                continue
            pred_label, confidence = info["func"](frame_predictions)
            y_true.append(true_label)
            y_pred.append(pred_label)
            video_details.append({
                "video": video_name,
                "true": id_to_species[str(true_label)],
                "pred": id_to_species[str(pred_label)],
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

    # 結果サマリー
    print(f"\n{'='*70}")
    print("Phase 5e 結果サマリー")
    print(f"{'='*70}")
    print(f"{'手法':<35} {'Accuracy':>10} {'F1':>8} {'正解数':>10}")
    print(f"{'-'*70}")
    print(f"{'[Phase4b] フレーム平均+LogReg':<35} {p4b_acc*100:>9.2f}% {'':>8} {'':>10}")
    print(f"{'-'*70}")
    for name, r in results.items():
        print(f"{r['description']:<35} {r['accuracy']*100:>9.2f}% "
              f"{r['macro_f1']:>7.4f} {r['n_correct']:>5}/{r['n_total']}")

    # 142-mallard の予測
    print(f"\n--- 142-mallard の各手法予測 ---")
    for name, r in results.items():
        for d in r["details"]:
            if "142-mallard" in d["video"]:
                mark = "○" if d["correct"] else "×"
                print(f"  {r['description']:<30} → {d['pred']:<22} "
                      f"(確信度 {d['confidence']:.4f}) {mark}")

    # 誤分類一覧
    print(f"\n--- 各手法の誤分類一覧 ---")
    for name, r in results.items():
        errors = [d for d in r["details"] if not d["correct"]]
        if errors:
            print(f"\n  {r['description']}:")
            for d in errors:
                print(f"    {d['video']}: {d['true']} → {d['pred']} "
                      f"(確信度 {d['confidence']:.4f})")
        else:
            print(f"\n  {r['description']}: 誤分類なし (100%)")

    # フレーム単位の安定性
    print(f"\n--- フレーム単位の予測安定性（動画内一致率）---")
    stabilities = []
    unstable_videos = []
    for video_name, vpred in video_preds.items():
        preds = [fp["predicted_label"] for fp in vpred["frame_predictions"]]
        if not preds:
            continue
        counter = Counter(preds)
        agreement = counter.most_common(1)[0][1] / len(preds)
        stabilities.append(agreement)
        if agreement < 0.7:
            unstable_videos.append((video_name, agreement, vpred["species"]))
    print(f"  平均一致率: {np.mean(stabilities)*100:.1f}%")
    print(f"  中央値:     {np.median(stabilities)*100:.1f}%")
    print(f"  最小一致率: {np.min(stabilities)*100:.1f}%")
    print(f"  最大一致率: {np.max(stabilities)*100:.1f}%")
    print(f"  一致率 < 70% の動画数: {len(unstable_videos)}")
    if unstable_videos:
        print(f"  不安定な動画:")
        for vn, ag, sp in unstable_videos:
            print(f"    {vn} ({sp}): 一致率 {ag:.0%}")

    # 失敗条件の判定
    print(f"\n--- 失敗条件の判定 ---")
    best_method = max(results.items(), key=lambda x: x[1]["accuracy"])
    best_acc = best_method[1]["accuracy"]
    if best_acc < p4b_acc:
        print(f"  NG: 最良の平滑化 ({best_method[1]['description']}: "
              f"{best_acc*100:.2f}%) < Phase 4b ({p4b_acc*100:.2f}%)")
    elif best_acc == p4b_acc:
        print(f"  同等: 最良の平滑化 ({best_method[1]['description']}: "
              f"{best_acc*100:.2f}%) = Phase 4b ({p4b_acc*100:.2f}%)")
    else:
        diff = best_acc - p4b_acc
        print(f"  OK: 最良の平滑化 ({best_method[1]['description']}: "
              f"{best_acc*100:.2f}%) vs Phase 4b ({p4b_acc*100:.2f}%): "
              f"{diff*100:+.2f}pp")

    # 142-mallard が修正されたか
    print()
    mallard_fixed_methods = []
    for name, r in results.items():
        for d in r["details"]:
            if "142-mallard" in d["video"] and d["correct"]:
                mallard_fixed_methods.append(r["description"])
    if mallard_fixed_methods:
        print(f"  OK: 142-mallard が以下の手法で修正された:")
        for m in mallard_fixed_methods:
            print(f"    - {m}")
    else:
        print(f"  NG: 142-mallard は全手法で修正されなかった")

    # 保存
    save_results = {k: {kk: vv for kk, vv in v.items() if kk != "details"}
                    for k, v in results.items()}
    save_results["phase4b_baseline"] = p4b_acc
    save_results["frame_stability"] = {
        "mean": float(np.mean(stabilities)),
        "median": float(np.median(stabilities)),
        "min": float(np.min(stabilities)),
        "max": float(np.max(stabilities)),
        "low_agreement_count": len(unstable_videos),
    }
    with open("../results/bird_phase5e/phase5e_results.json", "w", encoding="utf-8") as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    print(f"\n保存: results/bird_phase5e/phase5e_results.json")

    with open("../results/bird_phase5e/phase5e_details.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"保存: results/bird_phase5e/phase5e_details.json")


if __name__ == "__main__":
    main()
