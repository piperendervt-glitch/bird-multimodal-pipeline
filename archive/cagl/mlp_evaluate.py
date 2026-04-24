"""
段階C: MLP / XGBoost / CAGL V4 / 単純多数決 / 最良単独モデル の比較。

このスクリプトは core_cagl をインポートしない。
train_results.json, local_llm_summary.json, responses.json, questions.json
を読み込み、比較表とカテゴリ別正解率を出力する。
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
MLP_DIR = RESULTS_DIR / "mlp"
LOCAL_DIR = RESULTS_DIR / "local_llm"
EVAL_DIR = RESULTS_DIR / "evaluation"


def main():
    # MLP / XGBoost 結果
    with open(MLP_DIR / "train_results.json", "r", encoding="utf-8") as f:
        tr = json.load(f)
    n_q = tr["n_questions"]
    mlp_acc = tr["MLP"]["overall_accuracy"]
    xgb_acc = tr["XGBoost"]["overall_accuracy"]
    mlp_preds = np.array(tr["MLP"]["cv_preds"])
    xgb_preds = np.array(tr["XGBoost"]["cv_preds"])
    gt = np.array(tr["ground_truth"])
    cats = tr["categories_per_question"]
    feature_names = tr["feature_names"]
    mlp_imp = np.array(tr["MLP"]["feature_importance"])
    xgb_imp = np.array(tr["XGBoost"]["feature_importance"])

    # CAGL / 多数決 / 単独モデル
    with open(EVAL_DIR / "local_llm_summary.json", "r", encoding="utf-8") as f:
        local_sum = json.load(f)
    single_accs = local_sum["単独モデル正解率"]
    best_single = local_sum["最良単独モデル"]
    best_single_acc = single_accs[best_single]
    majority_acc = local_sum["単純多数決正解率"]
    v4_acc = local_sum["変種"]["V4_cagl"]["mean_accuracy"]

    print("=" * 70)
    print("統合層比較: MLP vs XGBoost vs CAGL vs 多数決 vs 最良単独")
    print("=" * 70)
    print(f"問題数（MLP/XGBoost の交差検証対象）: {n_q}")

    # 全体比較
    print()
    print("--- 全体正解率 ---")
    print(f"{'手法':<18} {'正解率':>8} {'vs 最良単独':>14} {'vs CAGL V4':>12}")
    rows = [
        ("最良単独(" + best_single + ")", best_single_acc, None, None),
        ("単純多数決", majority_acc, majority_acc - best_single_acc, None),
        ("CAGL V4", v4_acc, v4_acc - best_single_acc, None),
        ("MLP (5-fold CV)", mlp_acc, mlp_acc - best_single_acc, mlp_acc - v4_acc),
        ("XGBoost (5-fold CV)", xgb_acc, xgb_acc - best_single_acc, xgb_acc - v4_acc),
    ]
    for name, acc, d_best, d_v4 in rows:
        s1 = f"{d_best*100:+.2f}pp" if d_best is not None else "-"
        s2 = f"{d_v4*100:+.2f}pp" if d_v4 is not None else "-"
        print(f"  {name:<16} {acc*100:>7.2f}%  {s1:>12}  {s2:>10}")

    # カテゴリ別
    print()
    print("--- カテゴリ別正解率 ---")
    cats_arr = np.array(cats)
    cat_list = sorted(set(cats))
    # CAGL と最良単独のカテゴリ別正解率（local_llm_summary から）
    cagl_cat = local_sum.get("カテゴリ別", {})
    # 単純多数決のカテゴリ別正解率は、MLP/XGB の X から再計算できる:
    # 「vote_count_X」列の argmax に相当する。再計算するより、正解率を
    # local_llm から読み込めない場合はスキップ。

    print(f"  {'カテゴリ':<18} {'問題数':>5} "
          f"{'最良単独':>9} {'CAGL V4':>9} "
          f"{'MLP':>8} {'XGBoost':>9}")
    cat_rows = {}
    for c in cat_list:
        mask = (cats_arr == c)
        n_c = int(mask.sum())
        mlp_c = float((mlp_preds[mask] == gt[mask]).mean()) if n_c else 0.0
        xgb_c = float((xgb_preds[mask] == gt[mask]).mean()) if n_c else 0.0
        best_c = cagl_cat.get(c, {}).get("best_single_acc", None)
        v4_c = cagl_cat.get(c, {}).get("v4_cagl_acc", None)
        cat_rows[c] = {
            "n": n_c,
            "best_single_acc": best_c,
            "v4_cagl_acc": v4_c,
            "mlp_acc": mlp_c,
            "xgb_acc": xgb_c,
        }
        bs = f"{best_c*100:.1f}%" if best_c is not None else "-"
        vs = f"{v4_c*100:.1f}%" if v4_c is not None else "-"
        print(f"  {c:<18} {n_c:>5}  {bs:>8}  {vs:>8}  "
              f"{mlp_c*100:>7.1f}%  {xgb_c*100:>8.1f}%")

    # 改善/劣化の極値
    def diff_vs(target_key, base_key):
        diffs = []
        for c, row in cat_rows.items():
            a, b = row[target_key], row[base_key]
            if a is not None and b is not None:
                diffs.append((c, a - b, row["n"]))
        return diffs

    print()
    print("--- MLP の最良単独に対する改善/劣化 ---")
    diffs = sorted(diff_vs("mlp_acc", "best_single_acc"),
                   key=lambda x: x[1], reverse=True)
    print(f"  最も改善: {diffs[0][0]} ({diffs[0][1]*100:+.1f}pp, "
          f"n={diffs[0][2]})")
    print(f"  最も劣化: {diffs[-1][0]} ({diffs[-1][1]*100:+.1f}pp, "
          f"n={diffs[-1][2]})")

    print("--- XGBoost の最良単独に対する改善/劣化 ---")
    diffs = sorted(diff_vs("xgb_acc", "best_single_acc"),
                   key=lambda x: x[1], reverse=True)
    print(f"  最も改善: {diffs[0][0]} ({diffs[0][1]*100:+.1f}pp, "
          f"n={diffs[0][2]})")
    print(f"  最も劣化: {diffs[-1][0]} ({diffs[-1][1]*100:+.1f}pp, "
          f"n={diffs[-1][2]})")

    # 特徴量重要度 (上位10)
    def top_k(imp, names, k=10):
        order = np.argsort(imp)[::-1]
        return [(names[i], float(imp[i])) for i in order[:k]]

    print()
    print("--- MLP 特徴量重要度 (上位10) ---")
    for name, v in top_k(mlp_imp, feature_names, 10):
        print(f"  {name:<40} {v:.4f}")
    print()
    print("--- XGBoost 特徴量重要度 (上位10) ---")
    for name, v in top_k(xgb_imp, feature_names, 10):
        print(f"  {name:<40} {v:.4f}")

    # 判定
    print()
    print("--- 判定 ---")
    mlp_vs_best = mlp_acc - best_single_acc
    xgb_vs_best = xgb_acc - best_single_acc
    mlp_vs_v4 = mlp_acc - v4_acc
    xgb_vs_v4 = xgb_acc - v4_acc
    print(f"  MLP が最良単独を上回るか:        "
          f"{'YES' if mlp_vs_best > 0 else 'NO'} ({mlp_vs_best*100:+.2f}pp)")
    print(f"  XGBoost が最良単独を上回るか:    "
          f"{'YES' if xgb_vs_best > 0 else 'NO'} ({xgb_vs_best*100:+.2f}pp)")
    print(f"  MLP が CAGL V4 を上回るか:       "
          f"{'YES' if mlp_vs_v4 > 0 else 'NO'} ({mlp_vs_v4*100:+.2f}pp)")
    print(f"  XGBoost が CAGL V4 を上回るか:   "
          f"{'YES' if xgb_vs_v4 > 0 else 'NO'} ({xgb_vs_v4*100:+.2f}pp)")

    summary = {
        "評価日時": datetime.now(timezone.utc).isoformat(),
        "n_questions": n_q,
        "overall": {
            "最良単独": {"model": best_single, "acc": best_single_acc},
            "単純多数決": majority_acc,
            "CAGL_V4": v4_acc,
            "MLP": mlp_acc,
            "XGBoost": xgb_acc,
            "MLP_fold_accs": tr["MLP"]["fold_accs"],
            "XGBoost_fold_accs": tr["XGBoost"]["fold_accs"],
        },
        "カテゴリ別": cat_rows,
        "top_mlp_features": top_k(mlp_imp, feature_names, 15),
        "top_xgb_features": top_k(xgb_imp, feature_names, 15),
        "判定": {
            "MLP_beats_best_single": bool(mlp_vs_best > 0),
            "XGBoost_beats_best_single": bool(xgb_vs_best > 0),
            "MLP_beats_v4": bool(mlp_vs_v4 > 0),
            "XGBoost_beats_v4": bool(xgb_vs_v4 > 0),
        },
    }
    out_path = MLP_DIR / "comparison_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print()
    print(f"結果を保存: {out_path}")


if __name__ == "__main__":
    main()
