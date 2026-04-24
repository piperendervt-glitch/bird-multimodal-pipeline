"""
段階C: LLM アンサンブル CAGL の評価。

results/llm_benchmark/ と results/llm_predictions/ を読み込み、マクロF1 /
Cohen's d / 信頼区間 / p値 を計算。加えて argmax ベースの正解率、単純
多数決の正解率、単独モデル正解率、カテゴリ別正解率を出力する。
V4 の学習済み Weight 差別化度合いも診断する。

このスクリプトは core_cagl をインポートしない。
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel
from sklearn.metrics import f1_score


try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
LLM_DIR = RESULTS_DIR / "llm_benchmark"
LLM_PRED_DIR = RESULTS_DIR / "llm_predictions"
EVAL_DIR = RESULTS_DIR / "evaluation"

VARIANT_NAMES = ["V1_weight_only", "V2_gate_only", "V3_both_gt", "V4_cagl"]
STD_THRESHOLD = 0.05


def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def macro_f1(gt_eval, preds):
    n_labels = gt_eval.shape[1]
    return float(np.mean([
        f1_score(gt_eval[:, l], preds[:, l], zero_division=0)
        for l in range(n_labels)
    ]))


def cohens_d(x, y):
    diff = np.asarray(x) - np.asarray(y)
    if diff.std() < 1e-12:
        return 0.0
    return float(diff.mean() / diff.std())


def paired_bootstrap_ci(x, y, n_bootstrap=10000, alpha=0.05, seed=99):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    diff = x - y
    n = len(diff)
    boot = np.array([
        rng.choice(diff, size=n, replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    return float(np.percentile(boot, 100 * alpha / 2)), \
           float(np.percentile(boot, 100 * (1 - alpha / 2)))


def format_row(values, width=6, prec=3):
    return "[" + ", ".join(f"{v:>{width}.{prec}f}" for v in values) + "]"


def main():
    manifest_path = LLM_DIR / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # データ読み込み
    data = np.load(LLM_DIR / "pred_gt.npz")
    pred_raw = data["pred"]  # (n_q, n_models, n_labels)
    gt_raw = data["gt"]      # (n_q, n_labels)
    model_names = [str(x) for x in data["model_names"]]
    n_q, n_models, n_labels = pred_raw.shape
    gt_argmax = gt_raw.argmax(axis=1)  # 各問題の正解インデックス

    with open(LLM_DIR / "questions_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    categories = meta["categories"]

    print("=" * 70)
    print("LLM アンサンブル CAGL 評価（外部ベンチマーク、費用ゼロ）")
    print("=" * 70)
    print()
    print(f"データソース: {manifest['データソース']}")
    print(f"問題数: {n_q}")
    print(f"モデル数: {n_models}")
    print(f"選択肢数（最大）: {n_labels}")

    # 単独モデル正解率（各モデルの argmax が正解と一致する比率）
    print()
    print("--- 単独モデル正解率 ---")
    single_accs = {}
    for m_i, mname in enumerate(model_names):
        # 各問題で当該モデルの OVR 予測から argmax を取る
        model_argmax = pred_raw[:, m_i, :].argmax(axis=1)
        # 空予測（全0）の場合 argmax=0 になるので正解が 0 でない限り不正解
        has_pred = pred_raw[:, m_i, :].any(axis=1)
        correct = (model_argmax == gt_argmax) & has_pred
        acc = float(correct.mean())
        single_accs[mname] = acc
        print(f"  {mname:<25}: {acc*100:5.2f}%")
    best_model = max(single_accs, key=single_accs.get)
    print(f"  (最良単独: {best_model} = {single_accs[best_model]*100:.2f}%)")

    # 4変種の結果
    print()
    print("--- アンサンブル手法比較 ---")
    print(f"{'手法':<18} {'正解率':>7} {'d値':>8} {'p値':>9} "
          f"{'F1差分':>10} {'CI下限':>10} {'CI上限':>10}")

    # 単純多数決 = fixed baseline の argmax（全変種で同じなので V1 から取得）
    any_trial_path = LLM_PRED_DIR / "V1_weight_only" / "trial_000.npz"
    any_data = np.load(any_trial_path)
    any_order = any_data["order"]
    fixed_argmax = any_data["preds_fixed_argmax"]
    # order 順序を元に戻す
    original_fixed_argmax = np.zeros(n_q, dtype=np.int32)
    original_fixed_argmax[any_order] = fixed_argmax
    # 注: weight_only モードでは effective=w=0.5 uniform で argmax は
    # 複数モデル同票の場合に低インデックスを選ぶ → 単純多数決と等価
    majority_acc = float((original_fixed_argmax == gt_argmax).mean())
    print(f"{'単純多数決':<18} {majority_acc*100:>6.2f}%  "
          f"{'-':>7}  {'-':>8}  {'-':>9}  {'-':>9}  {'-':>9}")

    variants_out = {}
    for variant_name in VARIANT_NAMES:
        f1s_fixed, f1s_adapt = [], []
        acc_adapt_all = []
        for trial_path in sorted((LLM_PRED_DIR / variant_name).glob("trial_*.npz")):
            pd = np.load(trial_path)
            warmup = int(pd["warmup"])
            order = pd["order"]
            pf_bin = pd["preds_fixed"]
            pa_bin = pd["preds_adapt"]
            pa_arg = pd["preds_adapt_argmax"]

            # 並び替え順序に合わせた gt_eval
            gt_sorted = gt_raw[order]
            gt_eval = gt_sorted[warmup:]

            f1s_fixed.append(macro_f1(gt_eval, pf_bin[warmup:]))
            f1s_adapt.append(macro_f1(gt_eval, pa_bin[warmup:]))

            # 正解率（adaptive）: 準備期間後のみで argmax 一致を数える
            gt_argmax_sorted = gt_argmax[order][warmup:]
            acc_adapt_all.append(float((pa_arg[warmup:] == gt_argmax_sorted).mean()))

        d = cohens_d(f1s_adapt, f1s_fixed)
        ci_lo, ci_hi = paired_bootstrap_ci(f1s_adapt, f1s_fixed)
        diff = np.asarray(f1s_adapt) - np.asarray(f1s_fixed)
        p_val = 1.0 if diff.std() < 1e-12 else float(ttest_rel(f1s_adapt, f1s_fixed)[1])
        mean_acc = float(np.mean(acc_adapt_all))

        variants_out[variant_name] = {
            "d": d,
            "p": p_val,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
            "mean_fixed_f1": float(np.mean(f1s_fixed)),
            "mean_adapt_f1": float(np.mean(f1s_adapt)),
            "delta_f1": float(np.mean(f1s_adapt) - np.mean(f1s_fixed)),
            "mean_accuracy": mean_acc,
        }
        print(f"{variant_name:<18} {mean_acc*100:>6.2f}%  "
              f"{d:>+8.3f} {p_val:>9.4f} "
              f"{variants_out[variant_name]['delta_f1']:>+10.4f} "
              f"{ci_lo:>+10.4f} {ci_hi:>+10.4f}")

    # 判定
    v1 = variants_out["V1_weight_only"]
    v3 = variants_out["V3_both_gt"]
    v4 = variants_out["V4_cagl"]
    max_delta = max(v["delta_f1"] for v in variants_out.values())
    v4_best = v4["delta_f1"] >= max_delta - 1e-12
    v3_collapse = v3["delta_f1"] <= v1["delta_f1"] + 1e-12

    print()
    print("--- 判定 ---")
    print(f"  V4 が F1差分最大: {'YES' if v4_best else 'NO'}")
    print(f"  V3 崩壊 (V3<=V1):  {'YES' if v3_collapse else 'NO'}")

    # Weight 差別化診断
    print()
    print("--- Weight 差別化診断（V4_cagl 学習済み）---")
    all_w, all_g = [], []
    for trial_path in sorted((LLM_PRED_DIR / "V4_cagl").glob("trial_*.npz")):
        pd = np.load(trial_path)
        all_w.append(pd["final_weights"])
        all_g.append(pd["final_gates"])
    all_w = np.stack(all_w, axis=0)
    all_g = np.stack(all_g, axis=0)
    mean_w = all_w.mean(axis=0)
    mean_g = all_g.mean(axis=0)
    std_w = mean_w.std(axis=0)
    std_g = mean_g.std(axis=0)
    n_below = int((std_w < STD_THRESHOLD).sum())
    pct_below = 100.0 * n_below / n_labels

    print(f"  Weight 平均 (モデル x ラベル):")
    for m_i, mname in enumerate(model_names):
        print(f"    {mname:<25}: {format_row(mean_w[m_i])}")
    print(f"  Gate 平均 (モデル x ラベル):")
    for m_i, mname in enumerate(model_names):
        print(f"    {mname:<25}: {format_row(mean_g[m_i])}")
    print(f"  Weight のモデル間標準偏差（ラベルごと）: {format_row(std_w)}")
    print(f"  Gate   のモデル間標準偏差（ラベルごと）: {format_row(std_g)}")
    print(f"  Weight std < {STD_THRESHOLD} のラベル割合: "
          f"{n_below}/{n_labels} ({pct_below:.1f}%)")

    # カテゴリ別分析（単独最良モデルと V4 の正解率を比較）
    print()
    print("--- カテゴリ別正解率（最良単独モデル vs V4_cagl）---")
    # V4 の1試行目の予測で集計（代表値）
    v4_t0 = np.load(LLM_PRED_DIR / "V4_cagl" / "trial_000.npz")
    v4_order = v4_t0["order"]
    v4_argmax = v4_t0["preds_adapt_argmax"]
    v4_gt_arg = gt_argmax[v4_order]
    v4_cats = np.array(categories)[v4_order]

    best_m_idx = model_names.index(best_model)
    best_argmax = pred_raw[:, best_m_idx, :].argmax(axis=1)
    best_argmax_has = pred_raw[:, best_m_idx, :].any(axis=1)

    cat_list = sorted(set(categories))
    print(f"  {'カテゴリ':<18} {'問題数':>7} {'最良単独':>10} {'V4_cagl':>10}")
    cat_results = {}
    for c in cat_list:
        # 全体でのカテゴリマスク
        mask_all = np.array([cc == c for cc in categories])
        n_c = int(mask_all.sum())
        # 最良単独モデル
        best_acc_c = float(((best_argmax == gt_argmax) & best_argmax_has & mask_all).sum() / max(1, n_c))
        # V4 (1 試行目)
        mask_v4 = (v4_cats == c)
        if mask_v4.any():
            v4_acc_c = float((v4_argmax[mask_v4] == v4_gt_arg[mask_v4]).mean())
        else:
            v4_acc_c = 0.0
        print(f"  {c:<18} {n_c:>7}  {best_acc_c*100:>8.2f}%  {v4_acc_c*100:>8.2f}%")
        cat_results[c] = {
            "n": n_c,
            "best_single_acc": best_acc_c,
            "v4_cagl_acc": v4_acc_c,
        }

    # 保存
    summary = {
        "評価日時": datetime.now(timezone.utc).isoformat(),
        "データソース": manifest["データソース"],
        "問題数": n_q,
        "モデル数": n_models,
        "単独モデル正解率": single_accs,
        "最良単独モデル": best_model,
        "最良単独モデル正解率": single_accs[best_model],
        "単純多数決正解率": majority_acc,
        "変種": variants_out,
        "判定": {
            "v4_best_delta_f1": bool(v4_best),
            "v3_collapse": bool(v3_collapse),
        },
        "Weight診断": {
            "mean_weights": mean_w.tolist(),
            "mean_gates": mean_g.tolist(),
            "weight_std_per_label": std_w.tolist(),
            "gate_std_per_label": std_g.tolist(),
            "n_labels_below_std_threshold": n_below,
            "std_threshold": STD_THRESHOLD,
        },
        "カテゴリ別": cat_results,
    }
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = EVAL_DIR / "llm_benchmark_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print()
    print(f"サマリー出力: {out}")


if __name__ == "__main__":
    main()
