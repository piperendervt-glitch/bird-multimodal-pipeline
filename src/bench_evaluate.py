"""
段階C: ベンチマーク予測の評価。

results/benchmark_data/ と results/benchmark_predictions/ を読み込み、
各データセット × 各変種について F1 / コーエンのd / ブートストラップCI /
対応のあるt検定 を計算する。出力はデータセット別の詳細表と
benchmark_summary.json。

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
BENCH_DATA_DIR = RESULTS_DIR / "benchmark_data"
BENCH_PRED_DIR = RESULTS_DIR / "benchmark_predictions"
EVAL_DIR = RESULTS_DIR / "evaluation"

VARIANT_NAMES = ["V1_weight_only", "V2_gate_only", "V3_both_gt", "V4_cagl"]


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
    boot_means = np.array([
        rng.choice(diff, size=n, replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lo, hi


def evaluate_dataset(ds_name, ds_info):
    """1データセットの4変種を評価して辞書で返す。"""
    result = {}
    for variant in VARIANT_NAMES:
        f1s_fixed, f1s_adapt = [], []
        for fname in ds_info["files"].keys():
            data = np.load(BENCH_DATA_DIR / ds_name / fname)
            gt = data["gt"]

            pd = np.load(BENCH_PRED_DIR / ds_name / variant / fname)
            warmup = int(pd["warmup"])
            pf = pd["preds_fixed"]
            pa = pd["preds_adapt"]
            gt_eval = gt[warmup:]

            f1s_fixed.append(macro_f1(gt_eval, pf[warmup:]))
            f1s_adapt.append(macro_f1(gt_eval, pa[warmup:]))

        d = cohens_d(f1s_adapt, f1s_fixed)
        ci_lo, ci_hi = paired_bootstrap_ci(f1s_adapt, f1s_fixed)
        # ttest_rel は差分が全て 0 だと nan を返すので保護
        diff = np.asarray(f1s_adapt) - np.asarray(f1s_fixed)
        if diff.std() < 1e-12:
            p_val = 1.0
        else:
            _, p_val = ttest_rel(f1s_adapt, f1s_fixed)
        result[variant] = {
            "d": d,
            "p": float(p_val),
            "ci_low": ci_lo,
            "ci_high": ci_hi,
            "mean_fixed": float(np.mean(f1s_fixed)),
            "mean_adapt": float(np.mean(f1s_adapt)),
            "delta_f1": float(np.mean(f1s_adapt) - np.mean(f1s_fixed)),
            "per_trial_f1_fixed": f1s_fixed,
            "per_trial_f1_adapt": f1s_adapt,
        }
    return result


def load_v4_final_params(ds_name, ds_info):
    """V4_cagl の学習済み Weight / Gate を全試行分ロード。
    戻り値: all_w, all_g 各 shape (n_trials, n_topologies, n_labels)"""
    all_w, all_g = [], []
    for fname in ds_info["files"].keys():
        pd = np.load(BENCH_PRED_DIR / ds_name / "V4_cagl" / fname)
        all_w.append(pd["final_weights"])
        all_g.append(pd["final_gates"])
    return np.stack(all_w, axis=0), np.stack(all_g, axis=0)


def format_row(values, width=6, prec=3):
    return "[" + ", ".join(f"{v:>{width}.{prec}f}" for v in values) + "]"


def main():
    manifest_path = BENCH_DATA_DIR / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest_hash = sha256_of_file(manifest_path)
    short = manifest["classifier_short"]
    classifier_names = manifest["classifiers"]

    datasets_out = {}
    v4_best_delta_count = 0
    v3_collapse_count = 0
    total_labels = 0
    low_std_labels = 0
    STD_THRESHOLD = 0.05

    print("=" * 70)
    print("外部ベンチマーク評価結果")
    print("=" * 70)

    for ds_name, ds_info in manifest["datasets"].items():
        result = evaluate_dataset(ds_name, ds_info)

        # 判定
        max_delta = max(v["delta_f1"] for v in result.values())
        v4_is_best = result["V4_cagl"]["delta_f1"] >= max_delta - 1e-12
        v3_collapse = (
            result["V3_both_gt"]["delta_f1"]
            <= result["V1_weight_only"]["delta_f1"] + 1e-12
        )
        if v4_is_best:
            v4_best_delta_count += 1
        if v3_collapse:
            v3_collapse_count += 1

        datasets_out[ds_name] = {
            "n_test_samples": ds_info["n_test_samples"],
            "mean_classifier_accuracy": ds_info["mean_classifier_accuracy"],
            "variants": result,
            "v4_is_best_delta": bool(v4_is_best),
            "v3_collapse": bool(v3_collapse),
        }

        # 表示
        print()
        print(f"--- {ds_name} ---")
        acc = ds_info["mean_classifier_accuracy"]
        acc_str = ", ".join(f"{short[k]}={v:.2f}" for k, v in acc.items())
        print(f"分類器の正解率: {acc_str}")
        print(f"テストサンプル数: {ds_info['n_test_samples']}")
        print(
            f"{'変種':<18} {'d値':>9} {'p値':>10} "
            f"{'F1差分':>10} {'CI下限':>10} {'CI上限':>10}"
        )
        for variant in VARIANT_NAMES:
            r = result[variant]
            print(
                f"{variant:<18} {r['d']:>+9.3f} {r['p']:>10.4f} "
                f"{r['delta_f1']:>+10.4f} "
                f"{r['ci_low']:>+10.4f} {r['ci_high']:>+10.4f}"
            )
        print(
            f"V4 が F1差分最大: {'YES' if v4_is_best else 'NO'}  /  "
            f"V3 崩壊 (V3<=V1): {'YES' if v3_collapse else 'NO'}"
        )

        # V4_cagl 学習済み Weight / Gate の診断
        all_w, all_g = load_v4_final_params(ds_name, ds_info)
        mean_w = all_w.mean(axis=0)  # (n_topologies, n_labels)
        mean_g = all_g.mean(axis=0)
        # 各ラベルについて分類器間の標準偏差
        std_w_per_label = mean_w.std(axis=0)
        std_g_per_label = mean_g.std(axis=0)

        n_classes = mean_w.shape[1]
        print()
        print(f"[{ds_name}] V4_cagl 学習結果診断 "
              f"({all_w.shape[0]}試行平均)")
        print("  Weight 平均（分類器 x クラス）:")
        for t, name in enumerate(classifier_names):
            print(f"    {short[name]:<4}: {format_row(mean_w[t])}")
        print("  Gate 平均（分類器 x クラス）:")
        for t, name in enumerate(classifier_names):
            print(f"    {short[name]:<4}: {format_row(mean_g[t])}")
        print(f"  Weight の分類器間標準偏差（ラベルごと）: "
              f"{format_row(std_w_per_label)}")
        print(f"  Gate   の分類器間標準偏差（ラベルごと）: "
              f"{format_row(std_g_per_label)}")
        n_low = int((std_w_per_label < STD_THRESHOLD).sum())
        print(f"  Weight std < {STD_THRESHOLD} のラベル数: "
              f"{n_low}/{n_classes}")

        # 集計用
        total_labels += n_classes
        low_std_labels += n_low

        # JSON 出力に含める
        datasets_out[ds_name]["diagnostics"] = {
            "mean_weights": mean_w.tolist(),
            "mean_gates": mean_g.tolist(),
            "weight_std_per_label": std_w_per_label.tolist(),
            "gate_std_per_label": std_g_per_label.tolist(),
            "n_labels_below_threshold": n_low,
            "std_threshold": STD_THRESHOLD,
        }

    print()
    print("--- 総合判定 ---")
    n_ds = len(datasets_out)
    print(f"V4 が F1差分最大のデータセット数: {v4_best_delta_count}/{n_ds}")
    print(f"V3 崩壊が観察されたデータセット数: {v3_collapse_count}/{n_ds}")
    pct = 100.0 * low_std_labels / total_labels if total_labels else 0.0
    print(f"Weight 分類器間 std < {STD_THRESHOLD} のラベル数（全データセット合計）: "
          f"{low_std_labels}/{total_labels} ({pct:.1f}%)")

    summary = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_manifest_hash": manifest_hash,
        "datasets": datasets_out,
        "overall": {
            "n_datasets": n_ds,
            "v4_best_delta_count": v4_best_delta_count,
            "v3_collapse_count": v3_collapse_count,
            "weight_std_threshold": STD_THRESHOLD,
            "total_labels": total_labels,
            "labels_below_weight_std_threshold": low_std_labels,
        },
    }
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_DIR / "benchmark_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print()
    print(f"総合サマリー出力: {out_path}")


if __name__ == "__main__":
    main()
