"""
Phase 3: Evaluate frozen predictions against frozen ground truth.

Reads results/test_data/ (for gt) and results/predictions/{variant}/ (for
preds_fixed / preds_adapt). Computes macro F1, paired Cohen's d, paired
bootstrap CI, paired t-test. Writes results/evaluation/summary.json.

This script does NOT import core_cagl. It treats the prediction files as
opaque outputs from some black-box system.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel
from sklearn.metrics import f1_score


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
TEST_DATA_DIR = RESULTS_DIR / "test_data"
PRED_DIR = RESULTS_DIR / "predictions"
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


def main():
    manifest_path = TEST_DATA_DIR / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest_hash = sha256_of_file(manifest_path)
    n_trials = manifest["n_trials"]

    variants_out = {}
    for variant in VARIANT_NAMES:
        f1s_fixed, f1s_adapt = [], []
        for trial in range(n_trials):
            fname = f"trial_{trial:03d}.npz"
            data = np.load(TEST_DATA_DIR / fname)
            gt = data["gt"]

            pred_data = np.load(PRED_DIR / variant / fname)
            warmup = int(pred_data["warmup"])
            preds_fixed = pred_data["preds_fixed"]
            preds_adapt = pred_data["preds_adapt"]
            gt_eval = gt[warmup:]

            f1s_fixed.append(macro_f1(gt_eval, preds_fixed))
            f1s_adapt.append(macro_f1(gt_eval, preds_adapt))

        d = cohens_d(f1s_adapt, f1s_fixed)
        ci_lo, ci_hi = paired_bootstrap_ci(f1s_adapt, f1s_fixed)
        _, p_val = ttest_rel(f1s_adapt, f1s_fixed)

        variants_out[variant] = {
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

    # Interpretation
    v1 = variants_out["V1_weight_only"]
    v3 = variants_out["V3_both_gt"]
    v4 = variants_out["V4_cagl"]

    max_delta = max(v["delta_f1"] for v in variants_out.values())
    max_d = max(v["d"] for v in variants_out.values())
    v4_is_best_delta = v4["delta_f1"] >= max_delta - 1e-9
    collapse_observed = v3["delta_f1"] <= v1["delta_f1"] + 1e-9
    v4_is_best_d = v4["d"] >= max_d - 1e-9

    if v4_is_best_delta and collapse_observed:
        scenario = "A" if v4_is_best_d else "A-"
    elif v4_is_best_delta:
        scenario = "B"
    else:
        scenario = "C"

    summary = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "data_manifest_hash": manifest_hash,
        "n_trials": n_trials,
        "variants": variants_out,
        "checks": {
            "v4_is_best_delta": bool(v4_is_best_delta),
            "collapse_observed": bool(collapse_observed),
            "v4_is_best_d": bool(v4_is_best_d),
        },
        "scenario": scenario,
    }

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = EVAL_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Print table
    print("=" * 70)
    print("Phase 3 Evaluation (external)")
    print("=" * 70)
    print(f"data_manifest_hash: {manifest_hash}")
    print(f"n_trials:           {n_trials}")
    print()
    print(f"{'Variant':<20} {'d':>8} {'p':>10} {'dF1':>10} {'CI_low':>10} {'CI_high':>10}")
    for variant in VARIANT_NAMES:
        r = variants_out[variant]
        print(
            f"{variant:<20} {r['d']:>+8.3f} {r['p']:>10.4f} "
            f"{r['delta_f1']:>+10.4f} {r['ci_low']:>+10.4f} {r['ci_high']:>+10.4f}"
        )
    print()
    print(f"V4 highest dF1:        {'YES' if v4_is_best_delta else 'NO'}")
    print(f"V3 collapse (V3<=V1):  {'YES' if collapse_observed else 'NO'}")
    print(f"V4 highest d:          {'YES' if v4_is_best_d else 'NO'}")
    print(f"Scenario:              {scenario}")
    print()
    print(f"Summary written: {summary_path}")


if __name__ == "__main__":
    main()
