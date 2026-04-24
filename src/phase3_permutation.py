"""
Phase 3b: Permutation test (null distribution via GT label shuffling).

For V4_cagl only. For each of 1000 permutations, randomly shuffle the
ground-truth rows and feed the shuffled GT to CAGL.update(...). The
resulting adaptive predictions are evaluated against the REAL (unshuffled)
GT. This builds a null distribution for delta_F1 under "GT signal is
meaningless". The actual V4 delta_F1 (unshuffled) is compared against
that null to yield a permutation p-value.

Uses 5 trials for speed.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from core_cagl import CAGL


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
TEST_DATA_DIR = RESULTS_DIR / "test_data"
EVAL_DIR = RESULTS_DIR / "evaluation"

WARMUP = 200
N_PERMUTATIONS = 1000
N_TRIALS_FOR_PERM = 5


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


def run_v4_fixed(pred):
    """Fixed baseline preds (w=g=0.5, no learning). Deterministic per pred."""
    n_samples, n_top, n_lab = pred.shape
    model = CAGL(n_top, n_lab, gate_mode="none")
    out = np.zeros((n_samples, n_lab), dtype=np.uint8)
    for i in range(n_samples):
        out[i] = model.predict(pred[i], weight_mode="multiplicative")
    return out


def run_v4_adapt(pred, gt_for_update):
    """V4 online adaptive using the given gt for update signal."""
    n_samples, n_top, n_lab = pred.shape
    model = CAGL(n_top, n_lab, gate_mode="consensus")
    out = np.zeros((n_samples, n_lab), dtype=np.uint8)
    for i in range(n_samples):
        final = model.predict(pred[i], weight_mode="multiplicative")
        out[i] = final
        model.update(pred[i], gt_for_update[i], final)
    return out


def delta_for_trial(preds_fixed, preds_adapt, gt_real):
    gt_eval = gt_real[WARMUP:]
    f1_f = macro_f1(gt_eval, preds_fixed[WARMUP:])
    f1_a = macro_f1(gt_eval, preds_adapt[WARMUP:])
    return f1_a - f1_f


def main():
    manifest_path = TEST_DATA_DIR / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest_hash = sha256_of_file(manifest_path)

    # Load the 5 trials used for permutation
    trials = []
    for trial in range(N_TRIALS_FOR_PERM):
        fname = f"trial_{trial:03d}.npz"
        data = np.load(TEST_DATA_DIR / fname)
        trials.append((data["pred"].astype(np.uint8),
                       data["gt"].astype(np.uint8)))

    # Cache fixed baseline (invariant across permutations)
    fixed_cache = [run_v4_fixed(pred) for pred, _ in trials]

    # Actual delta (unshuffled V4 on the same 5 trials — apples-to-apples)
    print("Computing actual V4 delta on 5 trials...")
    actual_deltas = []
    for (pred, gt), pf in zip(trials, fixed_cache):
        pa = run_v4_adapt(pred, gt)
        actual_deltas.append(delta_for_trial(pf, pa, gt))
    actual_mean = float(np.mean(actual_deltas))
    print(f"  actual mean delta_F1 = {actual_mean:+.4f}")

    # Null distribution
    print(f"Running {N_PERMUTATIONS} permutations x {N_TRIALS_FOR_PERM} trials ...")
    null_means = np.zeros(N_PERMUTATIONS)
    for perm in range(N_PERMUTATIONS):
        rng_perm = np.random.default_rng(perm * 7 + 13)
        perm_deltas = []
        for (pred, gt), pf in zip(trials, fixed_cache):
            shuffled_gt = gt.copy()
            rng_perm.shuffle(shuffled_gt)  # shuffles rows (axis 0)
            pa = run_v4_adapt(pred, shuffled_gt)
            perm_deltas.append(delta_for_trial(pf, pa, gt))
        null_means[perm] = float(np.mean(perm_deltas))
        if (perm + 1) % 100 == 0:
            print(f"  {perm + 1}/{N_PERMUTATIONS} done "
                  f"(running null mean {null_means[:perm+1].mean():+.4f})",
                  flush=True)

    p_value = float((null_means >= actual_mean).mean())

    result = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "data_manifest_hash": manifest_hash,
        "n_permutations": N_PERMUTATIONS,
        "n_trials": N_TRIALS_FOR_PERM,
        "warmup": WARMUP,
        "actual_mean_delta_f1": actual_mean,
        "actual_per_trial": actual_deltas,
        "null_mean": float(null_means.mean()),
        "null_std": float(null_means.std()),
        "null_percentile_95": float(np.percentile(null_means, 95)),
        "null_percentile_99": float(np.percentile(null_means, 99)),
        "null_max": float(null_means.max()),
        "null_min": float(null_means.min()),
        "permutation_p_value": p_value,
    }

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_DIR / "permutation_test.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print()
    print("=" * 70)
    print("Permutation Test Result")
    print("=" * 70)
    print(f"Actual V4 mean delta_F1 (5 trials):   {actual_mean:+.4f}")
    print(f"Null distribution mean:                {null_means.mean():+.4f}")
    print(f"Null distribution std:                 {null_means.std():.4f}")
    print(f"Null 95th percentile:                  {np.percentile(null_means, 95):+.4f}")
    print(f"Null 99th percentile:                  {np.percentile(null_means, 99):+.4f}")
    print(f"Null max:                              {null_means.max():+.4f}")
    print(f"Permutation p-value:                   {p_value:.4f}")
    print(f"(fraction of null means >= actual; 1000 permutations)")
    print()
    print(f"Result written: {out_path}")


if __name__ == "__main__":
    main()
