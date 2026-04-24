"""
Phase 2: Run each CAGL variant on frozen test data, save predictions.

- Reads manifest.json and verifies every .npz file's sha256 before touching it.
- Runs 4 variants in online mode (predict -> record -> update).
- Saves preds_fixed / preds_adapt (post-warmup only) per variant per trial.

This script does NOT import sklearn. It does not score predictions.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from core_cagl import CAGL


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
TEST_DATA_DIR = RESULTS_DIR / "test_data"
PRED_DIR = RESULTS_DIR / "predictions"

WARMUP = 200

VARIANTS = {
    "V1_weight_only": ("none", "weight_only"),
    "V2_gate_only": ("consensus", "gate_only"),
    "V3_both_gt": ("gt", "multiplicative"),
    "V4_cagl": ("consensus", "multiplicative"),
}


def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def run_variant_on_trial(pred, gt, gate_mode, weight_mode):
    n_samples, n_topologies, n_labels = pred.shape

    model_fixed = CAGL(n_topologies, n_labels, gate_mode="none")
    preds_fixed = np.zeros((n_samples, n_labels), dtype=np.uint8)
    for i in range(n_samples):
        preds_fixed[i] = model_fixed.predict(pred[i], weight_mode=weight_mode)

    model = CAGL(n_topologies, n_labels, gate_mode=gate_mode)
    preds_adapt = np.zeros((n_samples, n_labels), dtype=np.uint8)
    for i in range(n_samples):
        final = model.predict(pred[i], weight_mode=weight_mode)
        preds_adapt[i] = final
        model.update(pred[i], gt[i], final)

    return preds_fixed[WARMUP:], preds_adapt[WARMUP:]


def main():
    manifest_path = TEST_DATA_DIR / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    print(f"Manifest: {manifest['n_trials']} trials, "
          f"{manifest['n_samples']} samples, "
          f"{manifest['n_topologies']} topologies")

    # Verify every data file's sha256
    print("Verifying sha256 hashes...")
    for fname, expected in manifest["files"].items():
        actual = sha256_of_file(TEST_DATA_DIR / fname)
        if actual != expected:
            raise RuntimeError(
                f"Hash mismatch for {fname}: expected {expected}, got {actual}"
            )
    print(f"  All {len(manifest['files'])} files verified.")

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        (PRED_DIR / variant).mkdir(parents=True, exist_ok=True)

    n_trials = manifest["n_trials"]
    for variant_name, (gate_mode, weight_mode) in VARIANTS.items():
        print(f"Running {variant_name} ({gate_mode}/{weight_mode}) ...", flush=True)
        for trial in range(n_trials):
            fname = f"trial_{trial:03d}.npz"
            data = np.load(TEST_DATA_DIR / fname)
            pred = data["pred"].astype(np.uint8)
            gt = data["gt"].astype(np.uint8)

            pf, pa = run_variant_on_trial(pred, gt, gate_mode, weight_mode)

            out = PRED_DIR / variant_name / fname
            np.savez_compressed(
                out,
                preds_fixed=pf,
                preds_adapt=pa,
                warmup=np.int64(WARMUP),
            )
        print(f"  {variant_name}: {n_trials} prediction files written")

    print("Phase 2 complete.")


if __name__ == "__main__":
    main()
