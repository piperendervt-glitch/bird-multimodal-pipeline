"""
Phase 1: Generate and freeze test data.

This script does NOT import core_cagl. Its only job is to produce fully
deterministic synthetic datasets, save them as .npz, and write a manifest
with sha256 hashes so downstream phases can verify the data was not modified.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from data_gen import generate_synthetic_data, DEFAULT_TOPO_ACCURACY


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
TEST_DATA_DIR = RESULTS_DIR / "test_data"

N_TRIALS = 50
N_SAMPLES = 5000
N_TOPOLOGIES = 5
N_LABELS = 2
SHARED_NOISE_RATE = 0.05


def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def main():
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

    files = {}
    print(f"Generating {N_TRIALS} trials -> {TEST_DATA_DIR}")
    for trial in range(N_TRIALS):
        seed = trial * 1000 + 42
        pred, gt = generate_synthetic_data(
            n_samples=N_SAMPLES,
            n_topologies=N_TOPOLOGIES,
            n_labels=N_LABELS,
            topo_accuracy=DEFAULT_TOPO_ACCURACY,
            shared_noise_rate=SHARED_NOISE_RATE,
            seed=seed,
        )
        fname = f"trial_{trial:03d}.npz"
        fpath = TEST_DATA_DIR / fname
        np.savez_compressed(
            fpath,
            pred=pred.astype(np.uint8),
            gt=gt.astype(np.uint8),
            seed=np.int64(seed),
        )
        files[fname] = sha256_of_file(fpath)
        if (trial + 1) % 10 == 0:
            print(f"  {trial + 1}/{N_TRIALS} done")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_trials": N_TRIALS,
        "n_samples": N_SAMPLES,
        "n_topologies": N_TOPOLOGIES,
        "n_labels": N_LABELS,
        "shared_noise_rate": SHARED_NOISE_RATE,
        "topo_accuracy": DEFAULT_TOPO_ACCURACY,
        "files": files,
    }
    manifest_path = TEST_DATA_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written: {manifest_path}")
    print(f"Total trial files: {len(files)}")


if __name__ == "__main__":
    main()
