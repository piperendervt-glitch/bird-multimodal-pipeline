"""
段階B: 固定されたベンチマーク予測に対して CAGL を適用。

results/benchmark_data/ の .npz を sha256 で検証した上で読み込み、
各データセット × 各試行 × 4変種についてオンライン学習を実行。
予測を results/benchmark_predictions/ に保存する。

このスクリプトは sklearn をインポートしない（評価しない）。
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from core_cagl import CAGL


try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
BENCH_DATA_DIR = RESULTS_DIR / "benchmark_data"
BENCH_PRED_DIR = RESULTS_DIR / "benchmark_predictions"

VARIANTS = {
    "V1_weight_only": ("none", "weight_only"),
    "V2_gate_only":   ("consensus", "gate_only"),
    "V3_both_gt":     ("gt", "multiplicative"),
    "V4_cagl":        ("consensus", "multiplicative"),
}


def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def compute_warmup(n_test_samples):
    """準備期間 = テストサンプル数の10%（最低20）。"""
    return max(20, int(n_test_samples * 0.10))


def run_variant_on_trial(pred, gt, gate_mode, weight_mode):
    """オンライン学習（predict → 記録 → update）。
    適応モデルの学習後の Weight / Gate も返す（診断用）。"""
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

    return preds_fixed, preds_adapt, model.w.copy(), model.g.copy()


def main():
    manifest_path = BENCH_DATA_DIR / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # 全データセットのファイル sha256 を検証
    print("sha256 検証中...")
    total_files = 0
    for ds_name, ds_info in manifest["datasets"].items():
        for fname, expected in ds_info["files"].items():
            actual = sha256_of_file(BENCH_DATA_DIR / ds_name / fname)
            if actual != expected:
                raise RuntimeError(
                    f"ハッシュ不一致: {ds_name}/{fname} "
                    f"expected={expected}, got={actual}"
                )
            total_files += 1
    print(f"  全 {total_files} ファイル検証 OK")

    BENCH_PRED_DIR.mkdir(parents=True, exist_ok=True)

    for ds_name, ds_info in manifest["datasets"].items():
        n_test_samples = ds_info["n_test_samples"]
        warmup = compute_warmup(n_test_samples)
        print(f"[{ds_name}] テストサンプル数={n_test_samples}, 準備期間={warmup}")

        for variant_name, (gate_mode, weight_mode) in VARIANTS.items():
            out_dir = BENCH_PRED_DIR / ds_name / variant_name
            out_dir.mkdir(parents=True, exist_ok=True)
            for fname in ds_info["files"].keys():
                data = np.load(BENCH_DATA_DIR / ds_name / fname)
                pred = data["pred"].astype(np.uint8)
                gt = data["gt"].astype(np.uint8)

                pf, pa, final_w, final_g = run_variant_on_trial(
                    pred, gt, gate_mode, weight_mode
                )

                save_kwargs = {
                    "preds_fixed": pf,
                    "preds_adapt": pa,
                    "warmup": np.int64(warmup),
                }
                if variant_name == "V4_cagl":
                    save_kwargs["final_weights"] = final_w
                    save_kwargs["final_gates"] = final_g
                np.savez_compressed(out_dir / fname, **save_kwargs)
        print(f"  4変種 × {len(ds_info['files'])} 試行 = "
              f"{4 * len(ds_info['files'])} 予測ファイル出力")

    print()
    print("段階B 完了")


if __name__ == "__main__":
    main()
