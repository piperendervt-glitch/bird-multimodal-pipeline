"""
段階B: ローカル LLM アンサンブル用に CAGL を適用。

results/local_llm/pred_gt.npz を読み込み、SHA256 で照合、4変種 × 20試行を
オンライン実行する。V4_cagl は学習済み Weight / Gate も保存する。

このスクリプトは sklearn をインポートしない。
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
LOCAL_DIR = RESULTS_DIR / "local_llm"
LOCAL_PRED_DIR = RESULTS_DIR / "local_llm_predictions"

N_TRIALS = 20
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


def compute_warmup(n_samples):
    return max(50, int(n_samples * 0.10))


def compute_scores(w, g, pred_row, weight_mode):
    if weight_mode == "weight_only":
        eff = w
    elif weight_mode == "gate_only":
        eff = 0.5 * g
    else:
        eff = w * g
    totals = eff.sum(axis=0)
    num = (eff * pred_row).sum(axis=0)
    safe_totals = np.where(totals > 1e-9, totals, 1.0)
    return np.where(totals > 1e-9, num / safe_totals, 0.0)


def run_variant(pred, gt, gate_mode, weight_mode, order):
    n_q, n_models, n_labels = pred.shape

    model_fixed = CAGL(n_models, n_labels, gate_mode="none")
    pf_bin = np.zeros((n_q, n_labels), dtype=np.uint8)
    pf_arg = np.zeros(n_q, dtype=np.int32)
    for step, i in enumerate(order):
        pf_bin[step] = model_fixed.predict(pred[i], weight_mode=weight_mode)
        s = compute_scores(model_fixed.w, model_fixed.g, pred[i], weight_mode)
        pf_arg[step] = int(np.argmax(s))

    model = CAGL(n_models, n_labels, gate_mode=gate_mode)
    pa_bin = np.zeros((n_q, n_labels), dtype=np.uint8)
    pa_arg = np.zeros(n_q, dtype=np.int32)
    for step, i in enumerate(order):
        final = model.predict(pred[i], weight_mode=weight_mode)
        pa_bin[step] = final
        s = compute_scores(model.w, model.g, pred[i], weight_mode)
        pa_arg[step] = int(np.argmax(s))
        model.update(pred[i], gt[i], final)

    return pf_bin, pa_bin, pf_arg, pa_arg, model.w.copy(), model.g.copy()


def main():
    manifest_path = LOCAL_DIR / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    expected = manifest["ファイルハッシュ"]["pred_gt.npz"]
    actual = sha256_of_file(LOCAL_DIR / "pred_gt.npz")
    if expected != actual:
        raise RuntimeError(f"pred_gt.npz ハッシュ不一致: {expected} vs {actual}")
    print("SHA256 検証 OK: pred_gt.npz")

    data = np.load(LOCAL_DIR / "pred_gt.npz")
    pred = data["pred"].astype(np.uint8)
    gt = data["gt"].astype(np.uint8)
    n_q = pred.shape[0]
    warmup = compute_warmup(n_q)
    print(f"問題数={n_q}, モデル数={pred.shape[1]}, "
          f"選択肢数={pred.shape[2]}, 準備期間={warmup}")

    for v in VARIANTS:
        (LOCAL_PRED_DIR / v).mkdir(parents=True, exist_ok=True)

    for variant_name, (gate_mode, weight_mode) in VARIANTS.items():
        print(f"  {variant_name} ...", flush=True)
        for trial in range(N_TRIALS):
            seed = trial * 1000 + 42
            rng = np.random.default_rng(seed)
            order = rng.permutation(n_q)

            pf_bin, pa_bin, pf_arg, pa_arg, final_w, final_g = run_variant(
                pred, gt, gate_mode, weight_mode, order
            )

            save_kwargs = dict(
                preds_fixed=pf_bin,
                preds_adapt=pa_bin,
                preds_fixed_argmax=pf_arg,
                preds_adapt_argmax=pa_arg,
                order=order.astype(np.int64),
                warmup=np.int64(warmup),
                seed=np.int64(seed),
            )
            if variant_name == "V4_cagl":
                save_kwargs["final_weights"] = final_w
                save_kwargs["final_gates"] = final_g

            np.savez_compressed(
                LOCAL_PRED_DIR / variant_name / f"trial_{trial:03d}.npz",
                **save_kwargs,
            )
        print(f"    {N_TRIALS} 試行出力完了")

    print()
    print("段階B 完了")


if __name__ == "__main__":
    main()
