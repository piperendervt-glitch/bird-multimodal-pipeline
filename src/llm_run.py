"""
段階B: LLM アンサンブル評価用に CAGL を適用。

results/llm_benchmark/pred_gt.npz を読み込み（SHA256 照合）、4変種を
20試行分のシャッフル順序でオンライン実行する。V4_cagl は学習済み
Weight / Gate も保存する。加えて、each 試行の argmax 予測も保存して
argmax ベースの正解率計算を可能にする。

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
LLM_DIR = RESULTS_DIR / "llm_benchmark"
LLM_PRED_DIR = RESULTS_DIR / "llm_predictions"

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
    """CAGL 内部と同じスコア計算を外側で再現。argmax 用。"""
    if weight_mode == "weight_only":
        eff = w
    elif weight_mode == "gate_only":
        eff = 0.5 * g
    else:
        eff = w * g
    totals = eff.sum(axis=0)
    num = (eff * pred_row).sum(axis=0)
    safe_totals = np.where(totals > 1e-9, totals, 1.0)
    scores = np.where(totals > 1e-9, num / safe_totals, 0.0)
    return scores


def run_variant(pred, gt, gate_mode, weight_mode, order):
    """
    order: 問題インデックスの並べ替え配列 (shape=(n,))
    戻り値: 並べ替えた順序での binary 予測 / argmax 予測 / 最終 w,g。
    """
    n_q, n_models, n_labels = pred.shape

    # Fixed baseline (初期 w=g=0.5、学習なし)
    model_fixed = CAGL(n_models, n_labels, gate_mode="none")
    preds_fixed_bin = np.zeros((n_q, n_labels), dtype=np.uint8)
    preds_fixed_arg = np.zeros(n_q, dtype=np.int32)
    for step, i in enumerate(order):
        p = model_fixed.predict(pred[i], weight_mode=weight_mode)
        preds_fixed_bin[step] = p
        s = compute_scores(model_fixed.w, model_fixed.g,
                           pred[i], weight_mode)
        preds_fixed_arg[step] = int(np.argmax(s))

    # Adaptive
    model = CAGL(n_models, n_labels, gate_mode=gate_mode)
    preds_adapt_bin = np.zeros((n_q, n_labels), dtype=np.uint8)
    preds_adapt_arg = np.zeros(n_q, dtype=np.int32)
    for step, i in enumerate(order):
        final = model.predict(pred[i], weight_mode=weight_mode)
        preds_adapt_bin[step] = final
        s = compute_scores(model.w, model.g, pred[i], weight_mode)
        preds_adapt_arg[step] = int(np.argmax(s))
        model.update(pred[i], gt[i], final)

    return (preds_fixed_bin, preds_adapt_bin,
            preds_fixed_arg, preds_adapt_arg,
            model.w.copy(), model.g.copy())


def main():
    manifest_path = LLM_DIR / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    expected_hash = manifest["ファイルハッシュ"]["pred_gt.npz"]
    actual_hash = sha256_of_file(LLM_DIR / "pred_gt.npz")
    if expected_hash != actual_hash:
        raise RuntimeError(
            f"pred_gt.npz のハッシュ不一致: "
            f"expected={expected_hash}, actual={actual_hash}"
        )
    print(f"SHA256 検証 OK: pred_gt.npz")

    data = np.load(LLM_DIR / "pred_gt.npz")
    pred = data["pred"].astype(np.uint8)
    gt = data["gt"].astype(np.uint8)
    model_names = list(data["model_names"])
    n_q = pred.shape[0]
    warmup = compute_warmup(n_q)
    print(f"問題数={n_q}, モデル数={pred.shape[1]}, "
          f"選択肢数={pred.shape[2]}, 準備期間={warmup}")

    for v in VARIANTS:
        (LLM_PRED_DIR / v).mkdir(parents=True, exist_ok=True)

    for variant_name, (gate_mode, weight_mode) in VARIANTS.items():
        print(f"  {variant_name} ({gate_mode}/{weight_mode}) ...", flush=True)
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
                LLM_PRED_DIR / variant_name / f"trial_{trial:03d}.npz",
                **save_kwargs,
            )
        print(f"    {N_TRIALS} 試行の予測ファイル出力完了")

    print()
    print("段階B 完了")


if __name__ == "__main__":
    main()
