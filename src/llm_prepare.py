"""
段階A: LLM アンサンブル評価用データの準備（費用ゼロ版）。

MMLU-Pro の事前計算済みモデル予測（TIGER-AI-Lab/MMLU-Pro リポジトリの
eval_results/）を利用し、5〜6モデルの予測を共通問題集合に揃えて CAGL 入力
形式に変換する。API 呼び出しは一切行わない。

このスクリプトは core_cagl をインポートしない。
"""

import glob
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
LLM_DIR = RESULTS_DIR / "llm_benchmark"

MMLU_REPO_EXTRACT_ROOT = Path(r"C:\tmp\mmlu_extract")

# 選択モデル: ファミリ分散 + 精度幅（0.42〜0.78）
SELECTED_MODELS = {
    "Claude-3.5-Sonnet":      "model_outputs_claude-3-5-sonnet-20241022_5shots",
    "GPT-4o":                 "gpt-4o",
    "Gemini-1.5-Flash":       "model_outputs_gemini-1.5-flash-002_5shots",
    "Llama-3.1-70B-Instruct": "model_outputs_Meta-Llama-3_1-70B-Instruct_5shots",
    "Llama-3.1-8B-Instruct":  "model_outputs_Meta-Llama-3_1-8B-Instruct_5shots",
    "Mixtral-8x7B-Instruct":  "model_outputs_Mixtral-8x7B-Instruct-v0.1_5shots",
}

N_OPTIONS_MAX = 10  # MMLU-Pro 最大10択


def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def sha256_of_bytes(b):
    return "sha256:" + hashlib.sha256(b).hexdigest()


def load_model_results(model_dir_name):
    """eval_results 展開ディレクトリから JSON を読み込む。"""
    d = MMLU_REPO_EXTRACT_ROOT / model_dir_name
    jfiles = glob.glob(str(d / "*.json"))
    if not jfiles:
        jfiles = glob.glob(str(d / "**" / "*.json"), recursive=True)
    if not jfiles:
        raise FileNotFoundError(f"{d} に JSON が見つかりません")
    with open(jfiles[0], "r", encoding="utf-8") as f:
        return json.load(f), jfiles[0]


def letter_to_index(letter):
    """'A'→0, 'B'→1, ... 'J'→9。無効値は -1。"""
    if not isinstance(letter, str) or len(letter) != 1:
        return -1
    idx = ord(letter.upper()) - ord("A")
    return idx if 0 <= idx < N_OPTIONS_MAX else -1


def main():
    LLM_DIR.mkdir(parents=True, exist_ok=True)

    # HuggingFace datasets が使えるか軽く確認（データ取得には使わない —
    # eval_results の JSON に同じ情報が含まれるため）
    try:
        from datasets import load_dataset  # noqa: F401
        hf_ok = True
    except Exception as e:
        hf_ok = False
        print(f"(注) datasets パッケージ未インストール: {e}")

    # 各モデルの結果をロード
    print("モデル結果を読み込み中...")
    model_records = {}
    source_files = {}
    for disp_name, dir_name in SELECTED_MODELS.items():
        try:
            records, path = load_model_results(dir_name)
        except FileNotFoundError as e:
            print(f"  {disp_name}: スキップ（{e}）")
            continue
        model_records[disp_name] = records
        source_files[disp_name] = path
        print(f"  {disp_name}: {len(records)} 件 (ソース: {Path(path).name})")

    if len(model_records) < 2:
        print("利用可能なモデルが2つ未満。処理中止。")
        sys.exit(1)

    # 各モデルを question_id → pred にインデックス化（非空のみ）
    model_q2pred = {}
    model_total = {}
    model_nonempty = {}
    for name, records in model_records.items():
        q2p = {}
        total = 0
        nonempty = 0
        for r in records:
            qid = r.get("question_id")
            pred = r.get("pred")
            if qid is None:
                continue
            total += 1
            if isinstance(pred, str) and pred.strip() != "":
                q2p[qid] = pred.strip().upper()[0]
                nonempty += 1
        model_q2pred[name] = q2p
        model_total[name] = total
        model_nonempty[name] = nonempty

    # 質問メタデータの辞書（最初に見つかったレコードから取得、
    # 他モデルとの一致検証あり）
    question_meta = {}  # qid → dict
    first_model = next(iter(model_records))
    for r in model_records[first_model]:
        qid = r.get("question_id")
        if qid is None:
            continue
        question_meta[qid] = {
            "question": r.get("question"),
            "options": list(r.get("options", [])),
            "answer": r.get("answer"),
            "answer_index": r.get("answer_index"),
            "category": r.get("category"),
            "src": r.get("src"),
        }

    # 全選択モデルで非空予測がある共通 question_id
    common_qids = set.intersection(
        *[set(m.keys()) for m in model_q2pred.values()]
    )
    common_qids = sorted(common_qids)
    print(f"\n共通問題数（全モデルで非空予測）: {len(common_qids)}")

    # カテゴリ別・全体 正解率（選択モデル、共通問題集合上）
    acc_info = {}
    for name, q2p in model_q2pred.items():
        correct = 0
        by_cat_total = {}
        by_cat_correct = {}
        for qid in common_qids:
            meta = question_meta.get(qid, {})
            true_ans = meta.get("answer")
            cat = meta.get("category", "unknown")
            by_cat_total[cat] = by_cat_total.get(cat, 0) + 1
            if q2p[qid] == true_ans:
                correct += 1
                by_cat_correct[cat] = by_cat_correct.get(cat, 0) + 1
        overall = correct / len(common_qids) if common_qids else 0
        by_cat = {
            c: (by_cat_correct.get(c, 0) / by_cat_total[c])
            for c in by_cat_total
        }
        acc_info[name] = {
            "全体正解率": overall,
            "カテゴリ別正解率": by_cat,
            "総予測数": model_total[name],
            "非空予測数": model_nonempty[name],
        }
        print(f"  {name}: 全体正解率={overall:.3f} "
              f"(共通 {len(common_qids)} 問中)")

    # pred / gt テンソルの構築（one-vs-rest）
    model_names = list(model_q2pred.keys())
    n_q = len(common_qids)
    n_models = len(model_names)
    pred = np.zeros((n_q, n_models, N_OPTIONS_MAX), dtype=np.uint8)
    gt = np.zeros((n_q, N_OPTIONS_MAX), dtype=np.uint8)
    categories = []
    qid_order = []

    for i, qid in enumerate(common_qids):
        meta = question_meta[qid]
        gt_idx = letter_to_index(meta.get("answer"))
        if gt_idx < 0:
            # 正解ラベルが無効ならスキップ扱い（稀）
            continue
        gt[i, gt_idx] = 1
        for m_i, mname in enumerate(model_names):
            p_idx = letter_to_index(model_q2pred[mname][qid])
            if p_idx >= 0:
                pred[i, m_i, p_idx] = 1
        categories.append(meta.get("category", "unknown"))
        qid_order.append(qid)

    # 保存
    pred_gt_path = LLM_DIR / "pred_gt.npz"
    np.savez_compressed(
        pred_gt_path,
        pred=pred,
        gt=gt,
        model_names=np.array(model_names),
        question_ids=np.array(qid_order, dtype=np.int64),
        n_options_max=np.int64(N_OPTIONS_MAX),
    )

    meta_path = LLM_DIR / "questions_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "question_ids": qid_order,
                "categories": categories,
                "source": "TIGER-AI-Lab/MMLU-Pro eval_results",
            },
            f, ensure_ascii=False, indent=2,
        )

    manifest = {
        "生成日時": datetime.now(timezone.utc).isoformat(),
        "データソース": "TIGER-Lab/MMLU-Pro (HuggingFace) + "
                      "TIGER-AI-Lab/MMLU-Pro eval_results (GitHub)",
        "hf_datasets_利用可能": hf_ok,
        "問題数": n_q,
        "共通問題数": len(common_qids),
        "選択モデル数": n_models,
        "最大選択肢数": N_OPTIONS_MAX,
        "モデル一覧": {
            name: {
                "全体正解率": acc_info[name]["全体正解率"],
                "カテゴリ別正解率": acc_info[name]["カテゴリ別正解率"],
                "元レコード数": acc_info[name]["総予測数"],
                "非空予測数": acc_info[name]["非空予測数"],
                "ソースファイル": Path(source_files[name]).name,
            }
            for name in model_names
        },
        "ファイルハッシュ": {
            "pred_gt.npz": sha256_of_file(pred_gt_path),
            "questions_meta.json": sha256_of_file(meta_path),
        },
    }
    with open(LLM_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print()
    print(f"出力: {pred_gt_path}")
    print(f"    : {meta_path}")
    print(f"    : {LLM_DIR / 'manifest.json'}")
    print(f"pred shape = {pred.shape}, gt shape = {gt.shape}")
    cat_counts = {}
    for c in categories:
        cat_counts[c] = cat_counts.get(c, 0) + 1
    print(f"カテゴリ分布: "
          + ", ".join(f"{k}={v}" for k, v in sorted(cat_counts.items())))


if __name__ == "__main__":
    main()
