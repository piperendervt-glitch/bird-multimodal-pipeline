"""
段階A: ローカル LLM (Ollama) による MMLU-Pro 推論とデータ固定。

MMLU-Pro のテストセットをカテゴリ均等サンプリングで 500 問に絞り、
Ollama 経由で5モデル分の予測を取得する。CAGL 入力形式に変換して保存。

このスクリプトは core_cagl をインポートしない。
進捗を逐次ディスクに書き出すため、中断・再開が可能。
"""

import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
LOCAL_DIR = RESULTS_DIR / "local_llm"

OLLAMA_MODELS = [
    "llama3.1:8b",
    "gemma2:9b",
    "mistral:7b",
    "qwen2.5:7b",
    "phi3:latest",  # phi3:mini は未 pull なので phi3:latest を使用（同じ mini 3.8B）
]

N_QUESTIONS = 500
N_OPTIONS_MAX = 10
OLLAMA_URL = "http://localhost:11434/api/generate"
REQUEST_TIMEOUT = 180  # 秒
PROGRESS_EVERY = 50


def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def call_ollama(model, prompt, temperature=0.0, max_tokens=256):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        result = json.loads(resp.read())
    return result.get("response", "")


def format_question(item):
    options = item["options"]
    option_text = "\n".join(
        f"{chr(65 + i)}) {opt}" for i, opt in enumerate(options)
    )
    return (
        "The following is a multiple choice question. "
        "Answer with ONLY the letter of the correct option (e.g. 'A').\n\n"
        f"Question: {item['question']}\n\n"
        f"{option_text}\n\n"
        "Answer:"
    )


def parse_answer(response, n_options):
    if not isinstance(response, str):
        return None
    valid = [chr(65 + i) for i in range(n_options)]
    s = response.strip()
    if not s:
        return None

    # パターン1: 先頭の文字が有効な選択肢
    if s[0] in valid:
        return s[0]

    # パターン2: "answer is X" 形式
    m = re.search(r"answer\s+is\s*\(?\s*([A-J])\s*\)?", s, re.IGNORECASE)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()

    # パターン3: 最初に出現する有効な大文字
    for ch in s:
        if ch in valid:
            return ch

    return None


def sample_questions():
    """MMLU-Pro をカテゴリ均等サンプリング。"""
    from datasets import load_dataset
    print("MMLU-Pro データを読み込み中...")
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

    random.seed(42)
    by_cat = {}
    for item in ds:
        by_cat.setdefault(item["category"], []).append(item)

    n_cats = len(by_cat)
    n_per = max(1, N_QUESTIONS // n_cats)
    sampled = []
    for cat in sorted(by_cat):
        items = by_cat[cat]
        k = min(n_per, len(items))
        sampled.extend(random.sample(items, k))

    # 500 に正規化
    if len(sampled) > N_QUESTIONS:
        sampled = random.sample(sampled, N_QUESTIONS)
    elif len(sampled) < N_QUESTIONS:
        # 足りない分はカテゴリ無関係に補充
        used = {q["question_id"] for q in sampled}
        rest = [x for x in ds if x["question_id"] not in used]
        sampled.extend(random.sample(rest, N_QUESTIONS - len(sampled)))

    print(f"サンプル数: {len(sampled)}, カテゴリ数: {n_cats}")
    return sampled


def ensure_model_available(model):
    """簡単な呼び出しで動作確認。"""
    try:
        call_ollama(model, "Hi.", max_tokens=8)
        return True
    except urllib.error.HTTPError as e:
        print(f"  {model}: HTTP {e.code} ({e.reason})")
    except Exception as e:
        print(f"  {model}: {e}")
    return False


def infer_one_model(model, questions, out_path):
    """1モデルで全問推論。完了済みファイルがあれば読み込んで再開。"""
    existing = []
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if len(existing) == len(questions):
            print(f"  {model}: 既に完了済 ({len(existing)}問)、スキップ")
            return existing
        print(f"  {model}: 途中から再開 (既完了 {len(existing)}問)")

    results = list(existing)
    start_idx = len(existing)
    t0 = time.time()

    for idx in range(start_idx, len(questions)):
        item = questions[idx]
        prompt = format_question(item)
        try:
            raw = call_ollama(model, prompt, temperature=0.0)
            parsed = parse_answer(raw, len(item["options"]))
        except Exception as e:
            raw = f"ERROR: {type(e).__name__}: {e}"
            parsed = None

        results.append({
            "question_id": int(item["question_id"]),
            "raw_response": raw,
            "parsed_answer": parsed,
            "correct_answer": item["answer"],
            "n_options": len(item["options"]),
        })

        if (idx + 1) % PROGRESS_EVERY == 0 or (idx + 1) == len(questions):
            correct = sum(
                1 for r in results if r["parsed_answer"] == r["correct_answer"]
            )
            parse_fail = sum(1 for r in results if r["parsed_answer"] is None)
            elapsed = time.time() - t0
            rate = (idx + 1 - start_idx) / max(elapsed, 1e-9)
            eta = (len(questions) - idx - 1) / max(rate, 1e-9)
            print(f"    {model}: {idx+1}/{len(questions)} "
                  f"(正解率 {correct/(idx+1)*100:.1f}%, "
                  f"パース失敗 {parse_fail}, "
                  f"速度 {rate:.2f}問/秒, "
                  f"残り約 {eta/60:.1f}分)", flush=True)
            # 逐次保存
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    return results


def letter_to_index(letter):
    if not isinstance(letter, str) or len(letter) != 1:
        return -1
    idx = ord(letter.upper()) - ord("A")
    return idx if 0 <= idx < N_OPTIONS_MAX else -1


def confirm_or_exit(total_inf):
    msg = (
        "ローカル LLM 推論の見積もり:\n"
        f"  問題数: {N_QUESTIONS}\n"
        f"  モデル数: {len(OLLAMA_MODELS)}\n"
        f"  推定推論回数: {total_inf}\n"
        "  推定所要時間: 約1〜3時間（GPU 性能に依存）"
    )
    print(msg)
    if os.environ.get("CAGL_CONFIRM") == "yes":
        print("CAGL_CONFIRM=yes により自動承認")
        return
    if not sys.stdin.isatty():
        print()
        print("非対話モードで実行されています。")
        print("続行するには CAGL_CONFIRM=yes を環境変数に設定してください。")
        sys.exit(1)
    resp = input("続行しますか？ [y/N]: ").strip().lower()
    if resp not in ("y", "yes"):
        print("中止しました")
        sys.exit(0)


def main():
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    # 1) 問題サンプリング（キャッシュ対応）
    questions_path = LOCAL_DIR / "questions.json"
    if questions_path.exists():
        with open(questions_path, "r", encoding="utf-8") as f:
            sampled = json.load(f)
        print(f"既存の問題ファイルを再利用: {len(sampled)} 問")
    else:
        sampled_ds = sample_questions()
        # datasets の Row を dict に落とす
        sampled = [
            {
                "question_id": int(x["question_id"]),
                "question": x["question"],
                "options": list(x["options"]),
                "answer": x["answer"],
                "answer_index": int(x["answer_index"]),
                "category": x["category"],
            }
            for x in sampled_ds
        ]
        with open(questions_path, "w", encoding="utf-8") as f:
            json.dump(sampled, f, ensure_ascii=False, indent=2)
        print(f"問題を保存: {questions_path} ({len(sampled)} 問)")

    # 2) 利用確認
    total_inf = len(sampled) * len(OLLAMA_MODELS)
    confirm_or_exit(total_inf)

    # 3) モデル利用可能性チェック
    print()
    print("Ollama モデル可用性チェック...")
    available = []
    for m in OLLAMA_MODELS:
        if ensure_model_available(m):
            print(f"  {m}: OK")
            available.append(m)
        else:
            print(f"  {m}: 利用不可 — スキップします")
    if not available:
        print("利用可能なモデルがありません。終了します。")
        sys.exit(1)

    # 4) 各モデルで推論
    responses_dir = LOCAL_DIR / "responses"
    responses_dir.mkdir(parents=True, exist_ok=True)
    all_responses = {}
    per_model_stats = {}
    for model in available:
        print()
        print(f"=== {model} ===")
        safe_name = model.replace(":", "_").replace("/", "_")
        out_path = responses_dir / f"{safe_name}.json"
        results = infer_one_model(model, sampled, out_path)
        all_responses[model] = results
        correct = sum(
            1 for r in results if r["parsed_answer"] == r["correct_answer"]
        )
        parse_fail = sum(1 for r in results if r["parsed_answer"] is None)
        per_model_stats[model] = {
            "全体正解率": correct / len(results) if results else 0.0,
            "パース失敗数": parse_fail,
            "総問題数": len(results),
        }
        print(f"  {model} 最終: 正解率 "
              f"{per_model_stats[model]['全体正解率']*100:.2f}%, "
              f"パース失敗 {parse_fail}/{len(results)}")

    # 5) 統合回答ファイル
    responses_path = LOCAL_DIR / "responses.json"
    with open(responses_path, "w", encoding="utf-8") as f:
        json.dump(all_responses, f, ensure_ascii=False)

    # 6) CAGL 入力形式に変換
    # 全モデルでパース成功した問題のみ採用
    model_list = list(all_responses.keys())
    qid_to_parsed = {m: {r["question_id"]: r["parsed_answer"]
                         for r in all_responses[m]} for m in model_list}
    qid_list = [q["question_id"] for q in sampled]
    common_qids = [
        qid for qid in qid_list
        if all(qid_to_parsed[m].get(qid) is not None for m in model_list)
    ]
    print(f"\n全モデルでパース成功した問題: {len(common_qids)} / {len(qid_list)}")

    qid_to_item = {q["question_id"]: q for q in sampled}
    n_q = len(common_qids)
    n_models = len(model_list)
    pred = np.zeros((n_q, n_models, N_OPTIONS_MAX), dtype=np.uint8)
    gt = np.zeros((n_q, N_OPTIONS_MAX), dtype=np.uint8)
    categories = []
    for i, qid in enumerate(common_qids):
        item = qid_to_item[qid]
        gt_idx = letter_to_index(item["answer"])
        if gt_idx >= 0:
            gt[i, gt_idx] = 1
        for m_i, m in enumerate(model_list):
            p_idx = letter_to_index(qid_to_parsed[m][qid])
            if p_idx >= 0:
                pred[i, m_i, p_idx] = 1
        categories.append(item["category"])

    pred_gt_path = LOCAL_DIR / "pred_gt.npz"
    np.savez_compressed(
        pred_gt_path,
        pred=pred,
        gt=gt,
        model_names=np.array(model_list),
        question_ids=np.array(common_qids, dtype=np.int64),
        n_options_max=np.int64(N_OPTIONS_MAX),
    )

    meta_path = LOCAL_DIR / "questions_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "question_ids": common_qids,
                "categories": categories,
                "source": "TIGER-Lab/MMLU-Pro + ローカル Ollama 推論",
            },
            f, ensure_ascii=False, indent=2,
        )

    # カテゴリ別正解率（共通問題集合上）
    for m in model_list:
        by_cat_total, by_cat_correct = {}, {}
        for qid in common_qids:
            cat = qid_to_item[qid]["category"]
            true_ans = qid_to_item[qid]["answer"]
            by_cat_total[cat] = by_cat_total.get(cat, 0) + 1
            if qid_to_parsed[m][qid] == true_ans:
                by_cat_correct[cat] = by_cat_correct.get(cat, 0) + 1
        per_model_stats[m]["カテゴリ別正解率"] = {
            c: by_cat_correct.get(c, 0) / by_cat_total[c]
            for c in by_cat_total
        }
        per_model_stats[m]["共通問題上の正解率"] = (
            sum(1 for qid in common_qids
                if qid_to_parsed[m][qid] == qid_to_item[qid]["answer"])
            / max(1, len(common_qids))
        )

    manifest = {
        "生成日時": datetime.now(timezone.utc).isoformat(),
        "データソース": "TIGER-Lab/MMLU-Pro (HuggingFace) + ローカル Ollama",
        "Ollamaバージョン": os.popen("ollama --version").read().strip(),
        "総問題数": len(sampled),
        "共通問題数": n_q,
        "最大選択肢数": N_OPTIONS_MAX,
        "モデル一覧": per_model_stats,
        "ファイルハッシュ": {
            "pred_gt.npz": sha256_of_file(pred_gt_path),
            "questions.json": sha256_of_file(questions_path),
            "responses.json": sha256_of_file(responses_path),
            "questions_meta.json": sha256_of_file(meta_path),
        },
    }
    manifest_path = LOCAL_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print()
    print(f"出力: {pred_gt_path}")
    print(f"    : {manifest_path}")
    print(f"pred shape={pred.shape}, gt shape={gt.shape}")


if __name__ == "__main__":
    main()
