"""
段階A: 既存のローカル LLM 予測から MLP 用特徴量を構築する。

results/local_llm/responses.json と questions.json を読み込み、各問題に
ついて以下の特徴量ベクトルを作る:
  1) 各モデルの予測 one-hot (5モデル × 10選択肢)
  2) 選択肢別の投票数 (10)
  3) 最大合意度 (1)
  4) モデルペアの予測一致 (5C2 = 10)
  5) カテゴリ one-hot (14)
全モデルでパース成功した問題のみを採用。追加の LLM 推論は行わない。

このスクリプトは core_cagl をインポートしない。
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
LOCAL_DIR = RESULTS_DIR / "local_llm"
MLP_DIR = RESULTS_DIR / "mlp"
N_OPTIONS_MAX = 10


def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def letter_to_index(letter):
    if not isinstance(letter, str) or len(letter) != 1:
        return -1
    idx = ord(letter.upper()) - ord("A")
    return idx if 0 <= idx < N_OPTIONS_MAX else -1


def main():
    MLP_DIR.mkdir(parents=True, exist_ok=True)

    with open(LOCAL_DIR / "responses.json", "r", encoding="utf-8") as f:
        responses = json.load(f)
    with open(LOCAL_DIR / "questions.json", "r", encoding="utf-8") as f:
        questions = json.load(f)

    model_names = list(responses.keys())
    n_models = len(model_names)
    print(f"モデル数: {n_models}  /  モデル一覧: {model_names}")
    print(f"総問題数: {len(questions)}")

    # question_id → モデル別の parsed_answer を索引化
    qid2parsed = {
        m: {r["question_id"]: r["parsed_answer"] for r in responses[m]}
        for m in model_names
    }

    # カテゴリ一覧（アルファベット順で固定）
    all_categories = sorted({q["category"] for q in questions})
    print(f"カテゴリ数: {len(all_categories)}  /  {all_categories}")

    # 特徴量名を先に作る
    feature_names = []
    # グループ1: 各モデルの予測 one-hot
    for m in model_names:
        for i in range(N_OPTIONS_MAX):
            feature_names.append(f"{m}_pred_{chr(65+i)}")
    # グループ2: 選択肢別投票数
    for i in range(N_OPTIONS_MAX):
        feature_names.append(f"vote_count_{chr(65+i)}")
    # グループ3: 最大合意度
    feature_names.append("max_agreement")
    # グループ4: ペア一致
    for i in range(n_models):
        for j in range(i + 1, n_models):
            feature_names.append(f"pair_{model_names[i]}__vs__{model_names[j]}")
    # グループ5: カテゴリ one-hot
    for c in all_categories:
        feature_names.append(f"cat_{c}")

    expected_dim = (
        n_models * N_OPTIONS_MAX
        + N_OPTIONS_MAX
        + 1
        + n_models * (n_models - 1) // 2
        + len(all_categories)
    )
    assert len(feature_names) == expected_dim, \
        f"特徴量名の数 {len(feature_names)} != 期待値 {expected_dim}"
    print(f"特徴量次元: {expected_dim} "
          f"(モデル予測 {n_models * N_OPTIONS_MAX} + "
          f"投票 {N_OPTIONS_MAX} + "
          f"合意 1 + "
          f"ペア {n_models * (n_models - 1) // 2} + "
          f"カテゴリ {len(all_categories)})")

    # 全モデルでパース成功した問題のみ採用
    X_list, y_list, qid_list, cat_list = [], [], [], []
    for q in questions:
        qid = q["question_id"]
        answers = [qid2parsed[m].get(qid) for m in model_names]
        if any(a is None for a in answers):
            continue

        feat = []
        # グループ1
        for a in answers:
            oh = [0] * N_OPTIONS_MAX
            idx = letter_to_index(a)
            if idx >= 0:
                oh[idx] = 1
            feat.extend(oh)
        # グループ2
        votes = [0] * N_OPTIONS_MAX
        for a in answers:
            idx = letter_to_index(a)
            if idx >= 0:
                votes[idx] += 1
        feat.extend(votes)
        # グループ3
        feat.append(max(votes) / n_models)
        # グループ4
        for i in range(n_models):
            for j in range(i + 1, n_models):
                feat.append(1 if answers[i] == answers[j] else 0)
        # グループ5
        cat_oh = [0] * len(all_categories)
        cat_oh[all_categories.index(q["category"])] = 1
        feat.extend(cat_oh)

        X_list.append(feat)
        y_list.append(letter_to_index(q["answer"]))
        qid_list.append(qid)
        cat_list.append(q["category"])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    print(f"\n採用問題数: {len(y)} / {len(questions)}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    out_path = MLP_DIR / "dataset.npz"
    np.savez(
        out_path,
        X=X,
        y=y,
        n_options=np.int64(N_OPTIONS_MAX),
        model_names=np.array(model_names),
        categories=np.array(all_categories),
        feature_names=np.array(feature_names),
        question_ids=np.array(qid_list, dtype=np.int64),
        question_categories=np.array(cat_list),
    )

    manifest = {
        "生成日時": datetime.now(timezone.utc).isoformat(),
        "ソース": "results/local_llm/responses.json + questions.json",
        "採用問題数": int(len(y)),
        "特徴量次元": int(X.shape[1]),
        "モデル数": n_models,
        "選択肢最大数": N_OPTIONS_MAX,
        "カテゴリ数": len(all_categories),
        "特徴量グループ構成": {
            "モデル別予測 one-hot": n_models * N_OPTIONS_MAX,
            "選択肢別投票数": N_OPTIONS_MAX,
            "最大合意度": 1,
            "ペア予測一致": n_models * (n_models - 1) // 2,
            "カテゴリ one-hot": len(all_categories),
        },
        "ファイルハッシュ": {
            "dataset.npz": sha256_of_file(out_path),
        },
    }
    with open(MLP_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"出力: {out_path}")
    print(f"    : {MLP_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
