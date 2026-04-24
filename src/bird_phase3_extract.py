"""Phase 3 段階C: BirdNET で音声から種の確信度特徴を抽出する。

BirdNET v2.4 モデルを用いて、各音声ファイルの予測結果を固定長のベクトルに
変換する。top-k の種の確信度 + 簡単な統計量を特徴量とする。
このスクリプトは DINOv2 や MLP をインポートしない（BirdNET のみ使用）。
"""

import json
import os

import numpy as np

# TensorFlow の大量ログを抑制
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


PHASE3_DIR = os.path.join("..", "results", "bird_phase3")
TOP_K = 10
# 特徴量次元: top_k 確信度 + 統計量 5 つ
FEAT_DIM = TOP_K + 5


def build_feature_vector(prediction_pairs):
    """BirdNET の予測 (種名, 確信度) のリストから固定長ベクトルを生成"""
    if not prediction_pairs:
        return np.zeros(FEAT_DIM, dtype=np.float32)

    confidences = sorted([float(c) for _, c in prediction_pairs], reverse=True)
    confidences = confidences[:TOP_K]
    while len(confidences) < TOP_K:
        confidences.append(0.0)
    conf_array = np.array(confidences, dtype=np.float32)

    features = list(conf_array)
    features.append(float(conf_array.max()))
    features.append(float(conf_array.mean()))
    features.append(float(conf_array.std()))
    features.append(float((conf_array > 0.5).sum()))
    if len(conf_array) >= 2:
        features.append(float(conf_array[0] - conf_array[1]))
    else:
        features.append(0.0)
    return np.array(features, dtype=np.float32)


def extract_predictions(model, audio_path):
    """1 ファイルに BirdNET を適用し、(種名, 確信度) のリストを返す"""
    try:
        result = model.predict(
            audio_path,
            top_k=TOP_K,
            apply_sigmoid=True,
            default_confidence_threshold=0.0,
            show_stats=None,
        )
    except Exception as e:
        print(f"    予測失敗: {os.path.basename(audio_path)}: {e}")
        return []

    pairs = []
    try:
        arr = result.to_structured_array()
    except Exception as e:
        print(f"    結果変換失敗: {e}")
        return []

    # 構造化配列の各行 = 1 セグメントの上位種
    # フィールド: (input, start_time, end_time, species_name, confidence)
    names = arr.dtype.names or ()
    if "species_name" in names and "confidence" in names:
        for row in arr:
            prob = float(row["confidence"])
            if prob <= 0.0:
                continue
            name = str(row["species_name"])
            pairs.append((name, prob))
    else:
        # フォールバック: species_probs プロパティを使う
        try:
            probs_matrix = np.asarray(result.species_probs)
            species_list = list(result.species_list)
            for seg in probs_matrix:
                seg = np.asarray(seg).ravel()
                top_idx = np.argsort(seg)[::-1][:TOP_K]
                for sid in top_idx:
                    prob = float(seg[sid])
                    if prob <= 0.0:
                        continue
                    name = species_list[sid] if 0 <= sid < len(species_list) else str(int(sid))
                    pairs.append((name, prob))
        except Exception as e:
            print(f"    species_probs 取得失敗: {e}")

    return pairs


def main():
    print("=== Phase 3C: BirdNET 音声特徴抽出 ===")

    manifest_path = os.path.join(PHASE3_DIR, "audio_manifest.json")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    # BirdNET モデルのロード（初回は重みを自動ダウンロード）
    print("\nBirdNET モデルをロード中（初回はモデル重みのダウンロードが発生）...")
    from birdnet import load
    model = load("acoustic", "2.4", "tf")
    print("  モデルロード完了")

    # 動作確認用の最初のファイル
    test_file = None
    for key, val in manifest.items():
        if key == "negative":
            continue
        recs = val.get("recordings", [])
        if recs:
            test_file = recs[0]["path"]
            break

    if test_file is None:
        print("エラー: テスト用の音声ファイルが見つかりません")
        return

    print(f"\n動作確認 (テストファイル): {test_file}")
    test_pairs = extract_predictions(model, test_file)
    if not test_pairs:
        print("警告: 初回予測が空でした（閾値や音声内容を確認してください）")
    else:
        print(f"  予測件数: {len(test_pairs)}, 例: {test_pairs[:3]}")

    test_feat = build_feature_vector(test_pairs)
    print(f"  特徴量次元: {len(test_feat)} (= top_k {TOP_K} + stats 5)")

    # 全ファイルの特徴抽出
    all_features = {}
    for key, val in manifest.items():
        recordings = val.get("recordings", [])
        if not recordings:
            continue
        label = val.get("species_name", key)
        print(f"\n--- [{key}] {label} ({len(recordings)} 件) ---")

        feats = []
        for i, rec in enumerate(recordings):
            path = rec["path"]
            if not os.path.exists(path):
                print(f"  警告: {path} が見つかりません")
                feats.append(np.zeros(FEAT_DIM, dtype=np.float32))
                continue
            pairs = extract_predictions(model, path)
            feats.append(build_feature_vector(pairs))
            if (i + 1) % 5 == 0:
                print(f"  {i+1}/{len(recordings)} 完了")

        arr = np.stack(feats).astype(np.float32) if feats else np.zeros((0, FEAT_DIM), dtype=np.float32)
        all_features[key] = arr
        print(f"  完了: shape {arr.shape}")

    # 保存
    save_dict = {f"features_{k}": v for k, v in all_features.items()}
    save_dict["feat_dim"] = np.array(FEAT_DIM)
    out_path = os.path.join(PHASE3_DIR, "features_birdnet.npz")
    np.savez(out_path, **save_dict)

    print(f"\n保存: {out_path}")
    for k, v in all_features.items():
        print(f"  {k}: {v.shape}")


if __name__ == "__main__":
    main()
