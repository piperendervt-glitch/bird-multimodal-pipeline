"""Phase 3b 段階B: BirdNET で音声から種の確信度特徴を抽出する。

Phase 3 の bird_phase3_extract.py と同じロジック。入出力だけ phase3b 用に差し替え。
"""

import json
import os

import numpy as np

# TensorFlow の大量ログを抑制
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


PHASE3B_DIR = os.path.join("..", "results", "bird_phase3b")
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

    names = arr.dtype.names or ()
    if "species_name" in names and "confidence" in names:
        for row in arr:
            prob = float(row["confidence"])
            if prob <= 0.0:
                continue
            name = str(row["species_name"])
            pairs.append((name, prob))
    else:
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
    print("=== Phase 3b-B: BirdNET 音声特徴抽出 ===")

    manifest_path = os.path.join(PHASE3B_DIR, "audio_manifest.json")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    print("\nBirdNET モデルをロード中...")
    from birdnet import load
    model = load("acoustic", "2.4", "tf")
    print("  モデルロード完了")

    # 動作確認
    test_file = None
    for key, val in manifest.items():
        if key == "negative":
            continue
        for rec in val.get("recordings", []):
            if os.path.exists(rec.get("path", "")):
                test_file = rec["path"]
                break
        if test_file:
            break

    if test_file:
        print(f"\n動作確認 (テストファイル): {test_file}")
        test_pairs = extract_predictions(model, test_file)
        if not test_pairs:
            print("警告: 初回予測が空でした")
        else:
            print(f"  予測件数: {len(test_pairs)}, 例: {test_pairs[:3]}")
    else:
        print("エラー: 利用可能な音声ファイルが見つかりません")
        return

    # 全ファイル抽出
    all_features = {}
    for key, val in manifest.items():
        recordings = val.get("recordings", [])
        if not recordings:
            continue
        label = val.get("species_name", key)
        print(f"\n--- [{key}] {label} ({len(recordings)} 件) ---")
        feats = []
        for i, rec in enumerate(recordings):
            path = rec.get("path", "")
            if not path or not os.path.exists(path):
                print(f"  警告: {path} が見つかりません")
                feats.append(np.zeros(FEAT_DIM, dtype=np.float32))
                continue
            pairs = extract_predictions(model, path)
            feats.append(build_feature_vector(pairs))
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(recordings)} 完了")
        arr = np.stack(feats).astype(np.float32) if feats else np.zeros((0, FEAT_DIM), dtype=np.float32)
        all_features[key] = arr
        print(f"  完了: shape {arr.shape}")

    save_dict = {f"features_{k}": v for k, v in all_features.items()}
    save_dict["feat_dim"] = np.array(FEAT_DIM)
    out_path = os.path.join(PHASE3B_DIR, "features_birdnet.npz")
    np.savez(out_path, **save_dict)

    print(f"\n保存: {out_path}")
    for k, v in all_features.items():
        nonzero = int((v.sum(axis=1) > 0).sum())
        print(f"  {k}: {v.shape} (有効 {nonzero}/{len(v)})")


if __name__ == "__main__":
    main()
