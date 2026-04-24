"""Phase 4b 段階C: YOLO 切り抜き画像を DINOv2 で特徴抽出し、音声特徴と時間窓で統合する。

- 切り抜き画像 → DINOv2 ViT-S/14 (384次元) → 動画単位で平均
- 音声窓特徴 → 動画単位で平均 (15次元)
- メタ特徴 5次元 (フレーム数・窓数・動画長・映像最大値平均・音声最大確信度)
- 結合特徴 404次元

splits.json の train_set / val_set / test_set で分割して保存。

このスクリプトは YOLO や BirdNET をインポートしない。
"""

import json
import os
import time

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


DATA_DIR = os.path.join("..", "data", "wetlandbirds")
OUT_DIR = os.path.join("..", "results", "bird_phase4b")

VISUAL_DIM = 384
AUDIO_DIM = 15
META_DIM = 5


def load_dinov2():
    print("DINOv2 読み込み中...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def get_transform():
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


def extract_frame_features(model, transform, crop_paths, batch_size=32):
    """切り抜き画像リストから DINOv2 特徴を抽出（バッチ処理）。"""
    if not crop_paths:
        return np.zeros((0, VISUAL_DIM), dtype=np.float32)

    device = next(model.parameters()).device
    feats = []
    for i in range(0, len(crop_paths), batch_size):
        batch_paths = crop_paths[i:i + batch_size]
        tensors = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                tensors.append(transform(img))
            except Exception:
                tensors.append(torch.zeros(3, 224, 224))
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            out = model(batch)
        feats.append(out.cpu().numpy().astype(np.float32))
    return np.concatenate(feats, axis=0)


def build_video_feature(frame_features, audio_windows, duration):
    """映像特徴（フレーム平均）・音声特徴（窓平均）・メタの統合ベクトルを構築。"""
    if len(frame_features) > 0:
        visual = frame_features.mean(axis=0).astype(np.float32)
        mean_max_visual = float(np.mean([f.max() for f in frame_features]))
    else:
        visual = np.zeros(VISUAL_DIM, dtype=np.float32)
        mean_max_visual = 0.0

    if audio_windows:
        audio_arr = np.array([w["features"] for w in audio_windows],
                             dtype=np.float32)
        audio = audio_arr.mean(axis=0)
    else:
        audio = np.zeros(AUDIO_DIM, dtype=np.float32)

    meta = np.array([
        float(len(frame_features)),
        float(len(audio_windows)),
        float(duration),
        mean_max_visual,
        float(audio[0]) if audio.size > 0 else 0.0,
    ], dtype=np.float32)

    combined = np.concatenate([visual, audio, meta]).astype(np.float32)
    return {"visual": visual, "audio": audio, "meta": meta, "combined": combined}


def main():
    print("=== Phase 4b-C: DINOv2 特徴抽出 + 時間窓統合 ===")

    with open(os.path.join(OUT_DIR, "frame_results.json"), encoding="utf-8") as f:
        frame_data = json.load(f)
    frame_results = frame_data["videos"]

    with open(os.path.join(OUT_DIR, "audio_results.json"), encoding="utf-8") as f:
        audio_data = json.load(f)
    audio_results = audio_data["videos"]

    with open(os.path.join(DATA_DIR, "splits.json"), encoding="utf-8") as f:
        splits = json.load(f)

    model = load_dinov2()
    transform = get_transform()

    all_features = {}
    start_total = time.time()

    for i, (video_name, vinfo) in enumerate(frame_results.items()):
        crops = vinfo.get("crops", [])
        crop_paths = [c["crop_path"] for c in crops
                      if c.get("crop_path") and os.path.exists(c["crop_path"])]

        frame_features = extract_frame_features(model, transform, crop_paths)

        ainfo = audio_results.get(video_name, {})
        audio_windows = ainfo.get("windows", [])
        duration = vinfo.get("duration", 0.0)

        fv = build_video_feature(frame_features, audio_windows, duration)

        all_features[video_name] = {
            "species": vinfo["species"],
            "visual": fv["visual"],
            "audio": fv["audio"],
            "meta": fv["meta"],
            "combined": fv["combined"],
        }

        if (i + 1) % 20 == 0 or (i + 1) == len(frame_results):
            elapsed = time.time() - start_total
            print(f"  {i+1}/{len(frame_results)} 処理完了 ({elapsed:.0f}秒)")

    elapsed_total = time.time() - start_total
    print(f"\n処理時間: {elapsed_total:.1f} 秒")

    # splits.json の値は "004-squacco_heron" 形式（拡張子なし）
    split_assignments = {}
    unmatched_split = []
    for split_name, stem_list in splits.items():
        for stem in stem_list:
            fname = f"{stem}.mp4"
            if fname in all_features:
                split_assignments[fname] = split_name
            else:
                unmatched_split.append((split_name, stem))
    if unmatched_split:
        print(f"\n警告: splits.json にあるが動画が見つからないエントリ: {len(unmatched_split)}")
        for sp, s in unmatched_split[:5]:
            print(f"  {sp} / {s}")

    not_in_split = [v for v in all_features if v not in split_assignments]
    if not_in_split:
        print(f"警告: splits.json に含まれない動画: {len(not_in_split)}")
        for v in not_in_split[:5]:
            print(f"  {v}")

    print(f"\n分割マッピング: {len(split_assignments)} 動画")
    for split_name in splits:
        n = sum(1 for v in split_assignments.values() if v == split_name)
        print(f"  {split_name}: {n} 動画")

    # 種名 → 数値ラベル
    all_species = sorted({v["species"] for v in all_features.values()})
    species_to_id = {s: i for i, s in enumerate(all_species)}
    print(f"\n種数: {len(all_species)}")
    for s, i in species_to_id.items():
        print(f"  {i}: {s}")

    # 分割ごとに npz を書き出し
    for split_name in splits:
        videos_in_split = [v for v, s in split_assignments.items() if s == split_name]
        if not videos_in_split:
            print(f"警告: {split_name} に動画がありません。スキップ。")
            continue

        X_visual = np.stack([all_features[v]["visual"] for v in videos_in_split])
        X_audio = np.stack([all_features[v]["audio"] for v in videos_in_split])
        X_meta = np.stack([all_features[v]["meta"] for v in videos_in_split])
        X_combined = np.stack([all_features[v]["combined"] for v in videos_in_split])
        y = np.array([species_to_id[all_features[v]["species"]]
                      for v in videos_in_split])

        # splits.json のキー "train_set" などをそのまま保存ファイル名に反映
        out_path = os.path.join(OUT_DIR, f"features_{split_name}.npz")
        np.savez(out_path,
                 X_visual=X_visual, X_audio=X_audio,
                 X_meta=X_meta, X_combined=X_combined,
                 y=y, video_names=np.array(videos_in_split))
        print(f"保存: {out_path} "
              f"(visual={X_visual.shape}, audio={X_audio.shape}, "
              f"combined={X_combined.shape})")

    out_map = os.path.join(OUT_DIR, "species_mapping.json")
    with open(out_map, "w", encoding="utf-8") as f:
        json.dump({
            "species_to_id": species_to_id,
            "id_to_species": {str(v): k for k, v in species_to_id.items()},
            "n_classes": len(all_species),
        }, f, indent=2, ensure_ascii=False)
    print(f"保存: {out_map}")


if __name__ == "__main__":
    main()
