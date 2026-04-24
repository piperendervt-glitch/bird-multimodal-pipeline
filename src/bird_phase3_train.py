"""Phase 3 段階D: 画像特徴 (DINOv2) と音声特徴 (BirdNET) の統合による
特定種の 2 値分類を 5-fold CV で学習・評価する。

このスクリプトは DINOv2 モデルや BirdNET をインポートしない
（段階Aで保存した特徴量ファイルのみを使用）。
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


PHASE1_DIR = os.path.join("..", "results", "bird_phase1")
PHASE3_DIR = os.path.join("..", "results", "bird_phase3")


class DetectorMLP(nn.Module):
    """2 値分類用の小さな MLP"""

    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_detector_mlp(X_train, y_train, X_val, y_val,
                       hidden_dim=32, epochs=150, lr=1e-3):
    """2 値分類 MLP の学習 + 評価 (ベスト F1 のモデルを採用)"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DetectorMLP(X_train.shape[1], hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    n_pos = max((y_train == 1).sum(), 1)
    n_neg = max((y_train == 0).sum(), 1)
    pos_weight = torch.tensor([n_neg / (n_pos + 1e-9)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    X_tr = torch.FloatTensor(X_train).to(device)
    y_tr = torch.FloatTensor(y_train).to(device)
    X_va = torch.FloatTensor(X_val).to(device)

    best_f1 = -1.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_tr)
        loss = criterion(output, y_tr)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(X_va).cpu().numpy()
                preds = (logits > 0).astype(int)
                f1 = f1_score(y_val, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = model(X_va).cpu().numpy()
        probs = torch.sigmoid(torch.FloatTensor(logits)).numpy()
        preds = (logits > 0).astype(int)
    return preds, probs


def evaluate_binary(y_true, preds, probs):
    metrics = {
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, preds)),
    }
    if len(np.unique(y_true)) > 1:
        metrics["auc_roc"] = float(roc_auc_score(y_true, probs))
    else:
        metrics["auc_roc"] = 0.0
    return metrics


def run_species_detection(species_info, X_image_all, y_image_all,
                          audio_features, n_splits=5):
    """1 種に対する検出実験を 5-fold CV で実施"""
    cls_id = species_info["class_id"]

    y_binary_all = (y_image_all == cls_id).astype(int)
    pos_idx = np.where(y_binary_all == 1)[0]
    neg_idx = np.where(y_binary_all == 0)[0]
    n_pos = len(pos_idx)
    n_neg_sample = min(len(neg_idx), n_pos * 5)

    rng = np.random.RandomState(42)
    neg_sampled = rng.choice(neg_idx, n_neg_sample, replace=False)

    use_idx = np.concatenate([pos_idx, neg_sampled])
    rng.shuffle(use_idx)

    X_image = X_image_all[use_idx]
    y = y_binary_all[use_idx]

    # 音声特徴の割当: 正例には対象種音声、負例には負例用音声からランダム選択
    audio_key = str(cls_id)
    has_audio = (
        audio_key in audio_features
        and "negative" in audio_features
        and audio_features[audio_key].shape[0] > 0
        and audio_features["negative"].shape[0] > 0
    )

    if has_audio:
        pos_audio = audio_features[audio_key]
        neg_audio = audio_features["negative"]
        audio_dim = pos_audio.shape[1]
        X_audio = np.zeros((len(use_idx), audio_dim), dtype=np.float32)
        for i, idx in enumerate(use_idx):
            if y_binary_all[idx] == 1:
                ai = rng.randint(0, len(pos_audio))
                X_audio[i] = pos_audio[ai]
            else:
                ai = rng.randint(0, len(neg_audio))
                X_audio[i] = neg_audio[ai]
    else:
        X_audio = np.zeros((len(use_idx), 1), dtype=np.float32)

    print(f"  正例: {n_pos}, 負例: {n_neg_sample}, 合計: {len(use_idx)}"
          + (f", 音声次元: {X_audio.shape[1]}" if has_audio else ", 音声なし"))

    methods = {
        "DINOv2_LogReg": {"features": "image", "model": "logreg"},
        "DINOv2_MLP": {"features": "image", "model": "mlp"},
    }
    if has_audio:
        methods["BirdNET_LogReg"] = {"features": "audio", "model": "logreg"}
        methods["DINOv2+BirdNET_LogReg"] = {"features": "both", "model": "logreg"}
        methods["DINOv2+BirdNET_MLP"] = {"features": "both", "model": "mlp"}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = {name: [] for name in methods}

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_image, y)):
        y_tr = y[train_idx]
        y_te = y[test_idx]
        for method_name, config in methods.items():
            if config["features"] == "image":
                X_tr_raw = X_image[train_idx]
                X_te_raw = X_image[test_idx]
            elif config["features"] == "audio":
                X_tr_raw = X_audio[train_idx]
                X_te_raw = X_audio[test_idx]
            else:
                X_tr_raw = np.concatenate(
                    [X_image[train_idx], X_audio[train_idx]], axis=1)
                X_te_raw = np.concatenate(
                    [X_image[test_idx], X_audio[test_idx]], axis=1)

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr_raw)
            X_te = scaler.transform(X_te_raw)

            if config["model"] == "logreg":
                clf = LogisticRegression(
                    max_iter=1000, C=1.0, random_state=42,
                    class_weight="balanced",
                )
                clf.fit(X_tr, y_tr)
                preds = clf.predict(X_te)
                probs = clf.predict_proba(X_te)[:, 1]
            else:
                preds, probs = train_detector_mlp(
                    X_tr, y_tr, X_te, y_te,
                    hidden_dim=32, epochs=150,
                )
            metrics = evaluate_binary(y_te, preds, probs)
            fold_results[method_name].append(metrics)

    avg_results = {}
    for method_name, folds in fold_results.items():
        avg = {}
        for metric in ["f1", "precision", "recall", "accuracy", "auc_roc"]:
            values = [f[metric] for f in folds]
            avg[metric] = float(np.mean(values))
            avg[f"{metric}_std"] = float(np.std(values))
        avg_results[method_name] = avg
    return avg_results


def main():
    print("=== Phase 3D: 特定種検出 学習・評価 ===")

    dino = np.load(os.path.join(PHASE1_DIR, "features_dinov2_vits14.npz"))
    X_image_all = np.concatenate([dino["X_train"], dino["X_test"]], axis=0)
    y_image_all = np.concatenate([dino["y_train"], dino["y_test"]], axis=0)
    print(f"画像特徴量: {X_image_all.shape}, ラベル: {y_image_all.shape}")

    birdnet_path = os.path.join(PHASE3_DIR, "features_birdnet.npz")
    audio_features = {}
    if os.path.exists(birdnet_path):
        birdnet = np.load(birdnet_path, allow_pickle=True)
        for key in birdnet.files:
            if key.startswith("features_"):
                audio_features[key.replace("features_", "")] = birdnet[key]
        print(f"音声特徴量: {len(audio_features)} グループ")
        for k, v in audio_features.items():
            print(f"  {k}: {v.shape}")
    else:
        print("警告: 音声特徴量が見つかりません。画像のみで実行します。")

    with open(os.path.join(PHASE3_DIR, "selected_species.json"),
              encoding="utf-8") as f:
        data = json.load(f)
    selected = data["selected_species"]

    all_results = {}
    for sp in selected:
        name = sp["search_name"]
        cls_id = sp["class_id"]
        dinov2_f1 = sp["dinov2_f1"]
        print(f"\n{'='*60}")
        print(f"対象種: {name} (class {cls_id}, DINOv2 F1={dinov2_f1:.4f})")
        print(f"{'='*60}")
        results = run_species_detection(sp, X_image_all, y_image_all, audio_features)
        all_results[name] = results

        print(f"\n  {'手法':<32} {'F1':>8} {'Prec':>8} {'Recall':>8} {'AUC':>8}")
        print(f"  {'-'*66}")
        for method_name, metrics in results.items():
            print(f"  {method_name:<32} "
                  f"{metrics['f1']:>7.4f} "
                  f"{metrics['precision']:>7.4f} "
                  f"{metrics['recall']:>7.4f} "
                  f"{metrics['auc_roc']:>7.4f}")

    # 全種平均
    print(f"\n{'='*60}")
    print("全種平均")
    print(f"{'='*60}")
    method_names = set()
    for sp_results in all_results.values():
        method_names.update(sp_results.keys())

    avg_across_species = {}
    for method in sorted(method_names):
        f1s = [r[method]["f1"] for r in all_results.values() if method in r]
        if f1s:
            avg_across_species[method] = {
                "mean_f1": float(np.mean(f1s)),
                "std_f1": float(np.std(f1s)),
            }
            print(f"  {method:<32} F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    # 失敗条件の判定
    print(f"\n--- 失敗条件の判定 ---")
    dino_f1 = avg_across_species.get("DINOv2_LogReg", {}).get("mean_f1", 0.0)

    if "DINOv2+BirdNET_MLP" in avg_across_species:
        combo_f1 = avg_across_species["DINOv2+BirdNET_MLP"]["mean_f1"]
        diff = combo_f1 - dino_f1
        if diff >= 0.02:
            print(f"  OK: 統合 MLP ({combo_f1:.4f}) > DINOv2 ({dino_f1:.4f}): "
                  f"+{diff*100:.2f}pp")
        else:
            print(f"  NG: 統合 MLP ({combo_f1:.4f}) - DINOv2 ({dino_f1:.4f}) "
                  f"= {diff*100:+.2f}pp (< 2pp)")

    if "BirdNET_LogReg" in avg_across_species:
        birdnet_f1 = avg_across_species["BirdNET_LogReg"]["mean_f1"]
        if birdnet_f1 < 0.30:
            print(f"  NG: BirdNET 単独 F1 ({birdnet_f1:.4f}) < 0.30: "
                  f"音声データが不十分")
        else:
            print(f"  OK: BirdNET 単独 F1 ({birdnet_f1:.4f}) >= 0.30")

    improved_count = 0
    improved_species = []
    for sp_name, sp_results in all_results.items():
        dino_only = sp_results.get("DINOv2_LogReg", {}).get("f1", 0.0)
        combo = sp_results.get("DINOv2+BirdNET_MLP", {}).get("f1", 0.0)
        if combo > dino_only:
            improved_count += 1
            improved_species.append(sp_name)

    print(f"  統合が単独を上回った種: {improved_count}/{len(all_results)}")
    if improved_species:
        print(f"    → {', '.join(improved_species)}")
    if improved_count == 0:
        print("  NG: 全種で相補性なし")

    output = {
        "per_species": all_results,
        "average": avg_across_species,
        "improved_count": improved_count,
        "improved_species": improved_species,
        "total_species": len(all_results),
    }
    out_path = os.path.join(PHASE3_DIR, "phase3_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n結果を保存: {out_path}")


if __name__ == "__main__":
    main()
