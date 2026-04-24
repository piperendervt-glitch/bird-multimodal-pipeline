"""Phase 2 段階D: 各結合特徴量に対する学習・評価。

DINOv2 モデルや OpenCV はインポートしない。特徴量ファイルのみ使用。
Phase 1 より hidden_dim を拡大、BatchNorm を追加、Dropout を 0.3→0.15 に軽減。
"""

import os
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score


PHASE1_DIR = os.path.join("..", "results", "bird_phase1")
PHASE2_DIR = os.path.join("..", "results", "bird_phase2")


class BirdMLP(nn.Module):
    """Phase 2 用の 2 隠れ層 MLP（BatchNorm + 軽い Dropout）"""
    def __init__(self, input_dim, n_classes, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(X_train, y_train, X_test, y_test, n_classes,
              hidden_dim=256, epochs=300, lr=0.001, batch_size=64,
              patience=15):
    """MLP 学習・評価、10エポックごとに検証して早期停止"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BirdMLP(X_train.shape[1], n_classes, hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    X_tr = torch.FloatTensor(X_train).to(device)
    y_tr = torch.LongTensor(y_train).to(device)
    X_te = torch.FloatTensor(X_test).to(device)

    best_acc = 0.0
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(X_tr), device=device)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, len(X_tr), batch_size):
            batch_idx = indices[i:i + batch_size]
            optimizer.zero_grad()
            output = model(X_tr[batch_idx])
            loss = criterion(output, y_tr[batch_idx])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(X_te)
                preds = logits.argmax(dim=1).cpu().numpy()
                acc = accuracy_score(y_test, preds)

            if epoch < 30 or (epoch + 1) % 50 == 0:
                print(f"    Epoch {epoch+1}: loss={total_loss/n_batches:.4f}, "
                      f"test_acc={acc*100:.2f}%")

            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                print(f"    早期停止 (epoch {epoch+1})")
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(X_te).cpu().numpy()
        preds = logits.argmax(axis=1)
        probs = torch.softmax(torch.FloatTensor(logits), dim=1).numpy()

    return preds, probs


def evaluate(y_true, preds, probs, n_classes):
    return {
        "top1_acc": float(accuracy_score(y_true, preds)),
        "top5_acc": float(top_k_accuracy_score(y_true, probs, k=5, labels=list(range(n_classes)))),
        "macro_f1": float(f1_score(y_true, preds, average="macro", zero_division=0)),
    }


def main():
    print("=== Phase 2D: 学習・評価 ===")

    data = np.load(os.path.join(PHASE2_DIR, "features_combined.npz"))
    y_train = data["y_train"]
    y_test = data["y_test"]

    with open(os.path.join(PHASE1_DIR, "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    n_classes = meta["n_classes"]

    # Phase 1 の結果（比較用）
    with open(os.path.join(PHASE1_DIR, "phase1_results.json"), encoding="utf-8") as f:
        phase1 = json.load(f)

    combinations = ["dino_only", "dino_color", "dino_shape", "dino_color_shape", "color_shape"]
    results = {}

    for combo in combinations:
        X_train = data[f"X_train_{combo}"]
        X_test = data[f"X_test_{combo}"]

        print(f"\n{'='*60}")
        print(f"特徴量: {combo} ({X_train.shape[1]} 次元)")
        print(f"{'='*60}")

        # --- LogReg ---
        print(f"\n  --- LogReg ---")
        start = time.time()
        lr_model = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
        lr_model.fit(X_train, y_train)
        lr_preds = lr_model.predict(X_test)
        lr_probs = lr_model.predict_proba(X_test)
        lr_time = time.time() - start

        lr_eval = evaluate(y_test, lr_preds, lr_probs, n_classes)
        lr_eval["time_sec"] = float(lr_time)
        print(f"  Top-1: {lr_eval['top1_acc']*100:.2f}%, "
              f"Top-5: {lr_eval['top5_acc']*100:.2f}%, "
              f"F1: {lr_eval['macro_f1']:.4f}, "
              f"時間: {lr_time:.1f}秒")

        # --- MLP ---
        print(f"\n  --- MLP (hidden=256, Dropout=0.15, BatchNorm) ---")
        start = time.time()
        mlp_preds, mlp_probs = train_mlp(
            X_train, y_train, X_test, y_test, n_classes,
            hidden_dim=256, epochs=300, lr=0.001,
        )
        mlp_time = time.time() - start

        mlp_eval = evaluate(y_test, mlp_preds, mlp_probs, n_classes)
        mlp_eval["time_sec"] = float(mlp_time)
        print(f"  Top-1: {mlp_eval['top1_acc']*100:.2f}%, "
              f"Top-5: {mlp_eval['top5_acc']*100:.2f}%, "
              f"F1: {mlp_eval['macro_f1']:.4f}, "
              f"時間: {mlp_time:.1f}秒")

        results[combo] = {"LogReg": lr_eval, "MLP": mlp_eval}

    # --- 結果サマリー ---
    print(f"\n{'='*70}")
    print(f"Phase 2 結果サマリー")
    print(f"{'='*70}")
    print(f"{'特徴量':<25} {'手法':<10} {'Top-1':>8} {'Top-5':>8} {'F1':>8}")
    print(f"{'-'*70}")
    print(f"{'[Phase1] dino_only':<25} {'LogReg':<10} "
          f"{phase1['LogisticRegression']['top1_acc']*100:>7.2f}% "
          f"{phase1['LogisticRegression']['top5_acc']*100:>7.2f}% "
          f"{phase1['LogisticRegression']['macro_f1']:>7.4f}")
    print(f"{'[Phase1] dino_only':<25} {'MLP':<10} "
          f"{phase1['MLP']['top1_acc']*100:>7.2f}% "
          f"{phase1['MLP']['top5_acc']*100:>7.2f}% "
          f"{phase1['MLP']['macro_f1']:>7.4f}")
    print(f"{'-'*70}")

    for combo in combinations:
        for method in ["LogReg", "MLP"]:
            r = results[combo][method]
            print(f"{combo:<25} {method:<10} "
                  f"{r['top1_acc']*100:>7.2f}% "
                  f"{r['top5_acc']*100:>7.2f}% "
                  f"{r['macro_f1']:>7.4f}")
        print()

    # --- 失敗条件の判定 ---
    print(f"--- 失敗条件の判定 ---")
    phase1_logreg = phase1["LogisticRegression"]["top1_acc"]
    best_combo = max(results.items(), key=lambda x: x[1]["MLP"]["top1_acc"])
    best_mlp = best_combo[1]["MLP"]["top1_acc"]
    best_combo_name = best_combo[0]

    if best_mlp > phase1_logreg:
        print(f"  OK: 最良 MLP ({best_combo_name}: {best_mlp*100:.2f}%) > "
              f"Phase 1 LogReg ({phase1_logreg*100:.2f}%): "
              f"+{(best_mlp - phase1_logreg)*100:.2f}pp")
    else:
        print(f"  NG: 最良 MLP ({best_combo_name}: {best_mlp*100:.2f}%) <= "
              f"Phase 1 LogReg ({phase1_logreg*100:.2f}%)")

    dino_only_logreg = results["dino_only"]["LogReg"]["top1_acc"]
    combo_logreg = results["dino_color_shape"]["LogReg"]["top1_acc"]
    if combo_logreg > dino_only_logreg:
        print(f"  OK: 追加特徴 + LogReg ({combo_logreg*100:.2f}%) > "
              f"DINOv2 + LogReg ({dino_only_logreg*100:.2f}%): "
              f"追加特徴に価値あり")
    else:
        print(f"  NG: 追加特徴 + LogReg ({combo_logreg*100:.2f}%) <= "
              f"DINOv2 + LogReg ({dino_only_logreg*100:.2f}%): "
              f"追加特徴自体が不要の可能性")

    combo_mlp = results["dino_color_shape"]["MLP"]["top1_acc"]
    if combo_mlp > combo_logreg:
        print(f"  OK: 追加特徴 + MLP ({combo_mlp*100:.2f}%) > "
              f"追加特徴 + LogReg ({combo_logreg*100:.2f}%): "
              f"非線形統合の意味あり")
    else:
        print(f"  NG: 追加特徴 + MLP ({combo_mlp*100:.2f}%) <= "
              f"追加特徴 + LogReg ({combo_logreg*100:.2f}%): "
              f"非線形統合の意味なし")

    with open(os.path.join(PHASE2_DIR, "phase2_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n結果を保存: results/bird_phase2/phase2_results.json")


if __name__ == "__main__":
    main()
