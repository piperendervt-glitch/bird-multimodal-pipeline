"""Phase 1 段階C: MLP + ベースライン（k-NN, ロジスティック回帰）の学習・評価。

特徴ファイルを読み込み、3手法で学習・評価する。
このスクリプトは DINOv2 モデルをインポートしない（特徴量ファイルのみ使用）。
"""

import os
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score


RESULTS_DIR = os.path.join("..", "results", "bird_phase1")


class BirdMLP(nn.Module):
    """Phase 1: シンプルな MLP 分類器"""
    def __init__(self, input_dim, n_classes, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(X_train, y_train, X_test, y_test, n_classes,
              hidden_dim=128, epochs=200, lr=0.001, batch_size=64,
              patience=10):
    """MLP の学習と評価（10エポックごとに検証、早期停止付き）"""
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
            batch_X = X_tr[batch_idx]
            batch_y = y_tr[batch_idx]

            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        # 10エポックごとにテストセットで検証
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(X_te)
                preds = logits.argmax(dim=1).cpu().numpy()
                acc = accuracy_score(y_test, preds)

            print(f"  Epoch {epoch+1}: loss={total_loss/n_batches:.4f}, "
                  f"test_acc={acc*100:.2f}%")

            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                print(f"  早期停止 (epoch {epoch+1})")
                break

    # 最良モデルで最終評価
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(X_te).cpu().numpy()
        preds = logits.argmax(axis=1)
        probs = torch.softmax(torch.FloatTensor(logits), dim=1).numpy()

    return preds, probs, model


def main():
    print("=== 段階C: 鳥種分類 学習・評価 ===")

    # 特徴量読み込み
    feat_path = os.path.join(RESULTS_DIR, "features_dinov2_vits14.npz")
    data = np.load(feat_path)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    feat_dim = int(data["feat_dim"])

    with open(os.path.join(RESULTS_DIR, "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    n_classes = meta["n_classes"]

    print(f"特徴量次元: {feat_dim}")
    print(f"学習: {X_train.shape[0]} 枚, テスト: {X_test.shape[0]} 枚")
    print(f"クラス数: {n_classes}")

    results = {}

    # --- ベースライン1: k-NN (k=5, コサイン距離) ---
    print(f"\n--- ベースライン: k-NN (k=5, コサイン距離) ---")
    start = time.time()
    knn = KNeighborsClassifier(n_neighbors=5, metric="cosine", n_jobs=-1)
    knn.fit(X_train, y_train)
    knn_preds = knn.predict(X_test)
    knn_probs = knn.predict_proba(X_test)
    knn_time = time.time() - start

    knn_acc = accuracy_score(y_test, knn_preds)
    knn_f1 = f1_score(y_test, knn_preds, average="macro", zero_division=0)
    knn_top5 = top_k_accuracy_score(y_test, knn_probs, k=5, labels=list(range(n_classes)))

    results["kNN"] = {
        "top1_acc": float(knn_acc),
        "top5_acc": float(knn_top5),
        "macro_f1": float(knn_f1),
        "time_sec": float(knn_time),
    }
    print(f"  Top-1: {knn_acc*100:.2f}%, Top-5: {knn_top5*100:.2f}%, "
          f"F1: {knn_f1:.4f}, 時間: {knn_time:.1f}秒")

    # --- ベースライン2: ロジスティック回帰 ---
    print(f"\n--- ベースライン: ロジスティック回帰 ---")
    start = time.time()
    lr_model = LogisticRegression(max_iter=1000, C=1.0, random_state=42, n_jobs=-1)
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    lr_probs = lr_model.predict_proba(X_test)
    lr_time = time.time() - start

    lr_acc = accuracy_score(y_test, lr_preds)
    lr_f1 = f1_score(y_test, lr_preds, average="macro", zero_division=0)
    lr_top5 = top_k_accuracy_score(y_test, lr_probs, k=5, labels=list(range(n_classes)))

    results["LogisticRegression"] = {
        "top1_acc": float(lr_acc),
        "top5_acc": float(lr_top5),
        "macro_f1": float(lr_f1),
        "time_sec": float(lr_time),
    }
    print(f"  Top-1: {lr_acc*100:.2f}%, Top-5: {lr_top5*100:.2f}%, "
          f"F1: {lr_f1:.4f}, 時間: {lr_time:.1f}秒")

    # --- MLP ---
    print(f"\n--- MLP (hidden=128) ---")
    start = time.time()
    mlp_preds, mlp_probs, _ = train_mlp(
        X_train, y_train, X_test, y_test, n_classes,
        hidden_dim=128, epochs=200, lr=0.001,
    )
    mlp_time = time.time() - start

    mlp_acc = accuracy_score(y_test, mlp_preds)
    mlp_f1 = f1_score(y_test, mlp_preds, average="macro", zero_division=0)
    mlp_top5 = top_k_accuracy_score(y_test, mlp_probs, k=5, labels=list(range(n_classes)))

    results["MLP"] = {
        "top1_acc": float(mlp_acc),
        "top5_acc": float(mlp_top5),
        "macro_f1": float(mlp_f1),
        "time_sec": float(mlp_time),
    }
    print(f"  Top-1: {mlp_acc*100:.2f}%, Top-5: {mlp_top5*100:.2f}%, "
          f"F1: {mlp_f1:.4f}, 時間: {mlp_time:.1f}秒")

    # --- 結果サマリー ---
    print(f"\n{'='*60}")
    print(f"Phase 1 結果サマリー")
    print(f"{'='*60}")
    print(f"{'手法':<25} {'Top-1':>8} {'Top-5':>8} {'F1':>8}")
    print(f"{'-'*60}")
    for name, r in results.items():
        print(f"{name:<25} {r['top1_acc']*100:>7.2f}% "
              f"{r['top5_acc']*100:>7.2f}% {r['macro_f1']:>7.4f}")

    # --- 失敗条件の判定 ---
    print(f"\n--- 失敗条件の判定 ---")
    mlp_top1 = results["MLP"]["top1_acc"]
    lr_top1 = results["LogisticRegression"]["top1_acc"]

    if mlp_top1 < 0.30:
        print(f"  !! Top-1 = {mlp_top1*100:.2f}% < 30%: パイプラインに根本的な問題あり")
    else:
        print(f"  OK: Top-1 = {mlp_top1*100:.2f}% (>= 30%)")

    if mlp_top1 <= lr_top1:
        print(f"  !! MLP ({mlp_top1*100:.2f}%) <= LogReg ({lr_top1*100:.2f}%): "
              f"非線形統合の意味なし")
    else:
        print(f"  OK: MLP ({mlp_top1*100:.2f}%) > LogReg ({lr_top1*100:.2f}%): "
              f"+{(mlp_top1-lr_top1)*100:.2f}pp")

    # 結果保存
    results_path = os.path.join(RESULTS_DIR, "phase1_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n結果を保存: {results_path}")


if __name__ == "__main__":
    main()
