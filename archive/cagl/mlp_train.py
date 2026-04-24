"""
段階B: MLP と XGBoost で 5-fold 交差検証を実施。

追加の LLM 推論は一切行わない（段階A の特徴量のみ使用）。
MLP の勾配ベース重要度 / XGBoost の gain ベース重要度の両方を出力する。
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

import xgboost as xgb


try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
MLP_DIR = RESULTS_DIR / "mlp"

TORCH_SEED = 42
N_SPLITS = 5
MLP_HIDDEN = 64
MLP_EPOCHS = 200
MLP_LR = 1e-3


class IntegrationMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp_fold(X_train, y_train, X_test, input_dim, n_classes, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = IntegrationMLP(input_dim, MLP_HIDDEN, n_classes)
    optimizer = optim.Adam(model.parameters(), lr=MLP_LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    Xt = torch.from_numpy(X_train.astype(np.float32))
    yt = torch.from_numpy(y_train.astype(np.int64))

    model.train()
    for _ in range(MLP_EPOCHS):
        optimizer.zero_grad()
        out = model(Xt)
        loss = criterion(out, yt)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        Xe = torch.from_numpy(X_test.astype(np.float32))
        preds = model(Xe).argmax(dim=1).numpy()
    return preds, model


def mlp_gradient_importance(model, X):
    """勾配の絶対平均を特徴量重要度として使用。"""
    model.eval()
    X_t = torch.from_numpy(X.astype(np.float32))
    X_t.requires_grad_(True)
    out = model(X_t)
    out.abs().sum().backward()
    imp = X_t.grad.detach().abs().mean(dim=0).numpy()
    if imp.sum() > 0:
        imp = imp / imp.sum()
    return imp


def run_mlp(X, y, n_classes):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=TORCH_SEED)
    all_preds = np.zeros(len(y), dtype=np.int64)
    fold_accs = []
    last_model = None

    for fold, (tr, te) in enumerate(skf.split(X, y)):
        preds, model = train_mlp_fold(
            X[tr], y[tr], X[te], X.shape[1], n_classes, seed=TORCH_SEED + fold
        )
        all_preds[te] = preds
        acc = accuracy_score(y[te], preds)
        fold_accs.append(float(acc))
        print(f"  Fold {fold+1}: {acc*100:.2f}%")
        last_model = model  # 重要度は最終fold のモデルで計算

    overall = float(accuracy_score(y, all_preds))
    importance = mlp_gradient_importance(last_model, X)
    return all_preds, fold_accs, overall, importance


def run_xgb(X, y, n_classes):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=TORCH_SEED)
    all_preds = np.zeros(len(y), dtype=np.int64)
    fold_accs = []
    last_model = None

    for fold, (tr, te) in enumerate(skf.split(X, y)):
        clf = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            random_state=TORCH_SEED,
            eval_metric="mlogloss",
            num_class=n_classes,
            tree_method="hist",
        )
        clf.fit(X[tr], y[tr])
        preds = clf.predict(X[te])
        all_preds[te] = preds
        acc = accuracy_score(y[te], preds)
        fold_accs.append(float(acc))
        print(f"  Fold {fold+1}: {acc*100:.2f}%")
        last_model = clf

    overall = float(accuracy_score(y, all_preds))
    importance = last_model.feature_importances_.astype(np.float64)
    if importance.sum() > 0:
        importance = importance / importance.sum()
    return all_preds, fold_accs, overall, importance


def print_top_features(importance, feature_names, k=15):
    order = np.argsort(importance)[::-1]
    for i in order[:k]:
        print(f"  {feature_names[i]:<40} {importance[i]:.4f}")


def main():
    data = np.load(MLP_DIR / "dataset.npz", allow_pickle=True)
    X = data["X"]
    y = data["y"]
    n_classes = int(data["n_options"])
    model_names = [str(x) for x in data["model_names"]]
    feature_names = [str(x) for x in data["feature_names"]]
    cats = [str(c) for c in data["question_categories"]]

    print("=" * 60)
    print("MLP 統合層 実験")
    print("=" * 60)
    print(f"問題数: {len(y)}")
    print(f"特徴量数: {X.shape[1]}")
    print(f"クラス数: {n_classes}")
    print(f"モデル: {model_names}")

    print("\n--- MLP ---")
    mlp_preds, mlp_folds, mlp_acc, mlp_imp = run_mlp(X, y, n_classes)
    print(f"  全体正解率: {mlp_acc*100:.2f}%")

    print("\n--- XGBoost ---")
    xgb_preds, xgb_folds, xgb_acc, xgb_imp = run_xgb(X, y, n_classes)
    print(f"  全体正解率: {xgb_acc*100:.2f}%")

    print("\n--- MLP 特徴量重要度 (上位15, 勾配ベース) ---")
    print_top_features(mlp_imp, feature_names, 15)

    print("\n--- XGBoost 特徴量重要度 (上位15, gain ベース) ---")
    print_top_features(xgb_imp, feature_names, 15)

    out = {
        "評価日時": datetime.now(timezone.utc).isoformat(),
        "n_questions": int(len(y)),
        "n_features": int(X.shape[1]),
        "n_classes": n_classes,
        "MLP": {
            "fold_accs": mlp_folds,
            "overall_accuracy": mlp_acc,
            "feature_importance": mlp_imp.tolist(),
            "cv_preds": mlp_preds.tolist(),
        },
        "XGBoost": {
            "fold_accs": xgb_folds,
            "overall_accuracy": xgb_acc,
            "feature_importance": xgb_imp.tolist(),
            "cv_preds": xgb_preds.tolist(),
        },
        "feature_names": feature_names,
        "model_names": model_names,
        "categories_per_question": cats,
        "ground_truth": y.tolist(),
    }
    out_path = MLP_DIR / "train_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n結果を保存: {out_path}")


if __name__ == "__main__":
    main()
