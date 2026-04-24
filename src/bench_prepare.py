"""
段階A: 外部ベンチマークデータの準備と分類器予測の固定（多クラス版）。

標準的な分類器（全てデフォルトパラメータ、ハンディキャップなし）が自然に
苦戦する多クラスデータセットを使用する。正解ラベルを one-vs-rest の2値
行列に変換して CAGL に渡せる形式で保存する。

このスクリプトは core_cagl を一切インポートしない。
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
BENCH_DATA_DIR = RESULTS_DIR / "benchmark_data"

N_TRIALS = 20
TEST_SIZE = 0.5
SPLIT_RANDOM_STATE = 42

# 分類器の短縮表示名
CLASSIFIER_SHORT = {
    "DecisionTree": "DT",
    "SVM_rbf": "SVM",
    "KNN_5": "KNN",
    "LogisticRegression": "LR",
    "NaiveBayes": "NB",
    "RandomForest": "RF",
}


def build_datasets():
    """3つの多クラスデータセットを構築。"""
    dg = load_digits()
    wn = load_wine()

    # 1. 手書き数字 10クラス
    ds1 = {"X": dg.data.copy(), "y": dg.target.copy(), "n_classes": 10}

    # 2. ワイン 3クラス
    ds2 = {"X": wn.data.copy(), "y": wn.target.copy(), "n_classes": 3}

    # 3. 手書き数字の混同しやすい4クラスのみ（3, 5, 8, 9）
    mask = np.isin(dg.target, [3, 5, 8, 9])
    X_hard = dg.data[mask]
    y_hard = dg.target[mask]
    label_map = {3: 0, 5: 1, 8: 2, 9: 3}
    y_hard = np.array([label_map[v] for v in y_hard])
    ds3 = {"X": X_hard, "y": y_hard, "n_classes": 4}

    return {
        "digits_10class": ds1,
        "wine_3class":    ds2,
        "digits_hard4":   ds3,
    }


def build_classifiers():
    """6つの分類器を毎試行新しく生成（全てデフォルトパラメータ）。"""
    return {
        "DecisionTree":       DecisionTreeClassifier(random_state=42),
        "SVM_rbf":            SVC(kernel="rbf", random_state=42),
        "KNN_5":              KNeighborsClassifier(n_neighbors=5),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "NaiveBayes":         GaussianNB(),
        "RandomForest":       RandomForestClassifier(
            n_estimators=100, random_state=42
        ),
    }


def to_ovr_binary(y, n_classes):
    """正解ラベルを (サンプル数, クラス数) の one-vs-rest 2値行列に変換。"""
    binary = np.zeros((len(y), n_classes), dtype=np.uint8)
    for i, label in enumerate(y):
        binary[i, label] = 1
    return binary


def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def main():
    BENCH_DATA_DIR.mkdir(parents=True, exist_ok=True)
    datasets = build_datasets()
    classifier_names = list(build_classifiers().keys())

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_trials": N_TRIALS,
        "test_size": TEST_SIZE,
        "split_random_state": SPLIT_RANDOM_STATE,
        "classifiers": classifier_names,
        "classifier_short": CLASSIFIER_SHORT,
        "classifier_params_note": (
            "全分類器はデフォルトパラメータ。random_state と "
            "LogisticRegression.max_iter=1000 以外は指定なし（ハンディキャップなし）。"
        ),
        "preprocessing": "StandardScaler (fit on train, apply to test)",
        "multiclass_format": "one-vs-rest binary labels per class",
        "datasets": {},
    }

    for ds_name, ds_info in datasets.items():
        X, y = ds_info["X"], ds_info["y"]
        n_classes = ds_info["n_classes"]
        print(f"[{ds_name}] 総サンプル数={len(X)}, "
              f"特徴量次元={X.shape[1]}, クラス数={n_classes}, "
              f"クラス分布={np.bincount(y).tolist()}")

        ds_dir = BENCH_DATA_DIR / ds_name
        ds_dir.mkdir(parents=True, exist_ok=True)

        splitter = StratifiedShuffleSplit(
            n_splits=N_TRIALS, test_size=TEST_SIZE,
            random_state=SPLIT_RANDOM_STATE,
        )

        per_trial_acc = []
        files = {}
        n_test_samples = None

        for trial_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # StandardScaler (train で fit, test に適用)
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            n_test_samples = len(X_test)

            classifiers = build_classifiers()
            pred_list = []  # 各要素: (n_test, n_classes) の OVR 2値行列
            acc_dict = {}   # 分類器別の overall / per_class 正解率

            for clf_name, clf in classifiers.items():
                clf.fit(X_train_s, y_train)
                y_pred = clf.predict(X_test_s)

                overall = float((y_pred == y_test).mean())
                per_class = []
                for c in range(n_classes):
                    m = (y_test == c)
                    if m.any():
                        per_class.append(float((y_pred[m] == c).mean()))
                    else:
                        per_class.append(float("nan"))
                acc_dict[clf_name] = {
                    "overall": overall,
                    "per_class": per_class,
                }

                pred_list.append(to_ovr_binary(y_pred, n_classes))

            per_trial_acc.append(acc_dict)

            # pred 形状: (n_test, n_classifiers=6, n_classes)
            pred_arr = np.stack(pred_list, axis=1)
            # gt 形状: (n_test, n_classes)
            gt_arr = to_ovr_binary(y_test, n_classes)

            fname = f"trial_{trial_idx:03d}.npz"
            fpath = ds_dir / fname
            np.savez_compressed(
                fpath,
                pred=pred_arr,
                gt=gt_arr,
                classifiers=np.array(classifier_names),
                n_classes=np.int64(n_classes),
            )
            files[fname] = sha256_of_file(fpath)

        # 分類器別の平均
        mean_overall = {
            name: float(np.mean([t[name]["overall"] for t in per_trial_acc]))
            for name in classifier_names
        }
        mean_per_class = {
            name: [
                float(np.nanmean([t[name]["per_class"][c]
                                  for t in per_trial_acc]))
                for c in range(n_classes)
            ]
            for name in classifier_names
        }

        manifest["datasets"][ds_name] = {
            "n_total": int(len(X)),
            "n_test_samples": int(n_test_samples),
            "n_features": int(X.shape[1]),
            "n_classes": int(n_classes),
            "class_distribution": np.bincount(y).tolist(),
            "per_trial_classifier_accuracy": per_trial_acc,
            "mean_classifier_accuracy": mean_overall,
            "mean_classifier_per_class_accuracy": mean_per_class,
            "files": files,
        }

        acc_str = ", ".join(
            f"{CLASSIFIER_SHORT[k]}={v:.3f}" for k, v in mean_overall.items()
        )
        print(f"  試行数={N_TRIALS}, テストサンプル数={n_test_samples}, "
              f"平均全体正解率: {acc_str}")

    manifest_path = BENCH_DATA_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    total_files = sum(
        len(d["files"]) for d in manifest["datasets"].values()
    )
    print()
    print(f"マニフェスト出力: {manifest_path}")
    print(f"総試行ファイル数: {total_files}")


if __name__ == "__main__":
    main()
