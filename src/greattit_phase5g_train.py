"""Phase 5g 段階C: ソングタイプ分類 + 個体識別の学習・評価。

事前計算済み 384 次元特徴ベクトルを使って、
- タスク1: ソングタイプ分類（class_id）
- タスク2: 個体識別（ID）
の 2 タスクを LogReg と k-NN で評価する。

DINOv2, BirdNET モデルはインポートしない。CSV ファイルのみ使用。
"""

import os
import sys
import json
import time

import numpy as np
import pandas as pd


try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score


DATA_DIR = "../data/great-tit-hits"
RESULTS_DIR = "../results/great_tit_phase5g"


def load_data(data_dir):
    """メタデータと特徴量を読み込む"""
    print("データ読み込み中...")

    gth = pd.read_csv(os.path.join(data_dir, "great-tit-hits.csv"))
    print(f"  great-tit-hits.csv: {len(gth)} 行")

    print("特徴ベクトル読み込み中（大きいファイル、時間がかかります）...")
    start = time.time()
    fv = pd.read_csv(os.path.join(data_dir, "feature_vectors.csv"))
    elapsed = time.time() - start
    print(f"  読み込み完了: {elapsed:.1f} 秒, 形状={fv.shape}")

    X = fv.values.astype(np.float32)
    print(f"特徴量: {X.shape}")

    # 行数の整合性チェック
    if len(X) != len(gth):
        print(f"  警告: 特徴量行数 ({len(X)}) と メタデータ行数 ({len(gth)}) が一致しません")

    return gth, X, elapsed


def run_classification(X, y, task_name, n_classes, test_size=0.2):
    """分類タスクの学習・評価（LogReg と k-NN）"""
    print(f"\n{'='*60}")
    print(f"タスク: {task_name}")
    print(f"{'='*60}")
    print(f"サンプル数: {len(y)}, クラス数: {n_classes}")

    # 学習/テスト分割（層化）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"学習: {len(X_train)}, テスト: {len(X_test)}")

    # 標準化
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {}

    # LogReg
    # 注: solver を "saga" から "lbfgs" に変更。
    #     SAGA は 725 クラス × 87k サンプルで 2 時間以上経っても未収束のため。
    #     lbfgs は L2 正則化に特化し、このタスク規模では同等の解を高速に出す。
    #     n_jobs は sklearn 1.8+ で非推奨のため指定しない。
    print(f"\n--- LogReg (solver=lbfgs) ---")
    start = time.time()
    lr = LogisticRegression(
        max_iter=2000, C=1.0, random_state=42,
        solver="lbfgs"
    )
    lr.fit(X_train_s, y_train)
    lr_preds = lr.predict(X_test_s)
    lr_probs = lr.predict_proba(X_test_s)
    lr_time = time.time() - start

    lr_acc = accuracy_score(y_test, lr_preds)
    lr_f1 = f1_score(y_test, lr_preds, average="macro", zero_division=0)

    if n_classes >= 5:
        lr_top5 = top_k_accuracy_score(
            y_test, lr_probs, k=5, labels=range(n_classes)
        )
    else:
        lr_top5 = lr_acc

    results["LogReg"] = {
        "accuracy": float(lr_acc),
        "top5": float(lr_top5),
        "macro_f1": float(lr_f1),
        "time_sec": float(lr_time),
    }
    print(f"  Top-1: {lr_acc*100:.2f}%, Top-5: {lr_top5*100:.2f}%, "
          f"F1: {lr_f1:.4f}, 時間: {lr_time:.1f}秒")

    # k-NN
    print(f"\n--- k-NN (k=5) ---")
    start = time.time()
    knn = KNeighborsClassifier(n_neighbors=5, metric="cosine", n_jobs=-1)
    # 注: k-NN は n_jobs=-1 が依然として有効（predict 時の並列化）
    knn.fit(X_train_s, y_train)
    knn_preds = knn.predict(X_test_s)
    knn_probs = knn.predict_proba(X_test_s)
    knn_time = time.time() - start

    knn_acc = accuracy_score(y_test, knn_preds)
    knn_f1 = f1_score(y_test, knn_preds, average="macro", zero_division=0)

    if n_classes >= 5:
        knn_top5 = top_k_accuracy_score(
            y_test, knn_probs, k=5, labels=range(n_classes)
        )
    else:
        knn_top5 = knn_acc

    results["kNN"] = {
        "accuracy": float(knn_acc),
        "top5": float(knn_top5),
        "macro_f1": float(knn_f1),
        "time_sec": float(knn_time),
    }
    print(f"  Top-1: {knn_acc*100:.2f}%, Top-5: {knn_top5*100:.2f}%, "
          f"F1: {knn_f1:.4f}, 時間: {knn_time:.1f}秒")

    return results


def main():
    print("=== Phase 5g-C: ソングタイプ分類 + 個体識別 ===")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    gth, X, load_elapsed = load_data(DATA_DIR)

    all_results = {
        "feature_load_time_sec": float(load_elapsed),
    }

    # ========================================
    # タスク1: ソングタイプ分類
    # ========================================
    min_samples = 50
    class_counts = gth["class_id"].value_counts()
    valid_classes = class_counts[class_counts >= min_samples].index

    mask = gth["class_id"].isin(valid_classes).values
    X_songtype = X[mask]

    le_song = LabelEncoder()
    y_songtype = le_song.fit_transform(gth.loc[mask, "class_id"])
    n_song_classes = len(le_song.classes_)

    print(f"\nソングタイプ分類: {min_samples}サンプル以上のクラスに限定")
    print(f"  使用サンプル: {len(y_songtype)} / {len(gth)}")
    print(f"  クラス数: {n_song_classes}")

    results_song = run_classification(
        X_songtype, y_songtype,
        f"ソングタイプ分類 ({n_song_classes} クラス)",
        n_song_classes,
    )
    all_results["songtype_classification"] = {
        "n_classes": int(n_song_classes),
        "n_samples": int(len(y_songtype)),
        "min_samples_per_class": int(min_samples),
        "results": results_song,
    }

    # ========================================
    # タスク2: 個体識別
    # ========================================
    id_counts = gth["ID"].value_counts()
    valid_ids = id_counts[id_counts >= min_samples].index

    mask_id = gth["ID"].isin(valid_ids).values
    X_individual = X[mask_id]

    le_id = LabelEncoder()
    y_individual = le_id.fit_transform(gth.loc[mask_id, "ID"])
    n_individuals = len(le_id.classes_)

    print(f"\n個体識別: {min_samples}サンプル以上の個体に限定")
    print(f"  使用サンプル: {len(y_individual)} / {len(gth)}")
    print(f"  個体数: {n_individuals}")

    results_id = run_classification(
        X_individual, y_individual,
        f"個体識別 ({n_individuals} 個体)",
        n_individuals,
    )
    all_results["individual_identification"] = {
        "n_classes": int(n_individuals),
        "n_samples": int(len(y_individual)),
        "min_samples_per_class": int(min_samples),
        "results": results_id,
    }

    # ========================================
    # サマリー
    # ========================================
    print(f"\n{'='*70}")
    print(f"Phase 5g 結果サマリー")
    print(f"{'='*70}")

    print(f"\n{'タスク':<35} {'手法':<10} {'Top-1':>8} {'Top-5':>8} {'F1':>8}")
    print(f"{'-'*70}")

    for task_name, task_data in all_results.items():
        if not isinstance(task_data, dict) or "results" not in task_data:
            continue
        n_cls = task_data["n_classes"]
        for method_name, r in task_data["results"].items():
            label = f"{task_name} ({n_cls}cls)"
            print(f"{label:<35} {method_name:<10} "
                  f"{r['accuracy']*100:>7.2f}% "
                  f"{r['top5']*100:>7.2f}% "
                  f"{r['macro_f1']:>7.4f}")

    # 失敗条件の判定
    print(f"\n--- 失敗条件の判定 ---")

    song_best = max(results_song.values(), key=lambda x: x["macro_f1"])
    if song_best["macro_f1"] < 0.5:
        print(f"  NG: ソングタイプ分類 F1 ({song_best['macro_f1']:.4f}) < 0.5")
    else:
        print(f"  OK: ソングタイプ分類 F1 ({song_best['macro_f1']:.4f}) >= 0.5")

    id_best = max(results_id.values(), key=lambda x: x["accuracy"])
    chance = 1.0 / n_individuals
    if id_best["accuracy"] < chance * 10:
        print(f"  NG: 個体識別精度 ({id_best['accuracy']*100:.2f}%) < "
              f"チャンスレートの10倍 ({chance*1000:.2f}%)")
    else:
        print(f"  OK: 個体識別精度 ({id_best['accuracy']*100:.2f}%) >> "
              f"チャンスレート ({chance*100:.2f}%)")

    all_results["judgment"] = {
        "songtype_f1_best": float(song_best["macro_f1"]),
        "songtype_pass": bool(song_best["macro_f1"] >= 0.5),
        "individual_accuracy_best": float(id_best["accuracy"]),
        "individual_chance_rate": float(chance),
        "individual_pass": bool(id_best["accuracy"] >= chance * 10),
    }

    # Phase 1 との比較
    print(f"\n--- Phase 1 (CUB-200 画像分類) との構造比較 ---")
    print(f"  Phase 1: 11,788画像 × 384次元 → 200種分類 → LogReg 87.31%")
    print(f"  Phase 5g ソングタイプ: "
          f"{all_results['songtype_classification']['n_samples']}曲 "
          f"× 384次元 → {n_song_classes}タイプ分類 → "
          f"LogReg {results_song['LogReg']['accuracy']*100:.2f}%")
    print(f"  Phase 5g 個体識別: "
          f"{all_results['individual_identification']['n_samples']}曲 "
          f"× 384次元 → {n_individuals}個体識別 → "
          f"LogReg {results_id['LogReg']['accuracy']*100:.2f}%")

    # 結果保存
    out_path = os.path.join(RESULTS_DIR, "phase5g_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n保存: {out_path}")


if __name__ == "__main__":
    main()
