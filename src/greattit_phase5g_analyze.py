"""Phase 5g 段階B: ダウンロードしたデータの構造を分析する。

torch, sklearn はインポートしない。pandas と numpy のみ使用。
"""

import os
import sys
import json

import pandas as pd
import numpy as np


try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main():
    print("=== Phase 5g-B: データ構造の確認 ===")

    DATA_DIR = "../data/great-tit-hits"

    # great-tit-hits.csv の読み込み
    print("\n--- great-tit-hits.csv ---")
    gth_path = os.path.join(DATA_DIR, "great-tit-hits.csv")
    if not os.path.exists(gth_path):
        print(f"  [NG]great-tit-hits.csv が見つかりません: {gth_path}")
        return
    gth = pd.read_csv(gth_path)
    print(f"行数: {len(gth)}")
    print(f"列数: {len(gth.columns)}")
    print(f"列名: {gth.columns.tolist()}")
    print(f"\n先頭5行:")
    print(gth.head())

    # class_id（ソングタイプ）の分析
    print(f"\n--- ソングタイプ (class_id) ---")
    n_classes = gth["class_id"].nunique()
    print(f"ソングタイプ数: {n_classes}")
    class_counts = gth["class_id"].value_counts()
    print(f"最頻タイプ: {class_counts.index[0]} ({class_counts.iloc[0]} 曲)")
    print(f"最少タイプ: {class_counts.index[-1]} ({class_counts.iloc[-1]} 曲)")
    print(f"タイプあたり平均: {class_counts.mean():.1f} 曲")
    print(f"タイプあたり中央値: {class_counts.median():.1f} 曲")

    print(f"\n上位20ソングタイプ:")
    for cls, cnt in class_counts.head(20).items():
        print(f"  {cls}: {cnt} 曲")

    # ID（個体）の分析
    print(f"\n--- 個体 (ID) ---")
    n_individuals = gth["ID"].nunique()
    print(f"個体数: {n_individuals}")
    id_counts = gth["ID"].value_counts()
    print(f"最多曲数の個体: {id_counts.index[0]} ({id_counts.iloc[0]} 曲)")
    print(f"最少曲数の個体: {id_counts.index[-1]} ({id_counts.iloc[-1]} 曲)")
    print(f"個体あたり平均: {id_counts.mean():.1f} 曲")

    # 個体とソングタイプの関係（レパートリー）
    print(f"\n--- 個体 × ソングタイプ ---")
    repertoire = gth.groupby("ID")["class_id"].nunique()
    print(f"レパートリーサイズ（1個体が歌うタイプ数）:")
    print(f"  平均: {repertoire.mean():.1f}")
    print(f"  最小: {repertoire.min()}")
    print(f"  最大: {repertoire.max()}")
    print(f"  中央値: {repertoire.median():.1f}")

    # 曲の長さの分布
    if "length_s" in gth.columns:
        print(f"\n--- 曲の長さ ---")
        print(f"  平均: {gth['length_s'].mean():.2f} 秒")
        print(f"  最短: {gth['length_s'].min():.2f} 秒")
        print(f"  最長: {gth['length_s'].max():.2f} 秒")
        print(f"  中央値: {gth['length_s'].median():.2f} 秒")

    # feature_vectors.csv の読み込み
    print(f"\n--- feature_vectors.csv ---")
    fv_path = os.path.join(DATA_DIR, "feature_vectors.csv")
    if os.path.exists(fv_path):
        fv_size = os.path.getsize(fv_path) / 1024 / 1024
        print(f"ファイルサイズ: {fv_size:.1f} MB")

        # 先頭5行で構造確認
        fv_head = pd.read_csv(fv_path, nrows=5)
        print(f"列数: {len(fv_head.columns)}")
        print(f"先頭5行 × 先頭5列:")
        print(fv_head.iloc[:, :5])

        # 全体の行数を確認（高速）
        with open(fv_path, encoding="utf-8") as f:
            n_rows = sum(1 for _ in f) - 1  # ヘッダー除く
        print(f"行数: {n_rows}")

        if n_rows == len(gth):
            print(f"  [OK] great-tit-hits.csv と行数が一致 ({n_rows})")
        else:
            print(f"  [NG]行数不一致: feature_vectors={n_rows}, great-tit-hits={len(gth)}")
    else:
        print(f"  [NG]feature_vectors.csv が見つかりません")

    # main.csv の読み込み（個体メタデータ）
    print(f"\n--- main.csv ---")
    main_path = os.path.join(DATA_DIR, "main.csv")
    if os.path.exists(main_path):
        main_df = pd.read_csv(main_path)
        print(f"行数: {len(main_df)}")
        print(f"列名: {main_df.columns.tolist()}")
        print(f"\n先頭3行:")
        print(main_df.head(3))
    else:
        print(f"  [NG]main.csv が見つかりません")

    # サマリー表示
    print(f"\n{'='*60}")
    print(f"Phase 5g データサマリー")
    print(f"{'='*60}")
    print(f"曲数:           {len(gth)}")
    print(f"ソングタイプ数: {n_classes}")
    print(f"個体数:         {n_individuals}")
    print(f"特徴量次元:     384")

    # 学習に使うタスクの規模
    print(f"\nタスク1: ソングタイプ分類")
    print(f"  クラス数: {n_classes}")
    print(f"  1クラスあたり平均: {class_counts.mean():.0f} サンプル")

    min_samples = 50
    valid_classes = class_counts[class_counts >= min_samples]
    print(f"  {min_samples}サンプル以上のクラス: {len(valid_classes)} / {n_classes}")
    valid_samples = gth[gth["class_id"].isin(valid_classes.index)]
    print(f"  該当サンプル数: {len(valid_samples)}")

    print(f"\nタスク2: 個体識別")
    print(f"  クラス数: {n_individuals}")
    print(f"  1個体あたり平均: {id_counts.mean():.0f} サンプル")
    valid_individuals = id_counts[id_counts >= min_samples]
    print(f"  {min_samples}サンプル以上の個体: {len(valid_individuals)} / {n_individuals}")

    # 結果保存
    os.makedirs("../results/great_tit_phase5g", exist_ok=True)
    summary = {
        "n_songs": int(len(gth)),
        "n_song_types": int(n_classes),
        "n_individuals": int(n_individuals),
        "feature_dim": 384,
        "class_counts_top20": {
            str(k): int(v) for k, v in class_counts.head(20).items()
        },
        "individual_counts_top20": {
            str(k): int(v) for k, v in id_counts.head(20).items()
        },
        "repertoire_stats": {
            "mean": float(repertoire.mean()),
            "min": int(repertoire.min()),
            "max": int(repertoire.max()),
            "median": float(repertoire.median()),
        },
        "valid_classes_min50": int(len(valid_classes)),
        "valid_individuals_min50": int(len(valid_individuals)),
    }
    out_path = "../results/great_tit_phase5g/data_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n保存: {out_path}")


if __name__ == "__main__":
    main()
