"""近縁種統合: BirdNET の種レベル確信度を科レベルに集約する。

Phase 5g 統合の audio_results.json を再分析し、
BirdNET の種混同を吸収するため「科レベル」の合計確信度を計算する。
追加の推論は不要。既存の JSON のみ使用。
"""

import csv
import json
import os
import sys

import numpy as np


# ========================================
# 科レベルの種リスト
# ========================================
# BirdNET が出力する種名（"学名_英名" の英名側）から科への対応表
# 主要な科のみ定義。対応表にない種は科名 None に分類。
FAMILY_MAP = {
    # Paridae（シジュウカラ科）— 今回のメインターゲット
    "Paridae": [
        "Great Tit", "Parus major",
        "Japanese Tit", "Parus minor",
        "Blue Tit", "Cyanistes caeruleus", "Eurasian Blue Tit",
        "Coal Tit", "Periparus ater",
        "Marsh Tit", "Poecile palustris",
        "Willow Tit", "Poecile montanus",
        "Crested Tit", "Lophophanes cristatus",
        "Varied Tit", "Sittiparus varius",
        "Azure Tit", "Cyanistes cyanus",
        "Sombre Tit", "Poecile lugubris",
        "Ground Tit", "Pseudopodoces humilis",
    ],

    # 他の科（WetlandBirds や VB100 で役立つ可能性）
    "Corvidae": [
        "Eurasian Magpie", "Pica pica",
        "Eurasian Jay", "Garrulus glandarius",
        "Hooded Crow", "Corvus cornix",
        "Carrion Crow", "Corvus corone",
        "Common Raven", "Corvus corax",
        "Western Jackdaw", "Coloeus monedula",
        "Rook", "Corvus frugilegus",
    ],

    "Turdidae": [
        "Eurasian Blackbird", "Turdus merula",
        "Song Thrush", "Turdus philomelos",
        "Mistle Thrush", "Turdus viscivorus",
        "Redwing", "Turdus iliacus",
        "Fieldfare", "Turdus pilaris",
        "American Robin", "Turdus migratorius",
    ],

    "Fringillidae": [
        "Common Chaffinch", "Fringilla coelebs",
        "Brambling", "Fringilla montifringilla",
        "European Greenfinch", "Chloris chloris",
        "Eurasian Bullfinch", "Pyrrhula pyrrhula",
        "European Goldfinch", "Carduelis carduelis",
        "Common Linnet", "Linaria cannabina",
        "Eurasian Siskin", "Spinus spinus",
        "Hawfinch", "Coccothraustes coccothraustes",
    ],

    "Sylviidae": [
        "Eurasian Blackcap", "Sylvia atricapilla",
        "Garden Warbler", "Sylvia borin",
        "Common Whitethroat", "Curruca communis",
        "Lesser Whitethroat", "Curruca curruca",
    ],

    "Strigidae": [
        "Eurasian Pygmy-Owl", "Glaucidium passerinum",
        "Tawny Owl", "Strix aluco",
        "Long-eared Owl", "Asio otus",
        "Eurasian Eagle-Owl", "Bubo bubo",
        "Little Owl", "Athene noctua",
    ],

    "Anatidae": [
        "Mallard", "Anas platyrhynchos",
        "Eurasian Teal", "Anas crecca",
        "Gadwall", "Mareca strepera",
        "Eurasian Wigeon", "Mareca penelope",
        "Common Pochard", "Aythya ferina",
        "Tufted Duck", "Aythya fuligula",
    ],

    "Rallidae": [
        "Eurasian Coot", "Fulica atra",
        "Common Moorhen", "Gallinula chloropus",
        "Water Rail", "Rallus aquaticus",
        "Eurasian Moorhen",
    ],
}


def get_family(species_name):
    """種名から科名を返す（対応表にない場合は None）。"""
    name_lower = species_name.lower()
    for family, species_list in FAMILY_MAP.items():
        for sp in species_list:
            sp_lower = sp.lower()
            if sp_lower in name_lower or name_lower in sp_lower:
                return family
    return None


def compute_family_confidence(top_species, target_family):
    """BirdNET のトップ種リストから特定の科の合計確信度を計算する。

    返り値: (合計確信度, 検出された該当科の種一覧)
    """
    total = 0.0
    matched_species = []
    for sp in top_species:
        name = sp.get("species", "")
        conf = sp.get("confidence", 0.0)
        family = get_family(name)
        if family == target_family:
            total += conf
            matched_species.append({"species": name, "confidence": conf})
    return total, matched_species


def load_audio_results(path):
    """audio_results.json を読み、{video_id: {...}} の dict を返す。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "videos" in data:
        return data["videos"]
    return data


def main():
    # Windows コンソール対策
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("=== 近縁種統合: 科レベル確信度集約 ===")

    # データ読み込み
    audio_results = load_audio_results(
        "../results/phase5g_youtube/audio_results.json"
    )

    labels = {}
    with open("../data/youtube_greattit/labels.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["video_id"]] = row

    # ========================================
    # 分析1: 種レベル vs 科レベルの確信度比較
    # ========================================
    print(f"\n{'='*100}")
    print(f"分析1: 種レベル vs 科レベル（Paridae）の確信度比較")
    print(f"{'='*100}")

    print(f"\n{'動画ID':<14} {'ラベル':<10} {'GT種最大':>9} "
          f"{'Paridae科最大':>13} {'差分':>8}  検出された Paridae 種")
    print("-" * 110)

    species_results = {}
    family_results = {}

    for vid, ar in audio_results.items():
        human_label = ar.get("singing_matches_video", "?")
        windows = ar.get("windows", [])

        species_gt_confs = []
        family_paridae_confs = []
        all_paridae_species = set()

        for w in windows:
            # 種レベル: Great Tit のみ
            gt_conf = w.get("great_tit_confidence", 0.0)
            species_gt_confs.append(gt_conf)

            # 科レベル: Paridae 全体
            top_species = w.get("top_species", [])
            fam_conf, matched = compute_family_confidence(top_species, "Paridae")
            family_paridae_confs.append(fam_conf)

            for m in matched:
                all_paridae_species.add(m["species"].split("_", 1)[-1])

        species_max = max(species_gt_confs) if species_gt_confs else 0.0
        family_max = max(family_paridae_confs) if family_paridae_confs else 0.0
        species_mean = (np.mean(species_gt_confs)
                        if species_gt_confs else 0.0)
        family_mean = (np.mean(family_paridae_confs)
                       if family_paridae_confs else 0.0)

        diff = family_max - species_max
        paridae_list = (", ".join(sorted(all_paridae_species))
                        if all_paridae_species else "なし")

        print(f"{vid:<14} {human_label:<10} {species_max:>9.4f} "
              f"{family_max:>13.4f} {diff:>+8.4f}  {paridae_list}")

        species_results[vid] = {
            "human_label": human_label,
            "species_max": float(species_max),
            "species_mean": float(species_mean),
        }
        family_results[vid] = {
            "human_label": human_label,
            "family_max": float(family_max),
            "family_mean": float(family_mean),
            "paridae_species": sorted(all_paridae_species),
        }

    # ========================================
    # 分析2: ラベル別の統計比較
    # ========================================
    print(f"\n{'='*80}")
    print(f"分析2: ラベル別の統計（種レベル vs 科レベル）")
    print(f"{'='*80}")

    label_groups = {"yes": [], "partial": [], "other": [], "no_sound": []}

    for vid in species_results:
        label = species_results[vid]["human_label"]
        if label in label_groups:
            label_groups[label].append({
                "video_id": vid,
                "species_max": species_results[vid]["species_max"],
                "family_max": family_results[vid]["family_max"],
            })

    print(f"\n{'ラベル':<12} {'動画数':>6} {'種GT最大':>10} "
          f"{'科Paridae最大':>14} {'改善':>8}")
    print("-" * 55)
    for label in ["yes", "partial", "other", "no_sound"]:
        videos = label_groups.get(label, [])
        if not videos:
            continue
        sp_mean = np.mean([v["species_max"] for v in videos])
        fm_mean = np.mean([v["family_max"] for v in videos])
        diff = fm_mean - sp_mean
        print(f"{label:<12} {len(videos):>6} {sp_mean:>10.4f} "
              f"{fm_mean:>14.4f} {diff:>+8.4f}")

    # ========================================
    # 分析3: Precision/Recall の比較（閾値 0.3）
    # ========================================
    print(f"\n{'='*80}")
    print(f"分析3: Precision/Recall の比較（閾値 0.3）")
    print(f"{'='*80}")

    threshold = 0.3
    yes_videos = set(vid for vid, sr in species_results.items()
                     if sr["human_label"] == "yes")

    species_detected = set(vid for vid, sr in species_results.items()
                           if sr["species_max"] > threshold)
    species_tp = len(species_detected & yes_videos)
    species_fp = len(species_detected - yes_videos)
    species_prec = species_tp / max(species_tp + species_fp, 1)
    species_recall = species_tp / max(len(yes_videos), 1)

    family_detected = set(vid for vid, fr in family_results.items()
                          if fr["family_max"] > threshold)
    family_tp = len(family_detected & yes_videos)
    family_fp = len(family_detected - yes_videos)
    family_prec = family_tp / max(family_tp + family_fp, 1)
    family_recall = family_tp / max(len(yes_videos), 1)

    print(f"\n{'手法':<22} {'検出':>6} {'TP':>4} {'FP':>4} "
          f"{'Precision':>10} {'Recall':>8}")
    print("-" * 58)
    print(f"{'種 (Great Tit)':<22} {len(species_detected):>6} {species_tp:>4} "
          f"{species_fp:>4} {species_prec*100:>9.1f}% "
          f"{species_recall*100:>7.1f}%")
    print(f"{'科 (Paridae)':<22} {len(family_detected):>6} {family_tp:>4} "
          f"{family_fp:>4} {family_prec*100:>9.1f}% "
          f"{family_recall*100:>7.1f}%")

    print(f"\n改善:")
    print(f"  Precision: {species_prec*100:.1f}% → "
          f"{family_prec*100:.1f}% "
          f"({(family_prec-species_prec)*100:+.1f}pp)")
    print(f"  Recall:    {species_recall*100:.1f}% → "
          f"{family_recall*100:.1f}% "
          f"({(family_recall-species_recall)*100:+.1f}pp)")

    # ========================================
    # 分析4: 複数の閾値での Precision/Recall カーブ
    # ========================================
    print(f"\n{'='*80}")
    print(f"分析4: 閾値別 Precision/Recall（種 vs 科）")
    print(f"{'='*80}")

    print(f"\n{'閾値':>6} | {'種 Prec':>8} {'種 Rec':>8} | "
          f"{'科 Prec':>8} {'科 Rec':>8} | {'Rec改善':>8}")
    print("-" * 65)

    for th in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
               0.40, 0.50, 0.60, 0.70, 0.80]:
        sp_det = set(vid for vid, sr in species_results.items()
                      if sr["species_max"] > th)
        sp_tp = len(sp_det & yes_videos)
        sp_fp = len(sp_det - yes_videos)
        sp_p = sp_tp / max(sp_tp + sp_fp, 1)
        sp_r = sp_tp / max(len(yes_videos), 1)

        fm_det = set(vid for vid, fr in family_results.items()
                      if fr["family_max"] > th)
        fm_tp = len(fm_det & yes_videos)
        fm_fp = len(fm_det - yes_videos)
        fm_p = fm_tp / max(fm_tp + fm_fp, 1)
        fm_r = fm_tp / max(len(yes_videos), 1)

        rec_diff = fm_r - sp_r
        print(f"{th:>6.2f} | {sp_p*100:>7.1f}% {sp_r*100:>7.1f}% | "
              f"{fm_p*100:>7.1f}% {fm_r*100:>7.1f}% | "
              f"{rec_diff*100:>+7.1f}pp")

    # ========================================
    # 分析5: 取りこぼし動画の詳細
    # ========================================
    print(f"\n{'='*80}")
    print(f"分析5: 種レベルで取りこぼし → 科レベルで回収できた動画")
    print(f"{'='*80}")

    recovered = (family_detected & yes_videos) - (species_detected & yes_videos)
    still_missed = yes_videos - family_detected

    if recovered:
        print(f"\n回収された動画 ({len(recovered)} 本):")
        for vid in sorted(recovered):
            sr = species_results[vid]
            fr = family_results[vid]
            print(f"  {vid}: 種GT {sr['species_max']:.4f} → "
                  f"科Paridae {fr['family_max']:.4f}")
            print(f"    検出された Paridae 種: "
                  f"{', '.join(fr['paridae_species'])}")
    else:
        print(f"\n回収された動画: なし")

    if still_missed:
        print(f"\nまだ取りこぼしている動画 ({len(still_missed)} 本):")
        for vid in sorted(still_missed):
            sr = species_results[vid]
            fr = family_results[vid]
            ar = audio_results[vid]
            top_species_all = {}
            for w in ar.get("windows", []):
                for sp in w.get("top_species", [])[:3]:
                    name = sp["species"]
                    conf = sp["confidence"]
                    if (name not in top_species_all
                            or conf > top_species_all[name]):
                        top_species_all[name] = conf

            top5 = sorted(top_species_all.items(),
                          key=lambda x: x[1], reverse=True)[:5]
            print(f"  {vid}: 種GT {sr['species_max']:.4f}, "
                  f"科Paridae {fr['family_max']:.4f}")
            print(f"    BirdNET トップ5:")
            for name, conf in top5:
                family = get_family(name)
                fam_str = f" [{family}]" if family else ""
                disp = name.split("_", 1)[-1]
                print(f"      {disp}{fam_str}: {conf:.4f}")

    # ========================================
    # 結論
    # ========================================
    print(f"\n{'='*80}")
    print(f"結論")
    print(f"{'='*80}")

    print(f"""
    種レベル (Great Tit のみ):
      Precision {species_prec*100:.0f}%, Recall {species_recall*100:.0f}%

    科レベル (Paridae 全体):
      Precision {family_prec*100:.0f}%, Recall {family_recall*100:.0f}%

    改善: Recall {species_recall*100:.0f}% → {family_recall*100:.0f}% \
({(family_recall-species_recall)*100:+.0f}pp)
    回収された動画: {len(recovered)} 本
    まだ取りこぼし: {len(still_missed)} 本
    """)

    if family_recall > species_recall:
        print(f"  科レベル集約により Recall が改善されました。")
        print(f"  動的適応ルールを「Paridae 科の確信度 > 0.3」に更新すべきです。")
    elif family_recall == species_recall:
        print(f"  科レベル集約による Recall の改善はありませんでした。")
        print(f"  取りこぼしは Paridae 以外の種への混同が原因です。")

    # 保存
    output = {
        "species_level": {
            "threshold": threshold,
            "precision": float(species_prec),
            "recall": float(species_recall),
            "tp": species_tp,
            "fp": species_fp,
        },
        "family_level": {
            "threshold": threshold,
            "precision": float(family_prec),
            "recall": float(family_recall),
            "tp": family_tp,
            "fp": family_fp,
        },
        "recovered_videos": sorted(recovered),
        "still_missed_videos": sorted(still_missed),
        "per_video": {
            vid: {
                "human_label": species_results[vid]["human_label"],
                "species_max": species_results[vid]["species_max"],
                "family_max": family_results[vid]["family_max"],
                "paridae_species": family_results[vid]["paridae_species"],
            }
            for vid in species_results
        },
    }

    os.makedirs("../results/phase5g_youtube", exist_ok=True)
    out_path = "../results/phase5g_youtube/family_aggregation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n保存: {out_path}")


if __name__ == "__main__":
    main()
