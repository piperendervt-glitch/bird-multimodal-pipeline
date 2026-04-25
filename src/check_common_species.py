"""CUB-200 / VB100 / WetlandBirds の共通種を確認する。

クロスデータセット評価の可能性を探るため、各データセットの種名を
正規化して完全一致・部分一致を抽出する。
"""

import csv
import json
import os
import sys


def get_cub200_species():
    """CUB-200 の種名リストを取得する。"""
    cub_dir = "../data/cub200"
    species = set()

    # classes.txt がある場合
    classes_file = os.path.join(cub_dir, "CUB_200_2011", "classes.txt")
    if os.path.exists(classes_file):
        with open(classes_file, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    name = " ".join(parts[1:])
                    # "001.Black_footed_Albatross" → "Black_footed_Albatross"
                    name = name.split(".", 1)[-1] if "." in name else name
                    species.add(name)

    # 画像フォルダから取得（フォールバック）
    if not species:
        images_dir = os.path.join(cub_dir, "CUB_200_2011", "images")
        if os.path.exists(images_dir):
            for d in os.listdir(images_dir):
                name = d.split(".", 1)[-1] if "." in d else d
                species.add(name)

    return species


def get_vb100_species():
    """VB100 の種名リストを取得する。"""
    vb_dir = "../data/vb100/vb100_video"
    species = set()
    if os.path.exists(vb_dir):
        for d in os.listdir(vb_dir):
            if os.path.isdir(os.path.join(vb_dir, d)):
                species.add(d)
    return species


def get_wetlandbirds_species():
    """WetlandBirds の種名リストを取得する。"""
    species = set()

    # 1) species_mapping.json を優先
    mapping_path = "../results/bird_phase4b/species_mapping.json"
    if os.path.exists(mapping_path):
        try:
            with open(mapping_path, encoding="utf-8") as f:
                mapping = json.load(f)
            s_to_id = mapping.get("species_to_id") or mapping.get("species") or {}
            if isinstance(s_to_id, dict):
                species = set(s_to_id.keys())
            elif isinstance(s_to_id, list):
                species = set(s_to_id)
        except Exception:
            pass

    # 2) species_ID.csv をフォールバック
    if not species:
        csv_path = "../data/wetlandbirds/species_ID.csv"
        if os.path.exists(csv_path):
            with open(csv_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row.get("species") or row.get("name")
                    if name:
                        species.add(name.strip())

    return species


def normalize_name(name):
    """種名を正規化して比較可能にする。"""
    return name.lower().replace("_", " ").replace("-", " ").strip()


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("=== CUB-200 と VB100 の共通種確認 ===")

    cub_species = get_cub200_species()
    vb_species = get_vb100_species()

    print(f"\nCUB-200: {len(cub_species)} 種")
    print(f"VB100:   {len(vb_species)} 種")

    # 正規化した名前で比較
    cub_normalized = {normalize_name(s): s for s in cub_species}
    vb_normalized = {normalize_name(s): s for s in vb_species}

    # 完全一致
    common_exact = set(cub_normalized.keys()) & set(vb_normalized.keys())

    print(f"\n完全一致: {len(common_exact)} 種")
    if common_exact:
        print(f"\n{'CUB-200 名':<40} {'VB100 名':<40}")
        print("-" * 80)
        for name in sorted(common_exact):
            print(f"{cub_normalized[name]:<40} {vb_normalized[name]:<40}")

    # 部分一致（完全一致しなかった種で類似名を検索）
    cub_remaining = set(cub_normalized.keys()) - common_exact
    vb_remaining = set(vb_normalized.keys()) - common_exact

    print(f"\n--- 部分一致の候補 ---")
    partial_matches = []
    for cub_name in sorted(cub_remaining):
        for vb_name in sorted(vb_remaining):
            cub_words = set(cub_name.split())
            vb_words = set(vb_name.split())
            common_words = cub_words & vb_words
            if (common_words
                    and len(common_words) / max(len(cub_words),
                                                  len(vb_words)) >= 0.5):
                partial_matches.append(
                    (cub_normalized[cub_name],
                     vb_normalized[vb_name],
                     common_words)
                )

    if partial_matches:
        print(f"\n{'CUB-200 名':<40} {'VB100 名':<40} {'共通語'}")
        print("-" * 100)
        for cub_orig, vb_orig, words in partial_matches:
            print(f"{cub_orig:<40} {vb_orig:<40} {words}")
    else:
        print("部分一致なし")

    # CUB-200 のみの種
    matched_cub = {m[0] for m in partial_matches}
    matched_vb = {m[1] for m in partial_matches}
    cub_only = sorted(cub_normalized[n] for n in cub_remaining
                      if cub_normalized[n] not in matched_cub)
    vb_only = sorted(vb_normalized[n] for n in vb_remaining
                     if vb_normalized[n] not in matched_vb)
    print(f"\nCUB-200 のみ: {len(cub_only)} 種")
    print(f"VB100 のみ:   {len(vb_only)} 種")

    # WetlandBirds との共通種
    print(f"\n--- WetlandBirds との共通種 ---")
    wb_species = get_wetlandbirds_species()

    if wb_species:
        wb_normalized = {normalize_name(s): s for s in wb_species}

        wb_cub = set(wb_normalized.keys()) & set(cub_normalized.keys())
        wb_vb = set(wb_normalized.keys()) & set(vb_normalized.keys())

        print(f"WetlandBirds: {len(wb_species)} 種")
        print(f"WetlandBirds ∩ CUB-200: {len(wb_cub)} 種")
        print(f"WetlandBirds ∩ VB100:   {len(wb_vb)} 種")

        if wb_cub:
            for n in sorted(wb_cub):
                print(f"  CUB: {wb_normalized[n]} = {cub_normalized[n]}")
        if wb_vb:
            for n in sorted(wb_vb):
                print(f"  VB:  {wb_normalized[n]} = {vb_normalized[n]}")

        # WetlandBirds 部分一致（CUB-200, VB100）
        wb_partial_cub = []
        wb_partial_vb = []
        wb_remaining = set(wb_normalized.keys()) - wb_cub - wb_vb

        for wb_n in sorted(wb_remaining):
            wb_words = set(wb_n.split())
            for cub_n in cub_normalized.keys():
                cub_words = set(cub_n.split())
                shared = wb_words & cub_words
                if shared and len(shared) / max(len(wb_words),
                                                 len(cub_words)) >= 0.5:
                    wb_partial_cub.append((wb_normalized[wb_n],
                                            cub_normalized[cub_n], shared))
            for vb_n in vb_normalized.keys():
                vb_words = set(vb_n.split())
                shared = wb_words & vb_words
                if shared and len(shared) / max(len(wb_words),
                                                 len(vb_words)) >= 0.5:
                    wb_partial_vb.append((wb_normalized[wb_n],
                                           vb_normalized[vb_n], shared))

        if wb_partial_cub:
            print(f"\nWetlandBirds × CUB-200 部分一致: {len(wb_partial_cub)} 件")
            for wb_o, cub_o, sh in wb_partial_cub:
                print(f"  {wb_o:<35} ↔ CUB: {cub_o:<35} {sh}")
        if wb_partial_vb:
            print(f"\nWetlandBirds × VB100 部分一致: {len(wb_partial_vb)} 件")
            for wb_o, vb_o, sh in wb_partial_vb:
                print(f"  {wb_o:<35} ↔ VB:  {vb_o:<35} {sh}")
    else:
        print("WetlandBirds の種リストが取得できませんでした。")

    # サマリー
    total_common = len(common_exact) + len(partial_matches)
    print(f"\n{'='*60}")
    print(f"サマリー")
    print(f"{'='*60}")
    print(f"CUB-200 ∩ VB100 完全一致: {len(common_exact)} 種")
    print(f"CUB-200 ∩ VB100 部分一致: {len(partial_matches)} 種")
    print(f"合計共通種: {total_common} 種")

    if total_common >= 10:
        print(f"\n→ クロスデータセット評価が実施可能（{total_common} 種）")
    elif total_common >= 3:
        print(f"\n→ 限定的なクロスデータセット評価が可能（{total_common} 種）")
    else:
        print(f"\n→ 共通種が少なすぎてクロスデータセット評価は困難")


if __name__ == "__main__":
    main()
