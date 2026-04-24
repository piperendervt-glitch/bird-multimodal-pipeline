"""Phase 3B の補完取得スクリプト。

- Xeno-canto v3 で 0 件だった対象種について、英名のハイフン挿入や別名を試す。
- 取得できた録音で audio_manifest.json を上書き更新する。
"""

import itertools
import json
import os
import sys
import time
import urllib.parse
import urllib.request

from bird_phase3_audio import (
    AUDIO_DIR,
    PHASE3_DIR,
    USER_AGENT,
    XCANTO_API,
    download_recording,
    parse_length_to_seconds,
    search_xeno_canto,
)

# 既知の別名（CUB-200 の古い英名 → 現行の国際名）
ALIASES = {
    "Eared Grebe": ["Black-necked Grebe"],
}


def generate_hyphen_variants(name):
    """単語間の任意の箇所にハイフンを入れた派生名を生成（重複なし）"""
    words = name.split()
    if len(words) < 2:
        return []
    results = set()
    positions = list(range(len(words) - 1))  # 各境界（word_i と word_{i+1} の間）
    # 1 箇所にハイフンを入れるパターンのみ考慮（複数入れるより現実的）
    for p in positions:
        w = words.copy()
        w[p] = w[p] + "-" + w[p + 1]
        del w[p + 1]
        results.add(" ".join(w))
    return sorted(results)


def build_candidate_names(name):
    """検索候補名のリストを返す (重複除去、元の名前を先頭)"""
    cands = [name]
    cands.extend(generate_hyphen_variants(name))
    cands.extend(ALIASES.get(name, []))
    seen, unique = set(), []
    for c in cands:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def fetch_with_fallback(species_name, output_dir, api_key, max_results=20):
    """候補名を順に試し、最初にヒットした名前でダウンロード"""
    for cand in build_candidate_names(species_name):
        print(f"  候補名で検索: {cand!r}")
        recs = search_xeno_canto(cand, api_key, max_results=max_results,
                                 quality_ok=("A", "B"),
                                 len_min=5, len_max=60)
        if len(recs) == 0:
            recs = search_xeno_canto(cand, api_key, max_results=max_results,
                                     quality_ok=None,
                                     len_min=3, len_max=180)
        if recs:
            print(f"  → ヒット: {cand!r} ({len(recs)} 件)")
            downloaded = []
            for i, rec in enumerate(recs):
                path = download_recording(rec, output_dir)
                if path:
                    downloaded.append({
                        "path": path,
                        "xc_id": rec.get("id"),
                        "species": rec.get("en", cand),
                        "country": rec.get("cnt", ""),
                        "quality": rec.get("q", ""),
                        "length": rec.get("length", ""),
                    })
                time.sleep(1)
                if (i + 1) % 5 == 0:
                    print(f"    {i+1}/{len(recs)} ダウンロード試行済")
            return cand, downloaded
    return None, []


def main():
    print("=== Phase 3B 補完: 0 件種の再取得 ===")

    api_key = os.environ.get("XENO_CANTO_API_KEY")
    if not api_key:
        print("エラー: XENO_CANTO_API_KEY が未設定")
        sys.exit(1)

    manifest_path = os.path.join(PHASE3_DIR, "audio_manifest.json")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    with open(os.path.join(PHASE3_DIR, "selected_species.json"),
              encoding="utf-8") as f:
        selected = json.load(f)["selected_species"]

    updated = False
    for sp in selected:
        cls_id = sp["class_id"]
        key = str(cls_id)
        cur = manifest.get(key, {})
        if cur.get("n_recordings", 0) > 0:
            continue
        species_name = sp["search_name"]
        print(f"\n--- 再取得: {species_name} (class {cls_id}) ---")
        sp_dir = os.path.join(AUDIO_DIR, f"class_{cls_id:03d}")
        os.makedirs(sp_dir, exist_ok=True)
        matched_name, downloaded = fetch_with_fallback(
            species_name, sp_dir, api_key)
        if downloaded:
            manifest[key] = {
                "species_name": species_name,
                "matched_name": matched_name,
                "n_recordings": len(downloaded),
                "recordings": downloaded,
            }
            updated = True
            print(f"  取得完了: {len(downloaded)} 件 (matched={matched_name!r})")
        else:
            print("  全候補で 0 件")

    if updated:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"\nマニフェスト更新: {manifest_path}")
    else:
        print("\n更新なし")

    print("\n=== 現在のマニフェスト ===")
    for k, v in manifest.items():
        print(f"  [{k}] {v.get('species_name', k)}: {v.get('n_recordings', 0)} 件")


if __name__ == "__main__":
    main()
