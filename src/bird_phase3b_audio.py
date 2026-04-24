"""Phase 3b 段階A: 対象 5 種の音声データを追加取得する。

Phase 3 の audio_manifest.json を再利用し、各種 100 件を目標にダウンロードする。
- Xeno-canto API v3 を使用（環境変数 XENO_CANTO_API_KEY）
- 既存の xc_id はスキップ
- 品質 A/B/C・長さ 5〜120 秒で緩和フィルタ
- 検索結果が 0 件なら Phase 3 で判明した別名にフォールバック
"""

import json
import os
import sys
import time
import urllib.parse
import urllib.request

from env_loader import load_env

load_env()


PHASE3_DIR = os.path.join("..", "results", "bird_phase3")
PHASE3B_DIR = os.path.join("..", "results", "bird_phase3b")
AUDIO_DIR = os.path.join("..", "data", "xeno-canto")
XCANTO_API = "https://xeno-canto.org/api/3/recordings"
USER_AGENT = "BirdPhase3b/1.0 (research use)"

TARGET_PER_SPECIES = 100
PER_PAGE = 100
MAX_PAGES = 6

# Phase 3 で判明した CUB-200 の英名 → Xeno-canto ヒット名
ALT_NAMES = {
    "Eared Grebe": "Black-necked Grebe",
    "American Three toed Woodpecker": "American Three-toed Woodpecker",
}


def search_xeno_canto_v3(species_name, api_key, page=1, per_page=PER_PAGE):
    """v3 API で en:"..." クエリを投げる（ページネーション対応）"""
    params = {
        "query": f'en:"{species_name}"',
        "key": api_key,
        "page": str(page),
        "per_page": str(per_page),
    }
    url = f"{XCANTO_API}?{urllib.parse.urlencode(params)}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"  検索エラー ({species_name}, page={page}): {e}")
        return None


def filter_recordings(recordings, min_length=5, max_length=120,
                     quality_ok=("A", "B", "C")):
    """品質・長さでフィルタ (Phase 3 より緩和版)"""
    filtered = []
    for rec in recordings:
        q = (rec.get("q") or "").upper()
        if q not in quality_ok:
            continue
        length = rec.get("length", "0:00")
        try:
            parts = length.split(":")
            if len(parts) == 2:
                seconds = int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 3:
                seconds = (
                    int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                )
            else:
                continue
        except Exception:
            continue
        if seconds < min_length or seconds > max_length:
            continue
        filtered.append(rec)
    return filtered


def gather_recordings(species_name, api_key, needed, known_ids):
    """必要数に達するまで複数ページから新規レコーディングを集める。
    検索結果が 0 件の場合は ALT_NAMES でフォールバック。"""
    candidates = [species_name]
    if species_name in ALT_NAMES:
        candidates.append(ALT_NAMES[species_name])

    collected = []
    for cand in candidates:
        if len(collected) >= needed:
            break
        print(f"  検索クエリ: en:\"{cand}\"")
        for page in range(1, MAX_PAGES + 1):
            if len(collected) >= needed:
                break
            result = search_xeno_canto_v3(cand, api_key, page=page)
            if result is None:
                break
            recordings = result.get("recordings") or []
            if not recordings:
                print(f"    ページ {page}: 0 件（総数={result.get('numRecordings','?')}）")
                break
            filtered = filter_recordings(recordings)
            added_this_page = 0
            for rec in filtered:
                rid = str(rec.get("id", ""))
                if not rid or rid in known_ids:
                    continue
                known_ids.add(rid)
                collected.append(rec)
                added_this_page += 1
                if len(collected) >= needed:
                    break
            print(f"    ページ {page}: 取得 {len(recordings)} 件, "
                  f"フィルタ後 {len(filtered)} 件, 新規追加 {added_this_page} 件, "
                  f"累計 {len(collected)}/{needed}")
            try:
                num_pages = int(result.get("numPages", 1))
            except Exception:
                num_pages = 1
            if page >= num_pages:
                break
            time.sleep(1)
    return collected


def download_recording(recording, output_dir):
    """1 件の録音を保存。既存はそのまま返す。"""
    rec_id = str(recording.get("id", ""))
    if not rec_id:
        return None
    output_path = os.path.join(output_dir, f"XC{rec_id}.mp3")
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return output_path
    file_url = recording.get("file", "")
    if not file_url:
        return None
    if file_url.startswith("//"):
        file_url = "https:" + file_url
    elif not file_url.startswith("http"):
        file_url = "https://" + file_url.lstrip("/")
    try:
        req = urllib.request.Request(file_url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        with open(output_path, "wb") as f:
            f.write(data)
        return output_path
    except Exception as e:
        print(f"    DL失敗 XC{rec_id}: {e}")
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass
        return None


def main():
    print("=== Phase 3b-A: 音声データ追加取得 ===")
    api_key = os.environ.get("XENO_CANTO_API_KEY")
    if not api_key:
        print("エラー: XENO_CANTO_API_KEY が未設定です")
        sys.exit(1)
    print("API キー: ****（.env から読み込み済み）")

    with open(os.path.join(PHASE3_DIR, "selected_species.json"),
              encoding="utf-8") as f:
        selected = json.load(f)["selected_species"]

    with open(os.path.join(PHASE3_DIR, "audio_manifest.json"),
              encoding="utf-8") as f:
        existing = json.load(f)

    os.makedirs(PHASE3B_DIR, exist_ok=True)
    updated_manifest = {}

    for sp in selected:
        cls_id = sp["class_id"]
        species_name = sp["search_name"]
        key = str(cls_id)
        sp_dir = os.path.join(AUDIO_DIR, f"class_{cls_id:03d}")
        os.makedirs(sp_dir, exist_ok=True)

        existing_recs = existing.get(key, {}).get("recordings", [])
        kept_existing = [
            r for r in existing_recs
            if r.get("path") and os.path.exists(r["path"])
        ]
        known_ids = {str(r.get("xc_id", "")) for r in kept_existing}

        print(f"\n--- {species_name} (class {cls_id}) ---")
        print(f"  既存: {len(kept_existing)} 件, 目標: {TARGET_PER_SPECIES} 件")
        needed = TARGET_PER_SPECIES - len(kept_existing)
        if needed <= 0:
            print("  既に目標達成済み。スキップ。")
            updated_manifest[key] = {
                "species_name": species_name,
                "n_recordings": len(kept_existing),
                "recordings": kept_existing,
            }
            continue

        new_candidates = gather_recordings(species_name, api_key, needed, known_ids)
        print(f"  新規候補: {len(new_candidates)} 件")

        new_downloaded = []
        for i, rec in enumerate(new_candidates):
            path = download_recording(rec, sp_dir)
            if path:
                new_downloaded.append({
                    "path": path,
                    "xc_id": str(rec.get("id", "")),
                    "species": rec.get("en", species_name),
                    "country": rec.get("cnt", ""),
                    "quality": rec.get("q", ""),
                    "length": rec.get("length", ""),
                })
            time.sleep(0.5)
            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{len(new_candidates)} DL 試行済")

        merged = kept_existing + new_downloaded
        updated_manifest[key] = {
            "species_name": species_name,
            "n_recordings": len(merged),
            "recordings": merged,
        }
        print(f"  最終取得数: {len(merged)} 件 (新規 {len(new_downloaded)} 件追加)")

    # 負例は Phase 3 の結果を引き継ぎ
    if "negative" in existing:
        updated_manifest["negative"] = existing["negative"]

    out_path = os.path.join(PHASE3B_DIR, "audio_manifest.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(updated_manifest, f, indent=2, ensure_ascii=False)

    print(f"\n=== サマリー ===")
    total = 0
    for k, v in updated_manifest.items():
        n = v.get("n_recordings", 0)
        total += n
        label = v.get("species_name", k)
        print(f"  [{k}] {label}: {n} 件")
    print(f"  合計: {total} 件")
    print(f"保存: {out_path}")


if __name__ == "__main__":
    main()
