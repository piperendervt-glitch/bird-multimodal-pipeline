"""Phase 3 段階B: Xeno-canto API v3 から音声を取得する。

v3 は API キーが必須なので、環境変数 XENO_CANTO_API_KEY から読み込む。
対象 5 種 + 一般的な負例 5 種の録音をダウンロードし、マニフェストに保存する。
このスクリプトは DINOv2 や MLP をインポートしない。
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
AUDIO_DIR = os.path.join("..", "data", "xeno-canto")
XCANTO_API = "https://xeno-canto.org/api/3/recordings"

USER_AGENT = "BirdPhase3/1.0 (research use)"
MAX_PER_SPECIES = 20
MAX_PER_NEGATIVE = 5
# 負例候補（対象種と異なる一般種）
NEGATIVE_SPECIES = [
    "House Sparrow",
    "European Robin",
    "Common Blackbird",
    "Great Tit",
    "Blue Jay",
]


def parse_length_to_seconds(length_str):
    """'m:ss' または 's' 形式を秒数 int にして返す（失敗時 None）"""
    if not length_str:
        return None
    try:
        parts = length_str.split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        else:
            return int(float(parts[0]))
    except Exception:
        return None


def search_xeno_canto(species_name, api_key, max_results=20,
                     quality_ok=("A", "B"), len_min=5, len_max=60):
    """Xeno-canto API v3 で en:"..." クエリを投げ、品質・長さでフィルタ"""
    query_str = f'en:"{species_name}"'
    params = {"query": query_str, "key": api_key}
    url = f"{XCANTO_API}?{urllib.parse.urlencode(params)}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read()
        data = json.loads(body)
    except Exception as e:
        print(f"  警告: '{species_name}' の検索失敗: {e}")
        return []

    if "error" in data:
        print(f"  警告: API エラー: {data.get('message') or data.get('error')}")
        return []

    recordings = data.get("recordings", []) or []
    total = len(recordings)

    filtered = []
    for rec in recordings:
        q = (rec.get("q") or "").upper()
        if quality_ok and q not in quality_ok:
            continue
        secs = parse_length_to_seconds(rec.get("length", ""))
        if secs is None:
            continue
        if secs < len_min or secs > len_max:
            continue
        filtered.append(rec)

    print(f"  API 取得: 総数 {total} 件 (numRecordings="
          f"{data.get('numRecordings', '?')}), フィルタ後 {len(filtered)} 件")
    return filtered[:max_results]


def download_recording(recording, output_dir):
    """1 件の録音を保存。既存ならそのパスを返す。"""
    rec_id = recording.get("id")
    if rec_id is None:
        return None

    file_url = recording.get("file")
    if not file_url:
        return None
    if file_url.startswith("//"):
        file_url = "https:" + file_url
    elif not file_url.startswith("http"):
        file_url = "https://" + file_url.lstrip("/")

    output_path = os.path.join(output_dir, f"XC{rec_id}.mp3")
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return output_path

    try:
        req = urllib.request.Request(file_url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        with open(output_path, "wb") as f:
            f.write(data)
        return output_path
    except Exception as e:
        print(f"    ダウンロード失敗 (XC{rec_id}): {e}")
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass
        return None


def fetch_for_species(species_name, output_dir, max_results, api_key):
    """1 種分の録音をダウンロードし結果リストを返す（厳格 → 緩和の 2 段構え）"""
    recs = search_xeno_canto(species_name, api_key,
                             max_results=max_results,
                             quality_ok=("A", "B"), len_min=5, len_max=60)
    if len(recs) == 0:
        print("  品質・長さ条件を緩和して再検索...")
        recs = search_xeno_canto(species_name, api_key,
                                 max_results=max_results,
                                 quality_ok=None, len_min=3, len_max=180)

    downloaded = []
    for i, rec in enumerate(recs):
        path = download_recording(rec, output_dir)
        if path:
            downloaded.append({
                "path": path,
                "xc_id": rec.get("id"),
                "species": rec.get("en", species_name),
                "country": rec.get("cnt", ""),
                "quality": rec.get("q", ""),
                "length": rec.get("length", ""),
            })
        time.sleep(1)
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(recs)} ダウンロード試行済")
    return downloaded


def main():
    print("=== Phase 3B: Xeno-canto 音声取得 (API v3) ===")

    api_key = os.environ.get("XENO_CANTO_API_KEY")
    if not api_key:
        print("エラー: XENO_CANTO_API_KEY 環境変数が設定されていません。")
        print("以下を実行してください:")
        print('  export XENO_CANTO_API_KEY="your_key_here"   # bash/zsh')
        print('  $env:XENO_CANTO_API_KEY="your_key_here"     # PowerShell')
        sys.exit(1)

    print("API キー: ****（.env から読み込み済み）")

    with open(os.path.join(PHASE3_DIR, "selected_species.json"),
              encoding="utf-8") as f:
        data = json.load(f)
    selected = data["selected_species"]

    os.makedirs(AUDIO_DIR, exist_ok=True)
    audio_manifest = {}

    # 対象種
    for sp in selected:
        cls_id = sp["class_id"]
        species_name = sp["search_name"]
        sp_dir = os.path.join(AUDIO_DIR, f"class_{cls_id:03d}")
        os.makedirs(sp_dir, exist_ok=True)

        print(f"\n--- {species_name} (class {cls_id}) ---")
        downloaded = fetch_for_species(species_name, sp_dir, MAX_PER_SPECIES, api_key)
        audio_manifest[str(cls_id)] = {
            "species_name": species_name,
            "n_recordings": len(downloaded),
            "recordings": downloaded,
        }
        print(f"  取得完了: {len(downloaded)} 件")

    # 負例
    print(f"\n--- 負例用音声の取得 ---")
    neg_dir = os.path.join(AUDIO_DIR, "negative")
    os.makedirs(neg_dir, exist_ok=True)

    neg_downloaded = []
    for neg_sp in NEGATIVE_SPECIES:
        print(f"\n  {neg_sp}...")
        recs = search_xeno_canto(neg_sp, api_key,
                                 max_results=MAX_PER_NEGATIVE,
                                 quality_ok=("A", "B"), len_min=5, len_max=60)
        if len(recs) == 0:
            recs = search_xeno_canto(neg_sp, api_key,
                                     max_results=MAX_PER_NEGATIVE,
                                     quality_ok=None, len_min=3, len_max=180)
        for rec in recs:
            path = download_recording(rec, neg_dir)
            if path:
                neg_downloaded.append({
                    "path": path,
                    "species": neg_sp,
                    "xc_id": rec.get("id"),
                    "quality": rec.get("q", ""),
                    "length": rec.get("length", ""),
                })
            time.sleep(1)

    audio_manifest["negative"] = {
        "species_name": "negative",
        "species_list": NEGATIVE_SPECIES,
        "n_recordings": len(neg_downloaded),
        "recordings": neg_downloaded,
    }

    out_path = os.path.join(PHASE3_DIR, "audio_manifest.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(audio_manifest, f, indent=2, ensure_ascii=False)

    print(f"\n=== 音声取得サマリー ===")
    total = 0
    for key, val in audio_manifest.items():
        n = val["n_recordings"]
        total += n
        label = val.get("species_name", key)
        print(f"  [{key}] {label}: {n} 件")
    print(f"  合計: {total} 件")
    print(f"保存: {out_path}")


if __name__ == "__main__":
    main()
