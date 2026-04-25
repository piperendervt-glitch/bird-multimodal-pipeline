"""
追加分のYouTube CC動画ダウンロードスクリプト（10本）。
カテゴリ：singing / other_species_possible / no_singing。
保存先と既存manifestはそのまま使い、追加・上書きの形でマージする。
"""
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "youtube_greattit"
VIDEO_DIR = DATA_DIR / "videos"
MANIFEST_PATH = DATA_DIR / "manifest.json"

VIDEO_DIR.mkdir(parents=True, exist_ok=True)

# 既存9本に対するカテゴリ（最初の依頼で「シジュウカラ鳴き声入り」と明示されたためsinging）
EXISTING_DEFAULT_CATEGORY = "singing"

ADDITIONAL_VIDEOS = [
    # 鳴き声あり（確実）
    {"url": "https://www.youtube.com/watch?v=O-oG0X3QRVg", "category": "singing", "note": ""},
    {"url": "https://www.youtube.com/watch?v=nSwoFDJHgNI", "category": "singing", "note": ""},
    {"url": "https://www.youtube.com/watch?v=R9O27RteTpQ", "category": "singing", "note": ""},
    {"url": "https://www.youtube.com/watch?v=CQ6mA0Pm_dM", "category": "singing", "note": ""},
    {"url": "https://www.youtube.com/watch?v=GUDH_fqB_-8", "category": "singing", "note": ""},
    # 鳴き声ありだが他の鳥種の可能性あり
    {"url": "https://www.youtube.com/watch?v=tRuHOvUmuYk", "category": "other_species_possible", "note": "他の鳥種の鳴き声の可能性"},
    {"url": "https://www.youtube.com/watch?v=BpaoT5t3fCU", "category": "other_species_possible", "note": "他の鳥種の鳴き声の可能性"},
    {"url": "https://www.youtube.com/watch?v=ty5Qz2xo3Qs", "category": "other_species_possible", "note": "他の鳥種の鳴き声の可能性"},
    # 鳴き声なし
    {"url": "https://www.youtube.com/watch?v=du1bdcnvXVU", "category": "no_singing", "note": "鳴き声なし"},
    {"url": "https://www.youtube.com/watch?v=ldIOzyRrl-w", "category": "no_singing", "note": "鳴き声なし"},
]


def get_metadata(url: str) -> dict | None:
    try:
        result = subprocess.run(
            ["yt-dlp", "--print-json", "--skip-download", "--no-warnings", url],
            capture_output=True, text=True, timeout=60, encoding="utf-8",
        )
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout)
        else:
            print(f"  メタデータ取得失敗: {result.stderr[:200]}")
            return None
    except Exception as e:
        print(f"  メタデータ取得例外: {e}")
        return None


def download_video(url: str, output_path: Path) -> bool:
    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]",
                "--merge-output-format", "mp4",
                "--no-warnings",
                "-o", str(output_path),
                url,
            ],
            capture_output=True, text=True, timeout=600, encoding="utf-8",
        )
        if result.returncode == 0:
            return True
        else:
            print(f"  ダウンロード失敗: {result.stderr[-300:]}")
            return False
    except Exception as e:
        print(f"  ダウンロード例外: {e}")
        return False


def main():
    # 既存マニフェスト読み込み（categoryを後付けで埋める）
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = []

    # 既存エントリにcategoryを補完
    for m in existing:
        if "category" not in m or not m.get("category"):
            m["category"] = EXISTING_DEFAULT_CATEGORY

    existing_ids = {m["video_id"] for m in existing}

    # 追加分を処理
    new_entries = []
    for i, video in enumerate(ADDITIONAL_VIDEOS, start=1):
        url = video["url"]
        video_id = url.split("v=")[1].split("&")[0]
        category = video["category"]
        note = video.get("note", "")
        output_path = VIDEO_DIR / f"{video_id}.mp4"

        print(f"\n[追加 {i}/{len(ADDITIONAL_VIDEOS)}] {video_id}  ({category})")
        print(f"  URL: {url}")

        if video_id in existing_ids:
            print("  → 既にmanifestに存在（スキップ）")
            continue

        meta = get_metadata(url)
        if meta is None:
            new_entries.append({
                "video_id": video_id,
                "url": url,
                "category": category,
                "note": note,
                "status": "metadata_failed",
            })
            continue

        info = {
            "video_id": video_id,
            "title": meta.get("title", ""),
            "duration": meta.get("duration", 0),
            "license": meta.get("license", ""),
            "uploader": meta.get("uploader", ""),
            "upload_date": meta.get("upload_date", ""),
            "url": url,
            "category": category,
            "note": note,
            "filename": f"{video_id}.mp4",
            "status": "pending",
        }
        print(f"  タイトル: {info['title'][:60]}")
        print(f"  長さ: {info['duration']}秒  ライセンス: {info['license']}  投稿者: {info['uploader']}")

        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"  → 既にダウンロード済み ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
            info["status"] = "already_downloaded"
        else:
            print("  → ダウンロード開始")
            ok = download_video(url, output_path)
            if ok and output_path.exists():
                info["status"] = "downloaded"
                size_mb = output_path.stat().st_size / 1024 / 1024
                print(f"  → 完了 ({size_mb:.1f} MB)")
            else:
                info["status"] = "download_failed"

        new_entries.append(info)

    # マージして保存
    merged = existing + new_entries
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"\nマニフェストを保存: {MANIFEST_PATH}")
    print(f"  既存: {len(existing)} 本 / 追加: {len(new_entries)} 本 / 合計: {len(merged)} 本")

    n_ok_added = sum(1 for m in new_entries if m.get("status") in ("downloaded", "already_downloaded"))
    print(f"追加分のダウンロード成功: {n_ok_added} / {len(ADDITIONAL_VIDEOS)} 本")


if __name__ == "__main__":
    main()
