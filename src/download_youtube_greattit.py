"""
YouTube Creative Commons動画ダウンロードスクリプト
Phase 5g統合のテストデータ準備用。
シジュウカラ（Great Tit）の鳴き声入り動画9本を取得。
"""
import json
import os
import subprocess
import sys
from pathlib import Path

# プロジェクトルート基準のパス
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "youtube_greattit"
VIDEO_DIR = DATA_DIR / "videos"
MANIFEST_PATH = DATA_DIR / "manifest.json"

VIDEO_DIR.mkdir(parents=True, exist_ok=True)

VIDEOS = [
    {"url": "https://www.youtube.com/watch?v=OunH7y_wmRw", "note": ""},
    {"url": "https://www.youtube.com/watch?v=kypF0KtidGw", "note": ""},
    {"url": "https://www.youtube.com/watch?v=w1adyz82pdI", "note": ""},
    {"url": "https://www.youtube.com/watch?v=hhy4u2dZXyw", "note": "1:02からシジュウカラ"},
    {"url": "https://www.youtube.com/watch?v=kU2HCB0kNW4", "note": ""},
    {"url": "https://www.youtube.com/watch?v=MjTTse261pU", "note": ""},
    {"url": "https://www.youtube.com/watch?v=XNooQ3qvRTI", "note": ""},
    {"url": "https://www.youtube.com/watch?v=DDPqrGDnlMw", "note": ""},
    {"url": "https://www.youtube.com/watch?v=3MqiAezCflQ", "note": ""},
]


def get_metadata(url: str) -> dict | None:
    """yt-dlpでメタデータのみ取得（ダウンロードなし）"""
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
    """720p以下のmp4でダウンロード"""
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
    manifest = []
    for i, video in enumerate(VIDEOS, start=1):
        url = video["url"]
        video_id = url.split("v=")[1].split("&")[0]
        # yt-dlpの出力テンプレで拡張子は自動付与されるため、ここではID基準に統一
        output_path = VIDEO_DIR / f"{video_id}.mp4"

        print(f"\n[{i}/{len(VIDEOS)}] {video_id}")
        print(f"  URL: {url}")

        meta = get_metadata(url)
        if meta is None:
            print("  → メタデータ取得不能のためスキップ")
            manifest.append({
                "video_id": video_id,
                "url": url,
                "note": video.get("note", ""),
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
            "note": video.get("note", ""),
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

        manifest.append(info)

    # マニフェスト保存
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nマニフェストを保存しました: {MANIFEST_PATH}")

    # 集計
    n_ok = sum(1 for m in manifest if m.get("status") in ("downloaded", "already_downloaded"))
    print(f"\nダウンロード成功: {n_ok} / {len(VIDEOS)} 本")


if __name__ == "__main__":
    main()
