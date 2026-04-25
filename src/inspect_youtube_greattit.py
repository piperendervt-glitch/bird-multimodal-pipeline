"""
ダウンロード済み動画の詳細をffprobeで確認するスクリプト。
解像度・フレームレート・音声・長さ・サイズを集計。
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


def probe(path: Path) -> dict | None:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", "-show_format", str(path)],
            capture_output=True, text=True, timeout=30, encoding="utf-8",
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except Exception as e:
        print(f"  ffprobe例外: {e}")
    return None


def main():
    videos = sorted([f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")])
    print(f"ダウンロード済み: {len(videos)} 本")
    print("-" * 100)

    total_size = 0.0
    total_dur = 0.0
    rows = []
    for v in videos:
        path = VIDEO_DIR / v
        size_mb = path.stat().st_size / 1024 / 1024
        total_size += size_mb

        info = probe(path)
        if info is None:
            print(f"{v}: ffprobe失敗")
            continue

        v_stream = next((s for s in info.get("streams", []) if s["codec_type"] == "video"), None)
        a_stream = next((s for s in info.get("streams", []) if s["codec_type"] == "audio"), None)
        fmt = info.get("format", {})

        w = v_stream.get("width", "?") if v_stream else "?"
        h = v_stream.get("height", "?") if v_stream else "?"
        fps_raw = v_stream.get("r_frame_rate", "0/1") if v_stream else "0/1"
        try:
            num, den = fps_raw.split("/")
            fps = round(int(num) / int(den), 2) if int(den) > 0 else 0
        except Exception:
            fps = fps_raw
        dur = float(fmt.get("duration", 0) or 0)
        total_dur += dur

        if a_stream:
            sr = a_stream.get("sample_rate", "?")
            ch = a_stream.get("channels", "?")
            a_codec = a_stream.get("codec_name", "?")
            audio_str = f"{a_codec} {sr}Hz {ch}ch"
        else:
            audio_str = "音声なし"

        v_codec = v_stream.get("codec_name", "?") if v_stream else "?"

        row = {
            "filename": v,
            "resolution": f"{w}x{h}",
            "fps": fps,
            "duration_sec": round(dur, 2),
            "size_mb": round(size_mb, 2),
            "video_codec": v_codec,
            "audio": audio_str,
        }
        rows.append(row)
        print(f"{v}: {w}x{h} {fps}fps {round(dur,1)}秒 {size_mb:.1f}MB | 映像:{v_codec} | 音声:{audio_str}")

    print("-" * 100)
    print(f"合計: {len(rows)}本 / {round(total_dur, 1)}秒 ({round(total_dur/60, 1)}分) / {total_size:.1f} MB")

    # JSON出力（後で参照しやすいように）
    out_path = DATA_DIR / "ffprobe_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "videos": rows,
            "total_count": len(rows),
            "total_duration_sec": round(total_dur, 2),
            "total_size_mb": round(total_size, 2),
        }, f, indent=2, ensure_ascii=False)
    print(f"サマリーを保存: {out_path}")


if __name__ == "__main__":
    main()
