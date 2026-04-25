"""
labels.csv のテンプレートを生成。
ユーザーが各動画を視聴して空欄を埋める運用を想定。
"""
import csv
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "youtube_greattit"
MANIFEST_PATH = DATA_DIR / "manifest.json"
LABELS_PATH = DATA_DIR / "labels.csv"


def parse_start_from_note(note: str) -> str:
    """noteに「1:02からシジュウカラ」のような記載があれば秒に変換、なければ空"""
    if not note or "から" not in note:
        return ""
    head = note.split("から")[0].strip()
    if ":" in head:
        try:
            parts = head.split(":")
            if len(parts) == 2:
                m, s = parts
                return str(int(m) * 60 + int(s))
            elif len(parts) == 3:
                h, m, s = parts
                return str(int(h) * 3600 + int(m) * 60 + int(s))
        except ValueError:
            return ""
    if head.isdigit():
        return head
    return ""


def main():
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)

    with open(LABELS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # ヘッダ（category列を追加した新フォーマット）
        writer.writerow([
            "video_id",                # 動画ID
            "species",                 # 種（great_tit固定）
            "category",                # singing / other_species_possible / no_singing
            "bird_visible",            # 鳥が画面に映っているか yes/no
            "bird_singing",            # 映っている鳥が鳴いているか yes/no
            "singing_matches_video",   # 音声と映像の鳥が一致するか yes/no
            "start_sec",               # シジュウカラが出現する開始秒
            "end_sec",                 # シジュウカラが出現する終了秒
            "quality",                 # high/medium/low
            "note",                    # メモ
        ])
        for m in manifest:
            note = m.get("note", "")
            category = m.get("category", "")
            start_sec = parse_start_from_note(note)
            writer.writerow([
                m["video_id"],
                "great_tit",
                category,
                "",  # bird_visible：ユーザー記入
                "",  # bird_singing：ユーザー記入
                "",  # singing_matches_video：ユーザー記入
                start_sec,
                "",  # end_sec：ユーザー記入
                "",  # quality：ユーザー記入
                note,
            ])

    print(f"labels.csv を作成しました: {LABELS_PATH}")
    print(f"行数: {len(manifest)} 行（ヘッダ除く）")
    print("ユーザーが各動画を視聴し、空欄を記入してください。")


if __name__ == "__main__":
    main()
