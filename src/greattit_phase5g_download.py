"""Phase 5g 段階A: Great Tit データセットのメタデータ・特徴量をダウンロード。

OSF からメタデータと特徴ベクトル CSV のみをダウンロードする。
音声ファイル（11.4 GB）はダウンロードしない。
"""

import os
import sys
import urllib.request
import json
import time


# Windows cp932 コンソールでも出力できるよう UTF-8 化
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


OSF_BASE = "https://osf.io/n8ac9/"
DATA_DIR = "../data/great-tit-hits"

# ダウンロード対象（音声 ZIP や大容量ファイルは除外）
TARGET_FILES = [
    "great-tit-hits.csv",
    "feature_vectors.csv",
    "main.csv",
    "morphometrics.csv",
    "nestboxes.csv",
]

# 明示的に除外したいファイル（音声 ZIP など）
EXCLUDE_KEYWORDS = [
    ".zip", ".wav", ".mp3", ".flac",
    "song-files", "songs.zip",
]


def fetch_json(url, timeout=30):
    """OSF API から JSON を取得"""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "GreatTitPhase5g/1.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def list_files_recursive(files_link, path_prefix="", depth=0, max_depth=5):
    """OSF の files リンクを再帰的にたどり、全ファイル情報を収集"""
    if depth > max_depth:
        return []

    result = []
    try:
        data = fetch_json(files_link)
    except Exception as e:
        print(f"    (取得失敗: {e})")
        return result

    for item in data.get("data", []):
        attrs = item.get("attributes", {})
        name = attrs.get("name", "")
        size = attrs.get("size", 0) or 0
        kind = attrs.get("kind", "")
        links = item.get("links", {})
        download_url = links.get("download", "")
        full_path = f"{path_prefix}/{name}" if path_prefix else name

        indent = "  " * (depth + 1)

        if kind == "folder":
            print(f"{indent}[DIR ] {full_path}")
            rel = item.get("relationships", {})
            sub_link = (
                rel.get("files", {})
                .get("links", {})
                .get("related", {})
                .get("href", "")
            )
            if sub_link:
                # フォルダ内を除外キーワードで早期スキップ
                if any(ex in name.lower() for ex in ["song-files", "audio", "wav", "mp3"]):
                    print(f"{indent}  (音声フォルダのためスキップ)")
                    continue
                result.extend(
                    list_files_recursive(sub_link, full_path, depth + 1, max_depth)
                )
        else:
            # 音声系は除外
            lower = name.lower()
            if any(ex in lower for ex in EXCLUDE_KEYWORDS):
                print(f"{indent}[SKIP] {full_path} ({size/1024/1024:.1f} MB, 音声系)")
                continue
            print(f"{indent}[FILE] {full_path:<45} {size/1024/1024:>8.1f} MB")
            result.append({
                "name": full_path,
                "basename": name,
                "size": size,
                "kind": kind,
                "download_url": download_url,
            })

    return result


def main():
    print("=== Phase 5g-A: Great Tit データセット ダウンロード ===")

    os.makedirs(DATA_DIR, exist_ok=True)

    # OSF API で全ファイル一覧を再帰取得
    api_url = "https://api.osf.io/v2/nodes/n8ac9/files/osfstorage/"
    print(f"\nOSF API からファイル一覧を取得中...")
    print(f"URL: {api_url}")

    print(f"\n--- ファイル一覧 ---")
    try:
        files_info = list_files_recursive(api_url, path_prefix="", depth=0)
    except Exception as e:
        print(f"\nOSF API エラー: {e}")
        print(f"\n代替手段: 以下の URL からブラウザで手動ダウンロードしてください:")
        print(f"  https://osf.io/n8ac9/")
        print(f"  必要なファイル:")
        for t in TARGET_FILES:
            print(f"    - {t}")
        print(f"  保存先: {DATA_DIR}")
        return

    print(f"\n取得できたファイル総数: {len(files_info)}")

    # マニフェスト保存
    try:
        with open(
            os.path.join(DATA_DIR, "download_manifest.json"),
            "w", encoding="utf-8"
        ) as f:
            json.dump(files_info, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"マニフェスト保存失敗: {e}")

    # 対象ファイルをダウンロード
    print(f"\n=== ダウンロード対象のファイル特定 ===")
    for target in TARGET_FILES:
        # basename 完全一致を優先、次に部分一致
        matched = [f for f in files_info if f["basename"] == target and f.get("download_url")]
        if not matched:
            matched = [
                f for f in files_info
                if target in f["name"] and f.get("download_url")
            ]

        if not matched:
            print(f"\n[WARN] {target} が見つかりません。")
            print(f"       OSF (https://osf.io/n8ac9/) から手動ダウンロードしてください。")
            continue

        f = matched[0]
        output_path = os.path.join(DATA_DIR, target)
        if os.path.exists(output_path):
            size = os.path.getsize(output_path)
            print(f"\n[SKIP] 既存: {target} ({size/1024/1024:.1f} MB)")
            continue

        print(f"\n[DL  ] {f['name']} ({f['size']/1024/1024:.1f} MB)")
        print(f"       URL: {f['download_url']}")
        try:
            urllib.request.urlretrieve(f["download_url"], output_path)
            saved_size = os.path.getsize(output_path)
            print(f"       保存: {output_path} ({saved_size/1024/1024:.1f} MB)")
        except Exception as dl_err:
            print(f"       ダウンロード失敗: {dl_err}")
        time.sleep(1)

    # ダウンロード結果の確認（ASCII のみ使用）
    print(f"\n=== ダウンロード結果 ===")
    for target in TARGET_FILES:
        path = os.path.join(DATA_DIR, target)
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"  [OK] {target}: {size/1024/1024:.1f} MB")
        else:
            print(f"  [NG] {target}: 未取得")


if __name__ == "__main__":
    main()
