"""
.env ファイルから環境変数を読み込むユーティリティ。
外部パッケージ不要（標準ライブラリのみ）。
"""
import os


def load_env(env_path=None):
    """プロジェクト内の .env を検索して環境変数にロードする。

    - env_path が None のときは、スクリプト位置の1つ上（プロジェクトルート）、
      カレントディレクトリ、カレントの1つ上を順に探す。
    - "ここにキーを貼る" のままのプレースホルダーは無視する。
    - 既に os.environ に同名キーが設定されていても .env の値で上書きする。
    """
    if env_path is None:
        candidates = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"),
            os.path.join(os.getcwd(), ".env"),
            os.path.join(os.getcwd(), "..", ".env"),
        ]
        for path in candidates:
            if os.path.exists(path):
                env_path = os.path.abspath(path)
                break

    if env_path is None or not os.path.exists(env_path):
        return False

    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if value and value != "ここにキーを貼る":
                    os.environ[key] = value

    return True
