"""全手法のベンチマーク一括実行スクリプト。

使い方:
  cd src
  python benchmark_all.py
  python benchmark_all.py --video-dir ../data/youtube_greattit/videos
  python benchmark_all.py --max-frames 300        # 各動画の最初の 300 フレームのみ
  python benchmark_all.py --methods bytetrack sahi  # 特定の手法のみ
  python benchmark_all.py --slice-size 256          # SAHI のスライスサイズ変更

注意:
- realtime_preview.py は手法ごとに別ファイル名で保存する仕様
  （benchmark_<tracker>.json / benchmark_sahi_<size>.json）
- 既に走行中の realtime_preview.py がある場合は競合するため停止してから実行
"""

import argparse
import json
import os
import subprocess
import sys
import time


METHODS = [
    {"name": "none",       "args": ["--tracker", "none"]},
    {"name": "bytetrack",  "args": ["--tracker", "bytetrack"]},
    {"name": "botsort",    "args": ["--tracker", "botsort"]},
    {"name": "sahi",       "args": ["--sahi"]},
    {"name": "sahi-track", "args": ["--sahi-track"]},
]


def benchmark_filename(method_name, slice_size=320):
    """手法ごとの保存ファイル名を返す（realtime_preview.py の規約と一致）。"""
    if method_name == "sahi":
        return f"benchmark_sahi_{slice_size}.json"
    if method_name == "sahi-track":
        return f"benchmark_sahi_track_{slice_size}.json"
    return f"benchmark_{method_name}.json"


def run_benchmark(method, video_dir, max_frames=0, skip=0,
                   slice_size=320):
    """1 つの手法でベンチマークを実行する。"""
    cmd = [
        sys.executable, "-u", "realtime_preview.py",
        "--benchmark", "--no-display",
    ] + method["args"]

    if method["name"] in ("sahi", "sahi-track"):
        cmd += ["--slice-size", str(slice_size)]
    if max_frames > 0:
        cmd += ["--max-frames", str(max_frames)]
    if skip > 0:
        cmd += ["--skip", str(skip)]

    print(f"\n{'='*60}")
    print(f"手法: {method['name']}")
    print(f"コマンド: {' '.join(cmd)}")
    print(f"{'='*60}", flush=True)

    start = time.time()

    # 標準出力をリアルタイム表示しながら最終出力も保持
    env = {**os.environ, "PYTHONUNBUFFERED": "1",
           "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", env=env,
    )

    captured = []
    try:
        for line in proc.stdout:
            captured.append(line)
            sys.stdout.write(line)
            sys.stdout.flush()
        proc.wait(timeout=7200)  # 2 時間タイムアウト
    except subprocess.TimeoutExpired:
        proc.kill()
        print("  タイムアウト（2 時間）で中断しました。")
        return None

    elapsed = time.time() - start

    if proc.returncode != 0:
        print(f"  エラー（終了コード {proc.returncode}）")
        return None

    # 手法ごとの保存ファイルを読み込み
    benchmark_path = os.path.join(
        "..", "results", "realtime",
        benchmark_filename(method["name"], slice_size),
    )
    if os.path.exists(benchmark_path):
        with open(benchmark_path, encoding="utf-8") as f:
            data = json.load(f)
        data["method"] = method["name"]
        data["wall_time_sec"] = elapsed
        return data

    print(f"  警告: 保存ファイルが見つかりません: {benchmark_path}")
    return None


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="全手法ベンチマーク一括実行"
    )
    parser.add_argument("--video-dir",
                         default="../data/youtube_greattit/videos",
                         help="動画ディレクトリ")
    parser.add_argument("--max-frames", type=int, default=0,
                         help="各動画の最大処理フレーム数（0=全フレーム）")
    parser.add_argument("--skip", type=int, default=0,
                         help="フレームスキップ数")
    parser.add_argument("--slice-size", type=int, default=320,
                         help="SAHI のスライスサイズ")
    parser.add_argument("--methods", nargs="+",
                         default=["none", "bytetrack", "botsort",
                                  "sahi", "sahi-track"],
                         help="実行する手法（none bytetrack botsort "
                              "sahi sahi-track）")
    parser.add_argument("--resume", action="store_true",
                         help="前回の中断から再開（完了済みをスキップ）")
    args = parser.parse_args()

    methods_to_run = [m for m in METHODS if m["name"] in args.methods]
    if not methods_to_run:
        print("エラー: 有効な手法がありません。"
              "選択肢: none, bytetrack, botsort, sahi")
        return

    print(f"=== 全手法ベンチマーク ===")
    print(f"手法: {[m['name'] for m in methods_to_run]}")
    print(f"動画: {args.video_dir}")
    if args.max_frames > 0:
        print(f"最大フレーム: {args.max_frames}")
    if any(m in args.methods for m in ("sahi", "sahi-track")):
        print(f"SAHI スライスサイズ: {args.slice_size}")

    # 既存結果の読み込み（--resume 時）
    results_path = "../results/realtime/benchmark_all.json"
    existing_results = {}
    if args.resume and os.path.exists(results_path):
        try:
            with open(results_path, encoding="utf-8") as f:
                prev = json.load(f)
            existing_results = prev.get("results", {})
            if existing_results:
                print(f"\n--resume: 既存結果を読み込み: "
                      f"{list(existing_results.keys())}")
        except Exception as e:
            print(f"既存 JSON の読込失敗: {e}（新規実行）")

    total_start = time.time()
    all_results = dict(existing_results)

    for method in methods_to_run:
        # 完了済みの手法はスキップ
        if args.resume and method["name"] in all_results:
            print(f"\nスキップ（完了済み）: {method['name']}")
            continue

        result = run_benchmark(
            method, args.video_dir,
            max_frames=args.max_frames,
            skip=args.skip,
            slice_size=args.slice_size,
        )
        if result:
            all_results[method["name"]] = result

        # 各手法完了後に中間保存（中断に備える）
        os.makedirs("../results/realtime", exist_ok=True)
        intermediate = {
            "methods_completed": list(all_results.keys()),
            "methods_planned": [m["name"] for m in methods_to_run],
            "elapsed_sec": time.time() - total_start,
            "results": all_results,
        }
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(intermediate, f, indent=2, ensure_ascii=False)
        print(f"  中間保存: {results_path}")

    total_elapsed = time.time() - total_start

    # ========================================
    # 統合比較表
    # ========================================
    print(f"\n{'='*80}")
    print(f"統合比較表")
    print(f"{'='*80}")

    print(f"\n{'手法':<12} {'平均FPS':>8} {'全体FPS':>8} {'処理時間':>10} "
          f"{'YOLO':>8} {'DINOv2':>8} {'検出数':>8}")
    print("-" * 75)

    for name, r in all_results.items():
        total_det = r.get("total_detections", "N/A")
        det_str = (f"{total_det:>8}" if isinstance(total_det, int)
                   else f"{total_det:>8}")
        print(f"{name:<12} {r.get('avg_fps', 0):>7.1f} "
              f"{r.get('overall_fps', 0):>7.1f} "
              f"{r.get('wall_time_sec', 0):>9.1f}s "
              f"{r.get('avg_yolo_ms', 0):>7.0f}ms "
              f"{r.get('avg_dinov2_ms', 0):>7.0f}ms "
              f"{det_str}")

    # ========================================
    # 追跡統計（追跡ありの手法のみ）
    # ========================================
    has_tracking = any(
        m["name"] in ("bytetrack", "botsort") for m in methods_to_run
    )
    if has_tracking:
        print(f"\n{'手法':<12} {'ユニークID合計':>14} {'最長追跡':>10} "
              f"{'平均追跡':>10}")
        print("-" * 55)

        for name, r in all_results.items():
            per_video = r.get("per_video", [])
            if not per_video:
                continue

            # フィールド名は realtime_preview.py の規約に合わせる
            total_ids = sum(v.get("unique_ids", 0) for v in per_video)
            max_track = max((v.get("max_track_frames", 0) for v in per_video),
                              default=0)
            mean_tracks = [v.get("mean_track_frames", 0.0)
                            for v in per_video]
            avg_track = (sum(mean_tracks) / len(mean_tracks)
                         if mean_tracks else 0.0)

            if total_ids > 0 or max_track > 0:
                print(f"{name:<12} {total_ids:>14} {max_track:>10} "
                      f"{avg_track:>9.1f}")

    print(f"\n総ベンチマーク時間: {total_elapsed:.0f} 秒 "
          f"({total_elapsed/60:.1f} 分)")

    # ========================================
    # 最終保存
    # ========================================
    os.makedirs("../results/realtime", exist_ok=True)
    output = {
        "methods_completed": list(all_results.keys()),
        "methods_planned": [m["name"] for m in methods_to_run],
        "total_time_sec": total_elapsed,
        "max_frames": args.max_frames,
        "skip_frames": args.skip,
        "slice_size": args.slice_size,
        "results": all_results,
    }
    out_path = "../results/realtime/benchmark_all.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n保存: {out_path}")


if __name__ == "__main__":
    main()
