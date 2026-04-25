"""リアルタイム鳥検出プレビュー（YOLO + DINOv2 バッチ推論）。

機能:
- 動画ファイルから 1 フレームずつ読み込み YOLO で鳥を検出
- 検出領域を DINOv2 でバッチ推論し特徴量を抽出
- OpenCV ウィンドウに枠と FPS を描画してリアルタイム表示
  （Claude Code 等ヘッドレス環境では --no-display / --benchmark）
- --benchmark で 19 本の動画を一括ベンチマーク

操作:
  q: 終了 / p: 一時停止
"""

import argparse
import json
import os
import sys
import time
from collections import deque

import cv2
import numpy as np


class RealtimeBirdDetector:
    """リアルタイム鳥検出・分類パイプライン。"""

    def __init__(self, species_mapping_path=None, classifier_data_path=None):
        print("=== リアルタイム鳥検出システム 初期化 ===")

        # 遅延 import（モデルロードの前に echo して初期化進捗を見えやすく）
        import torch
        import torchvision.transforms as T
        from ultralytics import YOLO

        self._torch = torch
        self._T = T

        # YOLO ロード
        print("YOLO 読み込み中...")
        self.yolo = YOLO("yolov8n.pt")
        self.BIRD_CLASS_ID = 14

        # DINOv2 ロード
        print("DINOv2 読み込み中...")
        self.dinov2 = torch.hub.load("facebookresearch/dinov2",
                                      "dinov2_vits14")
        self.dinov2.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dinov2 = self.dinov2.to(self.device)

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        # 学習済み分類器（あれば使用）
        self.classifier = None
        self.scaler = None
        self.species_names = None
        self._load_classifier(species_mapping_path, classifier_data_path)

        # パフォーマンス計測（直近 30 フレームの移動平均）
        self.fps_history = deque(maxlen=30)
        self.yolo_time = deque(maxlen=30)
        self.dinov2_time = deque(maxlen=30)

        print(f"デバイス: {self.device}")
        print("初期化完了\n")

    def _load_classifier(self, species_mapping_path, classifier_data_path):
        """学習済み分類器を試行的に読み込む。"""
        candidates = [
            ("../results/vb100_phase5e/species_mapping.json",
             "../results/vb100_phase5e/"),
            ("../results/bird_phase4b/species_mapping.json",
             "../results/bird_phase4b/"),
        ]
        if species_mapping_path:
            candidates.insert(0,
                              (species_mapping_path, classifier_data_path or ""))

        for mapping_path, data_dir in candidates:
            if os.path.exists(mapping_path):
                try:
                    with open(mapping_path, encoding="utf-8") as f:
                        mapping = json.load(f)
                    self.species_names = (
                        mapping.get("id_to_species")
                        or mapping.get("species_to_id")
                        or {}
                    )
                    print(f"種名マッピング読み込み: {mapping_path} "
                          f"({len(self.species_names)} 種)")
                    break
                except Exception:
                    continue

        if not self.species_names:
            print("種名マッピングが見つかりません。検出のみモードで動作します。")

    def detect_birds(self, frame):
        """1 フレームで鳥クラスのみ抽出して返す。"""
        results = self.yolo(frame, verbose=False, conf=0.25)

        detections = []
        for box in results[0].boxes:
            if int(box.cls[0]) == self.BIRD_CLASS_ID:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": conf,
                })
        return detections

    def extract_features_batch(self, frame, detections, padding_ratio=0.3):
        """複数検出をバッチで DINOv2 推論する。"""
        if not detections:
            return []

        h, w = frame.shape[:2]
        tensors = []
        valid_indices = []

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det["bbox"]
            bw, bh = x2 - x1, y2 - y1

            px1 = max(0, int(x1 - bw * padding_ratio))
            py1 = max(0, int(y1 - bh * padding_ratio))
            px2 = min(w, int(x2 + bw * padding_ratio))
            py2 = min(h, int(y2 + bh * padding_ratio))

            crop = frame[py1:py2, px1:px2]
            if crop.size == 0:
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = self.transform(crop_rgb)
            tensors.append(tensor)
            valid_indices.append(i)

        if not tensors:
            return [None] * len(detections)

        batch = self._torch.stack(tensors).to(self.device)
        with self._torch.no_grad():
            features_batch = self.dinov2(batch).cpu().numpy()

        results = [None] * len(detections)
        for idx, feat in zip(valid_indices, features_batch):
            results[idx] = feat
        return results

    def process_frame(self, frame):
        """1 フレームの完全処理パイプライン。"""
        frame_start = time.time()

        # YOLO
        yolo_start = time.time()
        detections = self.detect_birds(frame)
        yolo_elapsed = time.time() - yolo_start
        self.yolo_time.append(yolo_elapsed)

        # DINOv2 バッチ
        dinov2_start = time.time()
        features_list = self.extract_features_batch(frame, detections)
        dinov2_elapsed = time.time() - dinov2_start
        self.dinov2_time.append(dinov2_elapsed)

        results = []
        for det, feat in zip(detections, features_list):
            results.append({
                "bbox": det["bbox"],
                "yolo_confidence": det["confidence"],
                "features": feat,
                "species": "Unknown",
                "species_confidence": 0.0,
            })

        frame_elapsed = time.time() - frame_start
        self.fps_history.append(1.0 / max(frame_elapsed, 1e-9))
        return results

    def draw_results(self, frame, results):
        """検出結果と FPS をフレームに描画する。"""
        display = frame.copy()

        for r in results:
            x1, y1, x2, y2 = r["bbox"]
            yolo_conf = r["yolo_confidence"]
            species = r["species"]

            color = (0, 255, 0)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

            label = f"Bird {yolo_conf:.2f}"
            if species != "Unknown":
                label = f"{species} {r['species_confidence']:.2f}"

            (tw, th), _ = cv2.getTextSize(label,
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.6, 1)
            cv2.rectangle(display, (x1, y1 - th - 8),
                           (x1 + tw + 4, y1), color, -1)
            cv2.putText(display, label, (x1 + 2, y1 - 4),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        if self.fps_history:
            avg_fps = np.mean(list(self.fps_history))
            avg_yolo = np.mean(list(self.yolo_time)) * 1000
            avg_dinov2 = np.mean(list(self.dinov2_time)) * 1000

            info_lines = [
                f"FPS: {avg_fps:.1f}",
                f"YOLO: {avg_yolo:.0f}ms",
                f"DINOv2: {avg_dinov2:.0f}ms",
                f"Birds: {len(results)}",
            ]
            for i, line in enumerate(info_lines):
                y = 25 + i * 22
                cv2.putText(display, line, (10, y),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                             (0, 255, 255), 1)
        return display

    def process_video(self, video_path, display=True, save_path=None,
                       skip_frames=0, max_frames=0):
        """動画ファイルを処理する。"""
        print(f"\n処理開始: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"エラー: {video_path} を開けません")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"  解像度: {width}x{height}, FPS: {fps:.1f}, "
              f"総フレーム: {total_frames}")

        writer = None
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(save_path, fourcc, fps,
                                       (width, height))

        frame_count = 0
        processed_count = 0
        all_detections = []
        process_interval = max(1, skip_frames + 1)
        last_results = []

        # この動画のローカル平均を計算するため、開始位置のスナップショット
        local_yolo = []
        local_dinov2 = []
        local_fps = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if max_frames > 0 and frame_count > max_frames:
                break

            if frame_count % process_interval == 0:
                results = self.process_frame(frame)
                last_results = results
                processed_count += 1

                if self.yolo_time:
                    local_yolo.append(self.yolo_time[-1])
                if self.dinov2_time:
                    local_dinov2.append(self.dinov2_time[-1])
                if self.fps_history:
                    local_fps.append(self.fps_history[-1])

                all_detections.append({
                    "frame": frame_count,
                    "timestamp": frame_count / fps,
                    "n_birds": len(results),
                    "detections": [
                        {"bbox": r["bbox"],
                         "confidence": r["yolo_confidence"]}
                        for r in results
                    ],
                })

            if display or writer:
                display_frame = self.draw_results(frame, last_results)

                if display:
                    cv2.imshow("Bird Detection", display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        print("ユーザーにより中断")
                        break
                    elif key == ord("p"):
                        print("一時停止（任意のキーで再開）")
                        cv2.waitKey(0)

                if writer:
                    writer.write(display_frame)

        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()

        avg_fps = float(np.mean(local_fps)) if local_fps else 0.0
        avg_yolo = float(np.mean(local_yolo)) * 1000 if local_yolo else 0.0
        avg_dinov2 = (float(np.mean(local_dinov2)) * 1000
                      if local_dinov2 else 0.0)

        print(f"\n=== 処理サマリー ===")
        print(f"処理フレーム: {processed_count} / {frame_count}")
        print(f"平均 FPS: {avg_fps:.1f}")
        print(f"YOLO 平均: {avg_yolo:.0f} ms")
        print(f"DINOv2 平均: {avg_dinov2:.0f} ms")
        print(f"検出イベント: "
              f"{sum(d['n_birds'] > 0 for d in all_detections)}")

        return {
            "video_path": video_path,
            "total_frames": frame_count,
            "processed_frames": processed_count,
            "avg_fps": avg_fps,
            "avg_yolo_ms": avg_yolo,
            "avg_dinov2_ms": avg_dinov2,
            "detections": all_detections,
        }


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="リアルタイム鳥検出プレビュー"
    )
    parser.add_argument("video", nargs="?", default=None,
                         help="動画ファイルパス（省略時はテスト動画を使用）")
    parser.add_argument("--no-display", action="store_true",
                         help="表示なし（ヘッドレスモード）")
    parser.add_argument("--save", type=str, default=None,
                         help="結果動画の保存先パス")
    parser.add_argument("--skip", type=int, default=0,
                         help="フレームスキップ数（0=全フレーム処理）")
    parser.add_argument("--max-frames", type=int, default=0,
                         help="最大処理フレーム数（0=全フレーム）")
    parser.add_argument("--benchmark", action="store_true",
                         help="ベンチマークモード（19本の動画を処理）")
    args = parser.parse_args()

    detector = RealtimeBirdDetector()

    if args.benchmark:
        video_dir = "../data/youtube_greattit/videos"
        if not os.path.exists(video_dir):
            print(f"エラー: {video_dir} が見つかりません")
            return

        videos = sorted([f for f in os.listdir(video_dir)
                          if f.endswith(".mp4")])
        print(f"\nベンチマーク: {len(videos)} 動画\n")

        all_results = []
        total_start = time.time()

        for v in videos:
            path = os.path.join(video_dir, v)
            result = detector.process_video(
                path, display=False, skip_frames=args.skip,
                max_frames=args.max_frames,
            )
            if result is not None:
                all_results.append(result)

        total_elapsed = time.time() - total_start

        print(f"\n{'='*60}")
        print(f"ベンチマーク結果")
        print(f"{'='*60}")

        total_processed = sum(r["processed_frames"] for r in all_results)
        avg_fps_all = float(np.mean([r["avg_fps"] for r in all_results]))
        avg_yolo_all = float(np.mean([r["avg_yolo_ms"]
                                        for r in all_results]))
        avg_dinov2_all = float(np.mean([r["avg_dinov2_ms"]
                                          for r in all_results]))

        print(f"動画数: {len(all_results)}")
        print(f"総処理フレーム: {total_processed}")
        print(f"総処理時間: {total_elapsed:.1f} 秒")
        print(f"全体 FPS: {total_processed / total_elapsed:.1f}")
        print(f"平均 FPS（動画別平均）: {avg_fps_all:.1f}")
        print(f"YOLO 平均: {avg_yolo_all:.0f} ms")
        print(f"DINOv2 平均: {avg_dinov2_all:.0f} ms")

        print(f"\n動画別:")
        print(f"{'ファイル':<26} {'フレーム':>8} {'FPS':>6} "
              f"{'YOLO':>10} {'DINOv2':>10}")
        print("-" * 65)
        for r in all_results:
            name = os.path.basename(r["video_path"])[:25]
            print(f"{name:<26} {r['processed_frames']:>8} "
                  f"{r['avg_fps']:>5.1f} "
                  f"{r['avg_yolo_ms']:>7.0f} ms "
                  f"{r['avg_dinov2_ms']:>7.0f} ms")

        print(f"\n--- フレームスキップ別の実効 FPS 推定 ---")
        print(f"現在のスキップ: {args.skip}（{args.skip + 1} フレームに 1 回処理）")
        print(f"全フレーム処理時の推定 FPS: {avg_fps_all:.1f}")
        for skip in [0, 1, 2, 4, 9]:
            effective_fps = avg_fps_all * (skip + 1)
            print(f"  skip={skip}: 実効 {effective_fps:.0f} fps "
                  f"（{skip + 1} フレームに 1 回推論）")

        os.makedirs("../results/realtime", exist_ok=True)
        benchmark_result = {
            "total_videos": len(all_results),
            "total_frames": total_processed,
            "total_time_sec": total_elapsed,
            "overall_fps": total_processed / max(total_elapsed, 1e-9),
            "avg_fps": avg_fps_all,
            "avg_yolo_ms": avg_yolo_all,
            "avg_dinov2_ms": avg_dinov2_all,
            "skip_frames": args.skip,
            "device": detector.device,
            "per_video": [
                {
                    "video": os.path.basename(r["video_path"]),
                    "frames": r["processed_frames"],
                    "fps": r["avg_fps"],
                    "yolo_ms": r["avg_yolo_ms"],
                    "dinov2_ms": r["avg_dinov2_ms"],
                }
                for r in all_results
            ],
        }
        out_path = "../results/realtime/benchmark.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_result, f, indent=2, ensure_ascii=False)
        print(f"\n保存: {out_path}")

    else:
        # 単一動画モード
        video_path = args.video
        if not video_path:
            video_dir = "../data/youtube_greattit/videos"
            videos = sorted([f for f in os.listdir(video_dir)
                              if f.endswith(".mp4")])
            if videos:
                video_path = os.path.join(video_dir, videos[0])
                print(f"テスト動画: {video_path}")
            else:
                print("動画が見つかりません。パスを指定してください。")
                return

        detector.process_video(
            video_path,
            display=not args.no_display,
            save_path=args.save,
            skip_frames=args.skip,
            max_frames=args.max_frames,
        )


if __name__ == "__main__":
    main()
