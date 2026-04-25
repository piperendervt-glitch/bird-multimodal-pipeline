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

from degradation_detector import DegradationDetector


class SimpleTracker:
    """SAHI 検出結果に対する簡易 IoU ベーストラッカー。

    Ultralytics の model.track() は内部で検出と追跡を一体化しているため
    SAHI の検出結果を直接渡せない。代わりに本クラスで IoU 関連付けを行う。

    - 既存トラックと新検出の IoU を貪欲法でマッチング
    - 一致しない検出は新 ID を割り当て
    - 一致しないトラックは lost カウントを増やし max_lost で破棄
    """

    def __init__(self, iou_threshold=0.3, max_lost=30):
        self.tracks = {}     # {track_id: {"bbox": [...], "lost": 0}}
        self.next_id = 1
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost

    @staticmethod
    def _compute_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / max(union, 1e-9)

    def update(self, detections):
        """検出リストに track_id を付与して返す。

        detections: [{"bbox": [x1,y1,x2,y2], "confidence": float, ...}, ...]
        """
        # 検出ゼロのフレーム: 全トラックの lost を増やして古いものを破棄
        if not detections:
            for tid in list(self.tracks.keys()):
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"] > self.max_lost:
                    del self.tracks[tid]
            return []

        track_ids = list(self.tracks.keys())

        # 既存トラックなし → 全検出に新 ID
        if not track_ids:
            for det in detections:
                det["track_id"] = self.next_id
                self.tracks[self.next_id] = {
                    "bbox": det["bbox"], "lost": 0,
                }
                self.next_id += 1
            return detections

        # 全ペアの IoU を計算
        scores = []
        for di, det in enumerate(detections):
            for tid in track_ids:
                iou = self._compute_iou(det["bbox"],
                                          self.tracks[tid]["bbox"])
                if iou >= self.iou_threshold:
                    scores.append((iou, di, tid))

        # 高 IoU 順に貪欲マッチング
        scores.sort(reverse=True)
        matched_dets = set()
        matched_tracks = set()
        for iou, di, tid in scores:
            if di in matched_dets or tid in matched_tracks:
                continue
            detections[di]["track_id"] = tid
            self.tracks[tid] = {"bbox": detections[di]["bbox"], "lost": 0}
            matched_dets.add(di)
            matched_tracks.add(tid)

        # マッチしなかった検出 → 新 ID
        for di, det in enumerate(detections):
            if di not in matched_dets:
                det["track_id"] = self.next_id
                self.tracks[self.next_id] = {
                    "bbox": det["bbox"], "lost": 0,
                }
                self.next_id += 1

        # マッチしなかったトラック → lost カウント増加 / 古いものは破棄
        for tid in track_ids:
            if tid not in matched_tracks:
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"] > self.max_lost:
                    del self.tracks[tid]

        return detections


class CosineSimilarityOOD:
    """コサイン類似度ベースの OOD 検出器。

    CUB-200 の DINOv2 特徴量からクラスごとの L2 正規化プロトタイプを構築し
    入力特徴量と最も近いプロトタイプとの「1 - 類似度」を OOD スコアとする。
    高い = 既知の鳥分布から離れている = 誤検出（背景・枝・岩等）の可能性。
    """

    def __init__(self):
        self.prototypes = None  # (n_classes, feat_dim)
        self.enabled = False

    def fit(self, X, y):
        """各クラスの L2 正規化プロトタイプを計算する。"""
        classes = np.unique(y)
        protos = []
        for cls in classes:
            mask = y == cls
            proto = X[mask].mean(axis=0)
            proto = proto / (np.linalg.norm(proto) + 1e-9)
            protos.append(proto)
        self.prototypes = np.array(protos)
        self.enabled = True
        print(f"OOD フィルタ: {len(protos)} クラスのプロトタイプを構築")

    def score(self, features):
        """1 サンプルの OOD スコアを返す（高い = OOD）。"""
        if not self.enabled or features is None:
            return 0.0
        feat_norm = features / (np.linalg.norm(features) + 1e-9)
        sims = feat_norm @ self.prototypes.T
        return float(1.0 - sims.max())

    def score_batch(self, features_list):
        """複数サンプルの OOD スコアをまとめて返す。

        features_list の None 要素は OOD スコア 1.0（最大値）として扱う。
        """
        out = []
        for f in features_list:
            if f is None:
                out.append(1.0)
            else:
                out.append(self.score(f))
        return out


class RealtimeBirdDetector:
    """リアルタイム鳥検出・分類パイプライン。"""

    def __init__(self, species_mapping_path=None, classifier_data_path=None,
                 tracker="bytetrack", use_sahi=False, slice_size=320,
                 sahi_track=False, ood_filter=True, ood_threshold=0.71):
        print("=== リアルタイム鳥検出システム 初期化 ===")
        self.tracker = tracker  # "bytetrack" / "botsort" / "none"
        self.use_sahi = use_sahi
        self.slice_size = slice_size
        self.sahi_track = sahi_track  # SAHI + 簡易追跡モード
        if use_sahi:
            mode = "SAHI + 簡易追跡" if sahi_track else "SAHI のみ"
            print(f"SAHI モード: 有効 ({mode}, slice_size={slice_size})")

        # SAHI 用の簡易トラッカー（--sahi-track 時のみ使用）
        self.simple_tracker = (
            SimpleTracker(iou_threshold=0.3, max_lost=30)
            if sahi_track else None
        )

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

        # OOD フィルタ（CUB-200 プロトタイプによるコサイン類似度）
        self.ood = CosineSimilarityOOD()
        self.ood_threshold = ood_threshold
        self.ood_filtered_count = 0  # OOD で除去された累計検出数
        self.ood_total_count = 0     # OOD 評価対象の累計検出数
        if ood_filter:
            self._init_ood_filter()

        # カメラ映像の劣化検知（カメラモードでのみ更新される）
        self.degradation = DegradationDetector(window_size=30)
        self.last_degradation = None

        print(f"デバイス: {self.device}")
        print("初期化完了\n")

    def _init_ood_filter(self):
        """CUB-200 の特徴量を読み込んで OOD フィルタを初期化する。"""
        candidates = [
            "../results/bird_phase1/features_dinov2_vits14.npz",
            "../results/bird_phase1/features.npz",
        ]
        for path in candidates:
            if not os.path.exists(path):
                continue
            try:
                data = np.load(path, allow_pickle=True)
            except Exception:
                continue
            keys = list(data.keys())
            if "X_train" in keys:
                X = data["X_train"]
                y = data["y_train"]
            elif "X" in keys:
                X = data["X"]
                y = data["y"]
            else:
                X = data[keys[0]]
                y = (data[keys[1]] if len(keys) > 1
                     else np.zeros(len(X)))
            self.ood.fit(X, y)
            return

        print("警告: CUB-200 特徴量が見つかりません。OOD フィルタは無効。")

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

    def detect_birds(self, frame, use_tracker=True):
        """1 フレームで鳥クラスを検出（必要なら追跡 ID も付与）する。"""
        # SAHI モードはタイル分割推論。Ultralytics の tracker と同時併用不可。
        if self.use_sahi:
            return self._detect_birds_sahi(frame)

        if use_tracker and self.tracker != "none":
            tracker_yaml = f"{self.tracker}.yaml"
            results = self.yolo.track(frame, verbose=False, conf=0.15,
                                       persist=True, tracker=tracker_yaml)
        else:
            results = self.yolo(frame, verbose=False, conf=0.15)

        detections = []
        for box in results[0].boxes:
            if int(box.cls[0]) == self.BIRD_CLASS_ID:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])

                # トラッカー ID（未確定の場合 -1）
                track_id = -1
                if box.id is not None:
                    track_id = int(box.id[0])

                detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": conf,
                    "track_id": track_id,
                })
        return detections

    def _detect_birds_sahi(self, frame):
        """SAHI によるタイル分割推論。小鳥の検出精度を上げる。

        - slice_size × slice_size のタイルに分割し各タイルで YOLO 推論
        - タイル間で重なり率 0.2 を確保し境界の鳥も検出
        - 結果はタイル境界をまたぐ重複を NMS で統合
        """
        from sahi.predict import get_sliced_prediction
        from sahi import AutoDetectionModel

        if not hasattr(self, "_sahi_model"):
            self._sahi_model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path="yolov8n.pt",
                confidence_threshold=0.25,
                device=self.device,
            )

        result = get_sliced_prediction(
            frame,
            self._sahi_model,
            slice_height=self.slice_size,
            slice_width=self.slice_size,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            verbose=0,
        )

        detections = []
        for pred in result.object_prediction_list:
            if pred.category.id == self.BIRD_CLASS_ID:
                bbox = pred.bbox
                x1 = int(bbox.minx)
                y1 = int(bbox.miny)
                x2 = int(bbox.maxx)
                y2 = int(bbox.maxy)
                conf = float(pred.score.value)
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "track_id": -1,  # 既定は追跡なし
                })

        # SAHI + 簡易追跡モード時は IoU で ID を付与
        if self.simple_tracker is not None:
            detections = self.simple_tracker.update(detections)

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

        # OOD スコアの計算（特徴量のリストに対してまとめて）
        if self.ood.enabled and detections:
            ood_scores = self.ood.score_batch(features_list)
        else:
            ood_scores = [0.0] * len(detections)

        results = []
        for det, feat, ood_score in zip(detections, features_list,
                                          ood_scores):
            # OOD フィルタ: 閾値超過の検出は破棄して描画対象外とする
            if self.ood.enabled:
                self.ood_total_count += 1
                if ood_score > self.ood_threshold:
                    self.ood_filtered_count += 1
                    continue

            results.append({
                "bbox": det["bbox"],
                "yolo_confidence": det["confidence"],
                "track_id": det.get("track_id", -1),
                "features": feat,
                "species": "Unknown",
                "species_confidence": 0.0,
                "ood_score": ood_score,
            })

        # 劣化検知（OOD スコア + 検出情報を渡す）
        ood_scores_list = [r.get("ood_score", 0.0) for r in results]
        deg_features, deg_alerts, deg_score = self.degradation.compute(
            frame, detections, ood_scores_list
        )
        self.last_degradation = {
            "features": deg_features,
            "alerts": deg_alerts,
            "score": deg_score,
        }

        frame_elapsed = time.time() - frame_start
        self.fps_history.append(1.0 / max(frame_elapsed, 1e-9))
        return results

    # ID 別の色パレット（draw_results 用）
    COLORS = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (0, 255, 255), (255, 0, 255), (128, 255, 0), (0, 128, 255),
        (255, 128, 0), (128, 0, 255), (0, 255, 128), (255, 0, 128),
    ]

    def draw_results(self, frame, results):
        """検出結果を描画する。

        - ボックス上のラベルは ID のみ（短く）
        - 左上に半透明パネルで FPS と鳥の詳細情報を集約表示
        - ID ごとに色分け
        """
        display = frame.copy()

        # バウンディングボックス + 短いラベルのみ
        for r in results:
            x1, y1, x2, y2 = r["bbox"]
            track_id = r.get("track_id", -1)

            color = (self.COLORS[track_id % len(self.COLORS)]
                     if track_id >= 0 else (0, 255, 0))

            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

            label = f"#{track_id}" if track_id >= 0 else "?"

            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(display, (x1, y1 - th - 6),
                           (x1 + tw + 4, y1), color, -1)
            cv2.putText(display, label, (x1 + 2, y1 - 3),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # 左上の情報パネル
        info_lines = []

        if self.fps_history:
            avg_fps = np.mean(list(self.fps_history))
            avg_yolo = np.mean(list(self.yolo_time)) * 1000
            avg_dinov2 = np.mean(list(self.dinov2_time)) * 1000
            active_ids = {r.get("track_id", -1) for r in results
                          if r.get("track_id", -1) >= 0}

            info_lines.append(
                f"FPS: {avg_fps:.1f}  YOLO: {avg_yolo:.0f}ms  "
                f"DINOv2: {avg_dinov2:.0f}ms"
            )
            info_lines.append(
                f"Birds: {len(results)}  Active IDs: {len(active_ids)}"
            )

        if results:
            info_lines.append("---")
            for r in results:
                track_id = r.get("track_id", -1)
                yolo_conf = r["yolo_confidence"]
                ood_score = r.get("ood_score", 0.0)
                species = r.get("species", "Unknown")

                ood_str = (f" ood {ood_score:.2f}"
                           if self.ood.enabled else "")

                if track_id >= 0:
                    if species != "Unknown":
                        info_lines.append(
                            f"  #{track_id}: {species} "
                            f"({yolo_conf:.2f}){ood_str}"
                        )
                    else:
                        info_lines.append(
                            f"  #{track_id}: conf {yolo_conf:.2f}"
                            f"{ood_str}"
                        )
                else:
                    info_lines.append(
                        f"  Bird: conf {yolo_conf:.2f}{ood_str}"
                    )

        # 劣化検知情報（process_frame で last_degradation がセットされていれば）
        panel_color = (0, 0, 0)  # 既定は黒
        if self.last_degradation is not None:
            deg = self.last_degradation
            info_lines.append("---")
            info_lines.extend(self.degradation.get_status_text(
                deg["features"], deg["alerts"], deg["score"]
            ))
            # 劣化レベルに応じて背景色を変える
            if deg["score"] > 0.5:
                panel_color = (0, 0, 100)   # 暗い赤（DEGRADED）
            elif deg["score"] > 0.2:
                panel_color = (0, 50, 100)  # 暗いオレンジ（WARNING）

        if info_lines:
            line_height = 20
            panel_height = len(info_lines) * line_height + 16
            panel_width = 380

            # 半透明背景（劣化レベルで色を変える）
            overlay = display.copy()
            cv2.rectangle(overlay, (5, 5),
                           (5 + panel_width, 5 + panel_height),
                           panel_color, -1)
            cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)

            # 白文字でテキスト描画
            for i, line in enumerate(info_lines):
                y = 22 + i * line_height
                cv2.putText(display, line, (12, y),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                             (255, 255, 255), 1)

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

        # この動画分の OOD カウンタを記録するため、開始時の累計を保存
        ood_filtered_start = self.ood_filtered_count
        ood_total_start = self.ood_total_count

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
                         "confidence": r["yolo_confidence"],
                         "track_id": r.get("track_id", -1)}
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

        # 検出統計（SAHI 比較用）
        total_detections = sum(d["n_birds"] for d in all_detections)
        frames_with_bird = sum(d["n_birds"] > 0 for d in all_detections)
        max_birds_per_frame = (max(d["n_birds"] for d in all_detections)
                                if all_detections else 0)

        print(f"\n=== 処理サマリー ===")
        print(f"処理フレーム: {processed_count} / {frame_count}")
        print(f"平均 FPS: {avg_fps:.1f}")
        print(f"YOLO 平均: {avg_yolo:.0f} ms")
        print(f"DINOv2 平均: {avg_dinov2:.0f} ms")
        print(f"検出イベント（鳥が映ったフレーム数）: {frames_with_bird}")
        print(f"総検出数（全フレームでの鳥の延べ数）: {total_detections}")
        print(f"1 フレーム最大検出数: {max_birds_per_frame}")

        # 追跡統計（ID 別の出現フレーム数）
        track_frames = {}
        for d in all_detections:
            for det in d["detections"]:
                tid = det.get("track_id", -1)
                if tid >= 0:
                    track_frames[tid] = track_frames.get(tid, 0) + 1

        unique_ids = len(track_frames)
        max_track_frames = max(track_frames.values()) if track_frames else 0
        mean_track_frames = (float(np.mean(list(track_frames.values())))
                              if track_frames else 0.0)

        if self.tracker != "none" or self.simple_tracker is not None:
            print(f"ユニーク ID 数: {unique_ids}")
            print(f"最長追跡フレーム数: {max_track_frames}")
            print(f"平均追跡フレーム数: {mean_track_frames:.1f}")

        # この動画分の OOD 統計
        ood_filtered = self.ood_filtered_count - ood_filtered_start
        ood_total = self.ood_total_count - ood_total_start
        if self.ood.enabled:
            ood_rate = ood_filtered / max(ood_total, 1)
            print(f"OOD 除去: {ood_filtered}/{ood_total} "
                  f"({ood_rate*100:.1f}%)")

        return {
            "video_path": video_path,
            "total_frames": frame_count,
            "processed_frames": processed_count,
            "avg_fps": avg_fps,
            "avg_yolo_ms": avg_yolo,
            "avg_dinov2_ms": avg_dinov2,
            "total_detections": int(total_detections),
            "frames_with_bird": int(frames_with_bird),
            "max_birds_per_frame": int(max_birds_per_frame),
            "ood_filtered": int(ood_filtered),
            "ood_total": int(ood_total),
            "tracking": {
                "unique_ids": unique_ids,
                "max_track_frames": int(max_track_frames),
                "mean_track_frames": mean_track_frames,
                "id_frame_counts": {str(k): int(v)
                                     for k, v in track_frames.items()},
            },
            "detections": all_detections,
        }

    def process_camera(self, camera_id=0, save_path=None):
        """USB カメラからのリアルタイム推論プレビュー。

        操作:
          q: 終了
          p: 一時停止 / 再開
          s: スクリーンショットを保存
          o: OOD フィルタ ON/OFF 切替
        """
        print(f"\nUSB カメラ起動: デバイス {camera_id}")

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"エラー: カメラ {camera_id} を開けません")
            print("利用可能なカメラを確認してください")
            return

        # 720p を試みる
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"  解像度: {width}x{height}")
        print(f"  カメラ FPS: {fps}")
        print(f"  OOD フィルタ: "
              f"{'有効' if self.ood.enabled else '無効'}")
        print(f"  トラッカー: {self.tracker}")
        print(f"\n操作:")
        print(f"  q: 終了")
        print(f"  p: 一時停止/再開")
        print(f"  s: スクリーンショット保存")
        print(f"  o: OOD フィルタ ON/OFF 切替")

        writer = None
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(save_path, fourcc, 30,
                                       (width, height))
            print(f"  録画先: {save_path}")

        frame_count = 0
        screenshot_count = 0
        paused = False

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("カメラからフレームを取得できません")
                    break

                frame_count += 1

                # 推論
                results = self.process_frame(frame)

                # 描画
                display = self.draw_results(frame, results)

                # 左下にカメラモード情報を上書き表示
                mode_text = (f"CAMERA {camera_id} | "
                             f"Frame {frame_count}")
                cv2.putText(display, mode_text, (10, height - 15),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                              (0, 255, 255), 1)

                cv2.imshow("Bird Detection - Camera", display)

                if writer:
                    writer.write(display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print(f"\n終了（{frame_count} フレーム処理）")
                break
            elif key == ord("p"):
                paused = not paused
                print(f"{'一時停止' if paused else '再開'}")
            elif key == ord("s"):
                screenshot_count += 1
                ss_path = (
                    f"../results/realtime/"
                    f"screenshot_{screenshot_count:04d}.jpg"
                )
                os.makedirs("../results/realtime", exist_ok=True)
                cv2.imwrite(ss_path, display)
                print(f"スクリーンショット保存: {ss_path}")
            elif key == ord("o"):
                self.ood.enabled = not self.ood.enabled
                print(f"OOD フィルタ: "
                      f"{'ON' if self.ood.enabled else 'OFF'}")

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        # サマリー
        if self.fps_history:
            avg_fps = float(np.mean(list(self.fps_history)))
            print(f"\n=== カメラセッション サマリー ===")
            print(f"総フレーム: {frame_count}")
            print(f"平均 FPS: {avg_fps:.1f}")
            if self.ood_total_count > 0:
                rate = (self.ood_filtered_count
                        / max(self.ood_total_count, 1))
                print(f"OOD 除去: "
                      f"{self.ood_filtered_count}/"
                      f"{self.ood_total_count} ({rate*100:.1f}%)")

    def process_dual_camera(self, cam_a_id=0, cam_b_id=1):
        """2 台カメラの同時表示・劣化比較。

        - 両カメラを 640x480 で同時キャプチャ
        - 各フレームで YOLO + DINOv2 + OOD + 追跡を実行
        - カメラごとに独立した DegradationDetector で劣化スコアを算出
        - 横並び合成 + 下部に劣化比較バーを描画
        操作: q=終了, p=一時停止, s=スクリーンショット
        """
        print(f"\nデュアルカメラ起動: A={cam_a_id}, B={cam_b_id}")

        cap_a = cv2.VideoCapture(cam_a_id)
        cap_b = cv2.VideoCapture(cam_b_id)

        if not cap_a.isOpened():
            print(f"エラー: カメラ A ({cam_a_id}) を開けません")
            return
        if not cap_b.isOpened():
            print(f"エラー: カメラ B ({cam_b_id}) を開けません")
            cap_a.release()
            return

        for cap in (cap_a, cap_b):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        w_a = int(cap_a.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_a = int(cap_a.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w_b = int(cap_b.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_b = int(cap_b.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"  カメラ A: {w_a}x{h_a}")
        print(f"  カメラ B: {w_b}x{h_b}")
        print(f"\n操作: q=終了, p=一時停止, s=スクリーンショット")

        # カメラ B 用の独立した劣化検知器
        degradation_b = DegradationDetector(window_size=30)
        # カメラ A 用には self.degradation を流用するが、カメラ B 用は手動で動かす

        frame_count = 0
        paused = False

        while True:
            if not paused:
                ret_a, frame_a = cap_a.read()
                ret_b, frame_b = cap_b.read()
                if not ret_a or not ret_b:
                    print("カメラからフレームを取得できません")
                    break

                frame_count += 1

                # カメラ A: process_frame で内部の self.degradation が更新される
                results_a = self.process_frame(frame_a)
                deg_a = self.last_degradation
                display_a = self.draw_results(frame_a, results_a)
                cv2.putText(display_a,
                              f"CAM A ({cam_a_id}) Frame {frame_count}",
                              (10, h_a - 15),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                              (0, 255, 255), 1)

                # カメラ B: 検出は process_frame で行うが、劣化は別検知器で計算
                results_b = self.process_frame(frame_b)
                ood_scores_b = [r.get("ood_score", 0.0) for r in results_b]
                detections_b = [{"confidence": r["yolo_confidence"]}
                                for r in results_b]
                deg_features_b, deg_alerts_b, deg_score_b = (
                    degradation_b.compute(
                        frame_b, detections_b, ood_scores_b
                    )
                )

                # draw_results が参照する last_degradation を一時的に B に切替
                saved = self.last_degradation
                saved_engine = self.degradation
                self.last_degradation = {
                    "features": deg_features_b,
                    "alerts": deg_alerts_b,
                    "score": deg_score_b,
                }
                self.degradation = degradation_b
                display_b = self.draw_results(frame_b, results_b)
                self.last_degradation = saved
                self.degradation = saved_engine

                cv2.putText(display_b,
                              f"CAM B ({cam_b_id}) Frame {frame_count}",
                              (10, h_b - 15),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                              (0, 255, 255), 1)

                # 高さを揃えて横並び結合
                target_h = max(h_a, h_b)
                if h_a != target_h:
                    new_w_a = int(w_a * target_h / h_a)
                    display_a = cv2.resize(display_a, (new_w_a, target_h))
                if h_b != target_h:
                    new_w_b = int(w_b * target_h / h_b)
                    display_b = cv2.resize(display_b, (new_w_b, target_h))
                combined = np.hstack([display_a, display_b])

                # 下部に劣化比較バー
                bar_h = 30
                bar = np.zeros((bar_h, combined.shape[1], 3),
                                dtype=np.uint8)

                deg_a_score = deg_a["score"] if deg_a else 0.0
                deg_b_score = deg_score_b

                def _color(s):
                    if s < 0.2:
                        return (0, 255, 0)     # 緑
                    if s < 0.5:
                        return (0, 200, 255)   # オレンジ
                    return (0, 0, 255)         # 赤

                a_color = _color(deg_a_score)
                b_color = _color(deg_b_score)

                mid = combined.shape[1] // 2
                cv2.putText(bar, f"A: deg={deg_a_score:.2f}", (10, 22),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, a_color, 1)
                cv2.putText(bar, f"B: deg={deg_b_score:.2f}",
                              (mid + 10, 22),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, b_color, 1)

                if deg_a_score < deg_b_score:
                    label = "A > B"
                elif deg_b_score < deg_a_score:
                    label = "B > A"
                else:
                    label = "A = B"
                cv2.putText(bar, label, (mid - 40, 22),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                              (255, 255, 255), 1)

                combined = np.vstack([combined, bar])
                cv2.imshow("Dual Camera - Degradation Detection",
                            combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("p"):
                paused = not paused
                print(f"{'一時停止' if paused else '再開'}")
            elif key == ord("s") and not paused:
                ss_path = (
                    f"../results/realtime/"
                    f"dual_screenshot_{frame_count:06d}.jpg"
                )
                os.makedirs("../results/realtime", exist_ok=True)
                cv2.imwrite(ss_path, combined)
                print(f"スクリーンショット: {ss_path}")

        cap_a.release()
        cap_b.release()
        cv2.destroyAllWindows()
        print(f"\n=== デュアルカメラ サマリー ===")
        print(f"総フレーム: {frame_count}")


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
    parser.add_argument("--tracker", type=str, default="bytetrack",
                         choices=["bytetrack", "botsort", "none"],
                         help="トラッカー（bytetrack/botsort/none）")
    parser.add_argument("--sahi", action="store_true",
                         help="SAHI タイル分割推論を有効化（小鳥検出改善）")
    parser.add_argument("--sahi-track", action="store_true",
                         help="SAHI + 簡易 IoU 追跡（小鳥検出 + ID 付与）")
    parser.add_argument("--slice-size", type=int, default=320,
                         help="SAHI のスライスサイズ（デフォルト: 320）")
    parser.add_argument("--no-ood", action="store_true",
                         help="OOD フィルタを無効化")
    parser.add_argument("--ood-threshold", type=float, default=0.71,
                         help="OOD 除去閾値（デフォルト: 0.71、90%%ile）")
    parser.add_argument("--camera", type=int, default=None,
                         help="USB カメラのデバイス番号（例: 0）")
    parser.add_argument("--dual-camera", nargs=2, type=int, default=None,
                         metavar=("CAM_A", "CAM_B"),
                         help="2 台カメラ同時表示（例: --dual-camera 0 1）")
    args = parser.parse_args()

    # --sahi-track は SAHI を有効化したうえで簡易トラッカーを使う
    if args.sahi_track:
        args.sahi = True

    # SAHI モード時は Ultralytics 内蔵トラッカーは併用不可
    if args.sahi and args.tracker != "none":
        print(f"注意: SAHI モードでは Ultralytics 追跡が使えません。"
              f"tracker={args.tracker} → none に変更します。")
        args.tracker = "none"

    detector = RealtimeBirdDetector(
        tracker=args.tracker,
        use_sahi=args.sahi,
        slice_size=args.slice_size,
        sahi_track=args.sahi_track,
        ood_filter=not args.no_ood,
        ood_threshold=args.ood_threshold,
    )

    # デュアルカメラモード（最優先）
    if args.dual_camera is not None:
        detector.process_dual_camera(
            cam_a_id=args.dual_camera[0],
            cam_b_id=args.dual_camera[1],
        )
        return

    # 単一カメラモード（次に優先）
    if args.camera is not None:
        detector.process_camera(
            camera_id=args.camera,
            save_path=args.save,
        )
        return

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

        total_det_all = sum(r.get("total_detections", 0) for r in all_results)
        frames_with_bird_all = sum(r.get("frames_with_bird", 0)
                                     for r in all_results)
        ood_filtered_all = sum(r.get("ood_filtered", 0)
                                 for r in all_results)
        ood_total_all = sum(r.get("ood_total", 0) for r in all_results)

        print(f"動画数: {len(all_results)}")
        print(f"総処理フレーム: {total_processed}")
        print(f"総処理時間: {total_elapsed:.1f} 秒")
        print(f"全体 FPS: {total_processed / total_elapsed:.1f}")
        print(f"平均 FPS（動画別平均）: {avg_fps_all:.1f}")
        print(f"YOLO 平均: {avg_yolo_all:.0f} ms")
        print(f"DINOv2 平均: {avg_dinov2_all:.0f} ms")
        print(f"総検出数（鳥の延べ数、OOD 除去後）: {total_det_all}")
        print(f"鳥が映ったフレーム合計: {frames_with_bird_all}")
        if ood_total_all > 0:
            ood_rate = ood_filtered_all / ood_total_all
            print(f"OOD 除去合計: {ood_filtered_all}/{ood_total_all} "
                  f"({ood_rate*100:.1f}%, 閾値 "
                  f"{detector.ood_threshold:.2f})")

        print(f"\n動画別:")
        print(f"{'ファイル':<26} {'フレーム':>8} {'FPS':>6} "
              f"{'YOLO':>10} {'DINOv2':>10} {'検出':>6} "
              f"{'IDs':>5} {'最長':>5} {'平均':>7}")
        print("-" * 100)
        for r in all_results:
            name = os.path.basename(r["video_path"])[:25]
            tr = r.get("tracking", {})
            print(f"{name:<26} {r['processed_frames']:>8} "
                  f"{r['avg_fps']:>5.1f} "
                  f"{r['avg_yolo_ms']:>7.0f} ms "
                  f"{r['avg_dinov2_ms']:>7.0f} ms "
                  f"{r.get('total_detections', 0):>6} "
                  f"{tr.get('unique_ids', 0):>5} "
                  f"{tr.get('max_track_frames', 0):>5} "
                  f"{tr.get('mean_track_frames', 0):>7.1f}")

        # 追跡サマリー（Ultralytics トラッカー or SAHI 簡易追跡が有効な時）
        has_tracking = (args.tracker != "none") or args.sahi_track
        if has_tracking:
            total_unique = sum(r.get("tracking", {}).get("unique_ids", 0)
                                for r in all_results)
            total_max = (max(r.get("tracking", {}).get("max_track_frames", 0)
                              for r in all_results)
                         if all_results else 0)
            avg_unique = (total_unique / len(all_results)
                          if all_results else 0)
            tracker_name = ("simple_tracker (SAHI)"
                            if args.sahi_track else args.tracker)
            print(f"\n--- 追跡サマリー ---")
            print(f"トラッカー: {tracker_name}")
            print(f"全動画でのユニーク ID 合計: {total_unique}")
            print(f"動画別 平均ユニーク ID 数: {avg_unique:.1f}")
            print(f"全動画での最長追跡フレーム数: {total_max}")

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
            "total_detections": int(total_det_all),
            "frames_with_bird": int(frames_with_bird_all),
            "ood_filtered": int(ood_filtered_all),
            "ood_total": int(ood_total_all),
            "ood_threshold": float(detector.ood_threshold),
            "ood_enabled": bool(detector.ood.enabled),
            "skip_frames": args.skip,
            "tracker": args.tracker,
            "use_sahi": args.sahi,
            "slice_size": args.slice_size if args.sahi else None,
            "device": detector.device,
            "per_video": [
                {
                    "video": os.path.basename(r["video_path"]),
                    "frames": r["processed_frames"],
                    "fps": r["avg_fps"],
                    "yolo_ms": r["avg_yolo_ms"],
                    "dinov2_ms": r["avg_dinov2_ms"],
                    "total_detections": r.get("total_detections", 0),
                    "frames_with_bird": r.get("frames_with_bird", 0),
                    "max_birds_per_frame": r.get("max_birds_per_frame", 0),
                    "ood_filtered": r.get("ood_filtered", 0),
                    "ood_total": r.get("ood_total", 0),
                    "unique_ids": r.get("tracking", {}).get("unique_ids", 0),
                    "max_track_frames":
                        r.get("tracking", {}).get("max_track_frames", 0),
                    "mean_track_frames":
                        r.get("tracking", {}).get("mean_track_frames", 0.0),
                }
                for r in all_results
            ],
        }

        # ファイル名を手法別に分けて上書き衝突を防ぐ
        if args.sahi_track:
            suffix = f"sahi_track_{args.slice_size}"
        elif args.sahi:
            suffix = f"sahi_{args.slice_size}"
        else:
            suffix = args.tracker  # bytetrack / botsort / none
        out_path = f"../results/realtime/benchmark_{suffix}.json"
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
