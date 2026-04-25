"""カメラ映像の劣化検知モジュール。

劣化指標:
- 画像レベル: ぼけ（ラプラシアン分散）、露出（平均輝度）、コントラスト（輝度標準偏差）
- DINOv2 レベル: OOD スコアの時系列変化
- YOLO レベル: 検出数の急変、確信度の低下

最初の window_size フレームをキャリブレーション期間としてベースラインを学習し、
以降は閾値・時系列変化の双方で総合劣化スコアを返す。
"""

from collections import deque

import cv2
import numpy as np


class DegradationDetector:
    """カメラ映像の劣化をリアルタイムで検出する。"""

    def __init__(self, window_size=30):
        """
        window_size: 時系列監視のウィンドウサイズ（フレーム数）。
        """
        self.window_size = window_size
        self.history = deque(maxlen=window_size)

        # 閾値（必要に応じて外部から書き換え可能）
        self.thresholds = {
            "blur_low": 50,           # ラプラシアン分散がこれ以下 → ぼけ
            "brightness_low": 30,     # 平均輝度がこれ以下 → 暗すぎ
            "brightness_high": 225,   # 平均輝度がこれ以上 → 明るすぎ
            "contrast_low": 20,       # 輝度標準偏差がこれ以下 → 低コントラスト
            "ood_spike": 0.15,        # OOD スコアの急上昇量
            "detection_drop": 0.5,    # 検出数の急減割合
        }

        # 正常時のベースライン（キャリブレーション中は None）
        self.baseline = None
        self.calibrating = True
        self.calibration_data = []

    def compute(self, frame, detections=None, ood_scores=None):
        """1 フレームの劣化指標を計算する。

        Returns:
            features: dict 各指標の値
            alerts: list 警告メッセージ
            degradation_score: float（0=正常 〜 1=完全劣化）
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 画像品質指標
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = float(laplacian.var())
        brightness = float(gray.mean())
        contrast = float(gray.std())

        # 色彩指標（情報用、警告には使わない）
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation = float(hsv[:, :, 1].mean())

        # 検出品質指標
        n_detections = len(detections) if detections else 0
        if detections:
            confs = [d.get("confidence", d.get("yolo_confidence", 0))
                     for d in detections]
            mean_conf = float(np.mean(confs)) if confs else 0.0
        else:
            mean_conf = 0.0
        if ood_scores and len(ood_scores) > 0:
            mean_ood = float(np.mean(ood_scores))
        else:
            mean_ood = 1.0  # 検出なしは「OOD 全開」相当に扱う

        features = {
            "blur": blur_score,
            "brightness": brightness,
            "contrast": contrast,
            "saturation": saturation,
            "n_detections": n_detections,
            "mean_conf": mean_conf,
            "mean_ood": mean_ood,
        }

        self.history.append(features)

        # キャリブレーション（最初の window_size フレーム）
        if self.calibrating:
            self.calibration_data.append(features)
            if len(self.calibration_data) >= self.window_size:
                self._set_baseline()
                self.calibrating = False

        # ===== 劣化判定 =====
        alerts = []
        scores = []

        # 画像品質の絶対閾値
        if blur_score < self.thresholds["blur_low"]:
            alerts.append(f"ぼけ検出 (blur={blur_score:.0f})")
            scores.append(
                1.0 - min(blur_score / self.thresholds["blur_low"], 1.0)
            )

        if brightness < self.thresholds["brightness_low"]:
            alerts.append(f"暗すぎ (brightness={brightness:.0f})")
            scores.append(
                1.0 - brightness / self.thresholds["brightness_low"]
            )
        elif brightness > self.thresholds["brightness_high"]:
            alerts.append(f"明るすぎ (brightness={brightness:.0f})")
            scores.append(
                (brightness - self.thresholds["brightness_high"])
                / (255 - self.thresholds["brightness_high"])
            )

        if contrast < self.thresholds["contrast_low"]:
            alerts.append(f"低コントラスト (contrast={contrast:.0f})")
            scores.append(
                1.0 - contrast / self.thresholds["contrast_low"]
            )

        # 時系列変化（ベースライン設定後）
        if self.baseline and len(self.history) >= 10:
            recent = list(self.history)[-5:]

            # OOD スコアの急上昇
            recent_ood = float(np.mean([h["mean_ood"] for h in recent]))
            ood_change = recent_ood - self.baseline["mean_ood"]
            if ood_change > self.thresholds["ood_spike"]:
                alerts.append(f"OOD上昇 ({ood_change:+.3f})")
                scores.append(min(ood_change / 0.3, 1.0))

            # 検出数の急減
            recent_det = float(np.mean([h["n_detections"] for h in recent]))
            base_det = self.baseline["n_detections"]
            if (base_det > 0
                    and recent_det / max(base_det, 1)
                    < self.thresholds["detection_drop"]):
                alerts.append(f"検出数低下 ({recent_det:.1f}/{base_det:.1f})")
                scores.append(0.5)

            # ぼけの漸進的悪化
            recent_blur = float(np.mean([h["blur"] for h in recent]))
            base_blur = self.baseline["blur"]
            if base_blur > 0 and recent_blur / base_blur < 0.3:
                alerts.append(f"ぼけ悪化 ({recent_blur:.0f}/{base_blur:.0f})")
                scores.append(0.7)

        # 総合劣化スコア（0=正常, 1=劣化）
        degradation_score = float(max(scores)) if scores else 0.0
        return features, alerts, degradation_score

    def _set_baseline(self):
        """キャリブレーションデータの平均をベースラインに設定する。"""
        self.baseline = {}
        keys = self.calibration_data[0].keys()
        for key in keys:
            values = [d[key] for d in self.calibration_data]
            self.baseline[key] = float(np.mean(values))

        print(f"劣化検知ベースライン設定:")
        print(f"  blur={self.baseline['blur']:.0f}, "
              f"brightness={self.baseline['brightness']:.0f}, "
              f"contrast={self.baseline['contrast']:.0f}")

    def get_status_text(self, features, alerts, degradation_score):
        """表示用のステータステキスト（複数行）を返す。"""
        if degradation_score > 0.5:
            status = "DEGRADED"
        elif degradation_score > 0.2:
            status = "WARNING"
        elif self.calibrating:
            status = "CALIBRATING"
        else:
            status = "OK"

        lines = [
            f"Camera: {status} (deg={degradation_score:.2f})",
            (f"  blur={features['blur']:.0f} "
             f"brt={features['brightness']:.0f} "
             f"ctr={features['contrast']:.0f}"),
        ]
        for alert in alerts[:3]:
            lines.append(f"  ! {alert}")
        return lines
