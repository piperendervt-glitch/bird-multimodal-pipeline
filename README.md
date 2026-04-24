# Bird Multimodal Pipeline

マルチモーダル（映像 + 音声）による鳥種分類・検出パイプライン。

## 概要

DINOv2（画像特徴抽出）、YOLOv8（鳥検出）、BirdNET（音声分類）を統合し、
動画から鳥の種を分類するパイプラインを構築・検証。

## 主要な結果

- CUB-200-2011（200種静止画）: LogReg **87.31%**
- WetlandBirds（13種動画）: フレーム平均 **96.30%**、多数決 **100%**
- VB100（100種動画）: フレーム平均 **89.05%**（先行研究 84% を +5pp 上回る）
- Great Tit（725ソングタイプ）: LogReg **99.20%**

## セットアップ

```bash
pip install torch torchvision --break-system-packages
pip install scikit-learn opencv-python-headless birdnet librosa soundfile --break-system-packages
pip install ultralytics --break-system-packages
```

## データセット

- CUB-200-2011: https://data.caltech.edu/records/65de6-vp158
- WetlandBirds: https://zenodo.org/records/15696105
- VB100: https://zenodo.org/record/60375
- Great Tit Hits: https://osf.io/n8ac9/

## ライセンス

各データセットのライセンスに従ってください。
