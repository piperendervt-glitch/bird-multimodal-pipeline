# Synthetic Data Specification

## 目的

実データ（multimodal-aas-bird）の構造を最小限に抽象化した合成データを生成し、CAGL の効果が **データの構造的特徴のみから再現可能か** を検証する。

## データ構造

| Shape | 内容 |
|-------|------|
| `gt : (N, L)` | Ground truth (binary labels) |
| `pred : (N, K, L)` | K 個の topology 各々の binary 予測 |

| Parameter | Value | 意味 |
|-----------|-------|------|
| `n_samples` (N) | 500 | データ点数 |
| `n_topologies` (K) | 3 | Topology 数（audio / visual / fusion を模倣） |
| `n_labels` (L) | 2 | マルチラベル数 |
| `shared_noise_rate` | 0.1 | 全 topology に共通して flip を起こす確率（部分相関） |
| `seed` | `trial * 1000 + 42` | trial ごとに独立 |

## Topology 別 accuracy

```
DEFAULT_TOPO_ACCURACY = [
    [0.60, 0.85],   # Topology A (audio-like):  label 0 弱 / label 1 強
    [0.75, 0.70],   # Topology B (visual-like): 中庸
    [0.82, 0.88],   # Topology C (fusion-like): 全般に高精度
]
```

**設計意図:**

- **Topology 間の非対称性**: A は label 0 で弱く、B はバランス型、C は全般強。Weight が label 別に違う値を学習する余地を確保。
- **Label 間の非対称性**: label 0 と label 1 で難しさが異なる → Gate の寄与が出やすい。
- **Fusion topology が最強**: 実データと同じく、C が最も頼りになる → 学習後に C の `w*g` が支配的になることが期待。

## 生成手順

```python
gt[i, l] ~ Bernoulli(0.5)
for each (t, l):
    correct_i ~ Bernoulli(topo_accuracy[t][l])
    pred[i, t, l] = gt[i, l] if correct_i else 1 - gt[i, l]

# 共通ノイズ: 同一サンプル×同一ラベルで全 topology を一斉に flip
shared_noise[i, l] ~ Bernoulli(shared_noise_rate=0.1)
pred[i, t, l] = 1 - pred[i, t, l]   if shared_noise[i, l]
```

## 共通ノイズの役割

`shared_noise` は **部分相関** を再現するためのメカニズム。全 topology が同じ信号（例: センサー全体のノイズ）を受け取る状況を模擬し、consensus が崩れる局所事例を作る。

- `shared_noise_rate = 0` → 各 topology は完全に独立 → consensus が常に正しく、Gate 情報の価値が薄い
- `shared_noise_rate = 0.1` → 10% のサンプルで全員が間違う → consensus-agreement が非自明な情報を持つ

## Train / Eval 分割

- 前半 `N/2 = 250` サンプル → 学習
- 後半 `N/2 = 250` サンプル → 評価（macro F1）

## 評価指標

各 label ごとに F1 を計算し、label 平均 (macro) を取る:

```
macro_F1 = mean_over_labels(f1_score(gt_eval[:, l], pred[:, l]))
```

## 再現性

`seed = trial * 1000 + 42` により各 trial は完全に決定論的。20 trials でペアード統計を計算する。
