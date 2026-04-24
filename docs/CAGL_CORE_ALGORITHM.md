# CAGL Core Algorithm

## 概要

**CAGL (Consensus-Agreement Gate Learning)** は複数の classifier (topology) の予測を統合するオンライン学習アルゴリズム。各 topology × label ごとに2つのスカラー変数 `w`（Weight）と `g`（Gate）を保持し、multiplicative に統合する。

## 変数

| 変数 | 形状 | 初期値 | 意味 |
|------|------|--------|------|
| `w[t][l]` | (K, L) | 0.5 | Topology `t` が label `l` で **正しい** 確率の推定値（GT signal で学習） |
| `g[t][l]` | (K, L) | 0.5 | Topology `t` が label `l` で **consensus に従う** 度合い（consensus signal で学習） |

## Predict

単一データ点の統合予測を返す。

```
effective[t, l] = 
    w[t, l] * g[t, l]    if weight_mode == "multiplicative"
    w[t, l]              if weight_mode == "weight_only"
    0.5 * g[t, l]        if weight_mode == "gate_only"

score[l]  = Σ_t (effective[t, l] * pred[t, l]) / Σ_t effective[t, l]
final[l]  = 1 if score[l] >= 0.5 else 0
```

- 分母 `Σ_t effective[t, l]` がゼロに近い場合は `final[l] = 0`。

## Update

学習は2つの信号で並行に行う:

### Weight 更新（GT signal, 常に実行）
```
if pred[t, l] == gt[l]:
    w[t, l] += 0.1 * (1 - w[t, l])   # success_rate=0.1
else:
    w[t, l] *= 0.7                     # failure_rate=0.7
```

### Gate 更新（`gate_mode` に依存）

| `gate_mode` | 判定条件 | 意味 |
|-------------|----------|------|
| `"consensus"` | `pred[t, l] == final[l]` | **CAGL本体**: 統合予測との一致で学習 |
| `"gt"` | `pred[t, l] == gt[l]` | Ablation: GT signal で学習（Weightと冗長） |
| `"none"` | なし | Gate を固定（純粋に Weight のみ） |

成功/失敗時の更新式は Weight と同じ (`success_rate=0.1`, `failure_rate=0.7`)。

## Load-bearing novelty

**Weight は「各 topology が正しいか」を学び、Gate は「consensus に従うか」を学ぶ。**

この2つは **異なる情報** を encode しているため、multiplicative に組み合わせると相乗効果を生む:

- ある topology が **正しくて、かつ consensus に同調する**（`w*g` が高い）→ 強く寄与
- 正しいが consensus から外れる（`w` 高、`g` 低）→ 適度に補正
- 間違うが consensus に従う（`w` 低、`g` 高）→ 適度に抑制
- 正しくもなく consensus からも外れる（両方低）→ 大幅に抑制

Gate を GT signal で更新すると、Weight と同じ信号を学ぶため **冗長化** し、相乗効果が失われる（V3 の期待結果）。

## ハイパーパラメータ

| Parameter | Value | 意味 |
|-----------|-------|------|
| `success_rate` | 0.1 | 正解時の learning rate（`1 - w` に向けて引き寄せ） |
| `failure_rate` | 0.7 | 不正解時の減衰係数（掛け算） |
| Weight/Gate 初期値 | 0.5 | 中立 |

## 参考

元の研究: Phase 1.4d Stage 3 Target C で発見された consensus-agreement gate update rule。
