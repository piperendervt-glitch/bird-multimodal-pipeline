# Ablation Variants

## 4 Variants

CAGL の **Weight** と **Gate** のどの組み合わせが効果を生むかを切り分けるため、4 variant を比較する。

| Variant | `gate_mode` | `weight_mode` | Weight 更新信号 | Gate 更新信号 | 統合式 | 期待 d |
|---------|-------------|---------------|-----------------|---------------|--------|--------|
| **V1_weight_only** | `"none"`      | `"weight_only"`    | GT       | 更新なし (固定 0.5) | `w`        | ~+2.0 |
| **V2_gate_only**   | `"consensus"` | `"gate_only"`      | GT(w更新) だが統合に使わない | Consensus | `0.5 * g` | ~+2.0 |
| **V3_both_gt**     | `"gt"`        | `"multiplicative"` | GT       | **GT (Weightと冗長)** | `w * g` | ~+2.0（崩壊） |
| **V4_cagl**        | `"consensus"` | `"multiplicative"` | GT       | **Consensus** | `w * g` | **~+4.0** |

## 各 variant の意図

### V1: Weight only (純粋な GT 学習)
- 最も基本的な online ensemble learning。各 topology の正解率を学んで加重投票。
- ベースラインとして機能。

### V2: Gate only (consensus のみで学習)
- Gate は consensus signal で更新するが、統合には `0.5 * g` しか使わない（Weight なし）。
- consensus-agreement 情報 **単独** での効果を測る。

### V3: Both GT (冗長化による崩壊検証)
- Weight も Gate も **同じ GT signal** で更新。
- `w * g` は同じ情報を2回使うだけで、相乗効果が出ないことを検証する。
- 期待: V1 と同程度の d。multiplicative 統合自体には価値がないことを示す。

### V4: CAGL (本体)
- Weight は GT で学習、Gate は consensus で学習。**2つの独立な信号** を掛け合わせる。
- 相乗効果により他 variant より明確に高い d が出ることを期待。

## 比較の読み方

- **V4 > V1** : Gate 追加の効果
- **V4 > V2** : Weight 追加の効果
- **V4 > V3** : Gate を consensus で更新することの効果（load-bearing novelty）

特に **V4 vs V3** の差が本研究の核心。両者とも multiplicative 統合を使うが、**Gate の学習信号が違うだけ** で効果が大きく変わる。これが再現されれば novelty の妥当性が支持される。

## 実験パラメータ

- 20 trials × 500 samples × 4 variants
- Paired Cohen's d (adaptive vs fixed baseline)
- Paired bootstrap CI (10000 resamples, α=0.05)
- Paired t-test p値

## 実装上の注意

- Fixed baseline は全 variant 共通で `gate_mode="none"` の初期状態（w=g=0.5）を使い、学習をスキップする。
- 同じ `weight_mode` を fixed 側にも適用することで **学習の効果のみを切り分け** られる。
- `V2_gate_only` の fixed baseline は `0.5 * 0.5 = 0.25` が全 topology で一様 → 単純多数決と等価。
