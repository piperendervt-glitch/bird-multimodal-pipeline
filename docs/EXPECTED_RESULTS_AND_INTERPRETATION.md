# Expected Results and Interpretation

## 期待される結果

元の multimodal-aas-bird で観察された効果が合成データでも再現される場合、以下のパターンが見える:

| Variant | 期待 Cohen's d | 期待 ΔF1 |
|---------|----------------|----------|
| V1_weight_only | ~+2.0 | +0.02 ~ +0.05 |
| V2_gate_only   | ~+2.0 | +0.02 ~ +0.05 |
| V3_both_gt     | ~+2.0 | +0.02 ~ +0.05 |
| V4_cagl        | **~+4.0** | **+0.05 ~ +0.10** |

V4 が他3つに対して **d で 1.5倍以上、ΔF1 で 1.5 ~ 2倍** の差を出すことを期待。

## シナリオ判定基準

実験スクリプトは最後に以下の3シナリオに自動判定する:

### シナリオA: 完全再現
**条件**: `V4.d > 1.5 * max(V1.d, V2.d, V3.d)`

- 合成データの構造的特徴のみから CAGL 効果が再現されたことを意味する
- Load-bearing novelty の妥当性が強く支持される
- 元データ固有ではなく、**ensemble + 部分相関** という一般的条件下で効果が生じる

### シナリオB: 部分再現
**条件**: `V4.d > max(others.d)` だが 1.5倍未満

- 方向性は正しいが効果量が期待より小さい
- 合成データの設計パラメータ（accuracy, shared_noise）を調整すれば強化される可能性
- 実データ固有の構造が効果の一部を担っている可能性

### シナリオC: 再現されず
**条件**: `V4.d <= max(others.d)`

- 合成データ設計に問題がある（例: topology 間差異が小さすぎる、shared_noise が効いていない）
- または novelty 自体に見直しが必要
- 対処: `topo_accuracy` の非対称性を強める、`shared_noise_rate` を 0.15 ~ 0.20 に上げる、data size を増やす など

## 解釈のポイント

### なぜ V3 は崩壊するか

V3 は Weight も Gate も **GT signal** で更新するため、両者は強く相関したスカラーに収束する:

```
w[t, l] ≈ topo_accuracy[t][l]
g[t, l] ≈ topo_accuracy[t][l]
w * g   ≈ topo_accuracy[t][l]^2
```

これは単に Weight を 2乗しただけで、情報量は変わらない。結果として V1 と同程度の d に留まる。

### なぜ V4 は効くか

Gate は consensus signal で学習するため、`final` 予測が作られるダイナミクスに依存した値になる:

- fusion topology C のような強い topology の `final` への寄与度が高い
- それに同調する topology の `g` が上がる → **信頼できる topology 群が互いを補強**
- 非同調 topology の `g` が下がる → 異常値を自然に抑制

これは Weight（各 topology 単独の正解率）とは異なる情報であり、**独立した補正項** として機能する。

## 失敗時の診断

- **全 variant で d が小さい (< 1)**: 学習シグナルが弱い → `shared_noise_rate` を上げる、`N` を増やす
- **V4 ≒ V3**: Gate の consensus signal が機能していない → 統合予測 `final` の多様性を確認（例: 全 topology がほぼ同じ値を返す状況）
- **V2 が V1/V3 より著しく低い**: `gate_only` mode の仕様を確認（`0.5 * g` になっているか）

## 統計的有意性

- Paired t-test p値 < 0.05: 学習効果が偶然でないことの必要条件
- Bootstrap 95% CI が 0 を跨がない: 効果サイズの下界が正
- d の解釈: 0.2 (小) / 0.5 (中) / 0.8 (大) / 2.0 以上 (超大)

20 trials で d ≈ 4.0 は極めて強い効果であり、再現されれば novelty の妥当性は統計的に十分支持される。
