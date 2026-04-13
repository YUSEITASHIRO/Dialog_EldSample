# Model 04: Torres (2025) — PitchFlower 仕様書

## 論文情報
- **タイトル:** PitchFlower: Pitch-Controllable Voice Conversion via Flow Matching
- **著者:** Torres et al. (2025)
- **核心技術:** AutoEncoder-RVQ + Flow Matching による F0 条件付きメルスペクトログラム生成
- **公式リポジトリ:** https://github.com/diegotg2000/PitchFlower
- **学習済み重み:** `diegotg343/PitchFlower` (HuggingFace)

## 制御可能な特徴量

### 1. `pitch_shift_st` — 半音単位のピッチシフト
| 値 | 効果 |
|----|------|
| -6 | 6半音低下 → 大幅に低い声 |
| -3 | 3半音低下 → やや低い声 |
| **0** | **デフォルト (変化なし)** |
| +4 | 4半音上昇 → 高い声 |

**物理的意味:** F0 を対数スケール (log2) 上でシフトする。`log2(f0) += semitones / 12.0`。PitchFlower のFlow Matching は F0 情報を明示的な条件入力として受け取るため、スペクトル特性（声質）を保持したままピッチのみを精密に制御できる。

### 2. `guidance_w` — F0 条件付けの強さ (Classifier-Free Guidance)
| 値 | 効果 |
|----|------|
| 0.0 | F0条件を完全に無視 → 自由なピッチで生成 |
| 0.5 | 弱い条件付け → F0にある程度従う |
| **1.0** | **デフォルト (標準的な条件付け)** |
| 2.0 | 強い条件付け → F0に非常に忠実 |

**物理的意味:** Flow Matching の ODE ソルバーにおけるベクトル場の計算で、条件付き推論と無条件推論の加重平均を取る。`v = w * v_cond + (1-w) * v_uncond`。w=0 で無条件生成、w>1 で生成多様性と引き換えにF0への追従性が強化される (classifier-free guidance)。

## 生成ファイル一覧 (sample1のみ)

| # | ファイル名 | 変化した特徴量 |
|---|-----------|---------------|
| 0 | `sample1__baseline.wav` | なし (pitch=0, w=1.0) |
| 1 | `sample1__pitch_shift_st=-6.wav` | pitch_shift_st=-6 |
| 2 | `sample1__pitch_shift_st=-3.wav` | pitch_shift_st=-3 |
| 3 | `sample1__pitch_shift_st=0.wav` | pitch_shift_st=0 |
| 4 | `sample1__pitch_shift_st=4.wav` | pitch_shift_st=4 |
| 5 | `sample1__guidance_w=0.0.wav` | guidance_w=0.0 |
| 6 | `sample1__guidance_w=0.5.wav` | guidance_w=0.5 |
| 7 | `sample1__guidance_w=1.0.wav` | guidance_w=1.0 |
| 8 | `sample1__guidance_w=2.0.wav` | guidance_w=2.0 |

**注:** CPU 上の ODE 計算負荷のため、sample1 のみ処理（10秒にトリミング）。

## 他モデルとの比較ポイント

### PitchFlower vs Script 01/03 のピッチシフト
| 観点 | Script 01/03 (WORLD) | Script 04 (PitchFlower) |
|------|---------------------|------------------------|
| 手法 | F0 の直接スケーリング | Auto-Regressive Flow による条件付き生成 |
| 品質 | WORLDボコーダーの限界 | ニューラルフロー生成による高品質 |
| 制御精度 | 高い（直接操作） | 非常に高い（学習された潜在空間内での操作） |
| 計算コスト | 軽量 (< 1秒) | 重い (ODE計算: 数分/ファイル on CPU) |

### guidance_w の独自性
`guidance_w` は PitchFlower 固有の制御パラメータで、Flow Matching の条件付け強度を制御する。
- **w=0.0:** ピッチ情報を無視した生成。元のメルスペクトログラムからの「自由な再構成」
- **w=2.0:** ピッチ情報に極端に忠実な生成。イントネーションの精密な再現に有用
