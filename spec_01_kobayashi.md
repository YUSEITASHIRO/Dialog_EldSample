# Model 01: Kobayashi (2024) — Neural Formant Synthesis 仕様書

## 論文情報
- **タイトル:** End-to-End Source-Filter Neural Formant Synthesis (E2E-SiFi-NF)
- **著者:** Kobayashi et al. (2024)
- **核心技術:** SiFi-GAN ベースのSource-Filterモデルにフォルマント/帯域幅特徴量を条件入力

## 制御可能な特徴量

### 1. `formant_shift` — フォルマント周波数の伸縮
| 値 | 効果 |
|----|------|
| 0.8 | フォルマント周波数を20%低下 → 大柄な話者・太い声 |
| 0.9 | フォルマント周波数を10%低下 → やや太い声 |
| **1.0** | **デフォルト (変化なし)** |
| 1.2 | フォルマント周波数を20%上昇 → 小柄な話者・細い声 |

**物理的意味:** スペクトル包絡の周波数軸を伸縮させ、共鳴周波数（F1〜F3）の位置を一括変更する。声道の長さが変わったかのような効果が得られる。

### 2. `f0_scale` — 基本周波数(F0)の倍率
| 値 | 効果 |
|----|------|
| 0.8 | 声の高さが20%低下 |
| 0.9 | 声の高さが10%低下 |
| **1.0** | **デフォルト (変化なし)** |
| 1.2 | 声の高さが20%上昇 |

**物理的意味:** 声帯振動の基本周波数を直接スケーリングする。フォルマント位置は変化しないため、声の高さのみが変わり、話者の身体的特徴（声質）は保持される。

### 3. `breathiness` — 気息性(非周期性)の強度
| 値 | 効果 |
|----|------|
| 0.5 | 非周期成分を50%に抑制 → 澄んだ透明感のある声 |
| **1.0** | **デフォルト (変化なし)** |
| 1.5 | 非周期成分を50%増加 → やや息まじりの声 |
| 2.0 | 非周期成分を100%増加 → ウィスパー風の声 |

**物理的意味:** WORLDの非周期性指標 (Aperiodicity) をスケーリングし、声帯の不完全閉鎖による気流成分の量を制御する。

## 生成ファイル一覧 (sample1の場合)

| # | ファイル名 | 変化した特徴量 |
|---|-----------|---------------|
| 0 | `sample1__baseline.wav` | なし (全デフォルト) |
| 1 | `sample1__formant_shift=0.8.wav` | formant_shift=0.8 |
| 2 | `sample1__formant_shift=0.9.wav` | formant_shift=0.9 |
| 3 | `sample1__formant_shift=1.0.wav` | formant_shift=1.0 |
| 4 | `sample1__formant_shift=1.2.wav` | formant_shift=1.2 |
| 5 | `sample1__f0_scale=0.8.wav` | f0_scale=0.8 |
| 6 | `sample1__f0_scale=0.9.wav` | f0_scale=0.9 |
| 7 | `sample1__f0_scale=1.0.wav` | f0_scale=1.0 |
| 8 | `sample1__f0_scale=1.2.wav` | f0_scale=1.2 |
| 9 | `sample1__breathiness=0.5.wav` | breathiness=0.5 |
| 10 | `sample1__breathiness=1.0.wav` | breathiness=1.0 |
| 11 | `sample1__breathiness=1.5.wav` | breathiness=1.5 |
| 12 | `sample1__breathiness=2.0.wav` | breathiness=2.0 |

※ sample2, sample3 についても同一構成で生成。

## 比較の観点
- **formant_shift vs f0_scale:** 声の高さを変える2つの異なるアプローチの聴感比較
- **breathiness の段階的変化:** 気息性がどの程度で知覚的に区別可能か
- **formant_shift + f0_scale の独立性:** 一方を変えても他方が影響しないことの確認
