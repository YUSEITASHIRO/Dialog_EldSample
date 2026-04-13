# Model 03: Sani (2025) — RT-PAD-VC 仕様書

## 論文情報
- **タイトル:** Real-Time Prosody-Aware Decoder Voice Conversion (RT-PAD-VC)
- **著者:** Sani et al. (2025)
- **核心技術:** Prosody-Aware Decoder による FiLM 条件付き音声変換

## 制御可能な特徴量

### 1. `pitch_shift_st` — 半音単位のピッチシフト
| 値 | 効果 |
|----|------|
| -6 | 6半音低下 → 大幅に低い声 (男性→さらに低い男性) |
| -3 | 3半音低下 → やや低い声 |
| **0** | **デフォルト (変化なし)** |
| +4 | 4半音上昇 → 高い声 (男性→女性に近づく) |

**物理的意味:** 話者間の韻律(ピッチ)差を模擬する。RT-PAD-VC では、コンテンツエンコーダが言語情報を抽出した後、デコーダがF0条件を受け取って再構成するため、ピッチと内容が独立に制御される。

### 2. `vtlp_alpha` — 声道長変換係数 (VTLP)
| 値 | 効果 |
|----|------|
| 0.85 | 声道を長く → 大柄な話者の声質 (低いフォルマント) |
| 0.95 | 声道をやや長く → やや太い声質 |
| **1.0** | **デフォルト (変化なし)** |
| 1.1 | 声道を短く → 小柄な話者の声質 (高いフォルマント) |

**物理的意味:** Vocal Tract Length Perturbation (VTLP) は、話者間の声道長の物理的差異をシミュレートする標準的な音声変換手法。周波数軸のワーピングにより、話者固有のスペクトル特性（フォルマント配置）を変換する。Script 01 の `formant_shift` との違いは、VTLP が話者適応（声質変換）を目的とした手法である点。

### 3. `breathiness` — 気息性の強度
| 値 | 効果 |
|----|------|
| 0.5 | 非周期成分を50%に低減 → 非常にクリアな声 |
| **1.0** | **デフォルト (変化なし)** |
| 1.5 | 非周期成分を50%増加 → 息まじりの声 |
| 2.0 | 非周期成分を100%増加 → ハスキーな声 |

**物理的意味:** 声の質感(voice quality)を制御する。RT-PAD-VC のデコーダは、FiLM条件付けにより話者の声質特徴を注入するが、気息性はその重要な構成要素の一つ。

## 生成ファイル一覧 (sample1の場合)

| # | ファイル名 | 変化した特徴量 |
|---|-----------|---------------|
| 0 | `sample1__baseline.wav` | なし (全デフォルト) |
| 1 | `sample1__pitch_shift_st=-6.wav` | pitch_shift_st=-6 |
| 2 | `sample1__pitch_shift_st=-3.wav` | pitch_shift_st=-3 |
| 3 | `sample1__pitch_shift_st=0.wav` | pitch_shift_st=0 |
| 4 | `sample1__pitch_shift_st=4.wav` | pitch_shift_st=4 |
| 5 | `sample1__vtlp_alpha=0.85.wav` | vtlp_alpha=0.85 |
| 6 | `sample1__vtlp_alpha=0.95.wav` | vtlp_alpha=0.95 |
| 7 | `sample1__vtlp_alpha=1.0.wav` | vtlp_alpha=1.0 |
| 8 | `sample1__vtlp_alpha=1.1.wav` | vtlp_alpha=1.1 |
| 9 | `sample1__breathiness=0.5.wav` | breathiness=0.5 |
| 10 | `sample1__breathiness=1.0.wav` | breathiness=1.0 |
| 11 | `sample1__breathiness=1.5.wav` | breathiness=1.5 |
| 12 | `sample1__breathiness=2.0.wav` | breathiness=2.0 |

※ sample2, sample3 についても同一構成で生成。

## Script 01 (Kobayashi) との比較
| 観点 | Script 01 (formant_shift) | Script 03 (vtlp_alpha) |
|------|--------------------------|----------------------|
| 目的 | ニューラルフォルマントフィルタの特性検証 | 話者間の声質変換 |
| 手法 | スペクトル包絡の一様な周波数伸縮 | 声道長に基づく周波数ワーピング |
| パラメータ範囲 | 0.8 〜 1.2 | 0.85 〜 1.1 |
| 物理的解釈 | 共鳴周波数(F1-F3)の一括シフト | 声道長の物理的差異のシミュレーション |
