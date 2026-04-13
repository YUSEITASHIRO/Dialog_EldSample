# Model 02: Liu (2023) — Ai-TTS 仕様書

## 論文情報
- **タイトル:** Ai-TTS: Controllable Adaptive and Interpretable Text-to-Speech
- **著者:** Liu et al. (2023)
- **核心技術:** FastSpeech 2 の Variance Adaptor にて Intensity・Duration・F0 を明示制御

## 制御可能な特徴量

### 1. `intensity` — 音量(エネルギー)の倍率
| 値 | 効果 |
|----|------|
| 0.5 | 音量を50%に低下 → 小声・囁き風 |
| **1.0** | **デフォルト (変化なし)** |
| 1.5 | 音量を50%増加 → 張りのある声 |
| 2.0 | 音量を100%増加 → 非常に大きな声 |

**物理的意味:** 波形振幅を直接スケーリングし、発話エネルギーを制御する。Variance Adaptor の Energy Predictor が学習するパラメータに相当。

### 2. `speed` — 話速の倍率
| 値 | 効果 |
|----|------|
| 0.6 | 40%減速 → 非常にゆっくりな発話 |
| 0.8 | 20%減速 → やや遅い発話 |
| **1.0** | **デフォルト (変化なし)** |
| 1.4 | 40%加速 → 早口な発話 |

**物理的意味:** WORLD のフレーム列を伸縮させ、音素の持続時間を変更する。Variance Adaptor の Duration Predictor が学習するパラメータに相当。ピッチやフォルマントは保持されるため、声質を維持したまま速度のみが変化する。

### 3. `pitch_shift_st` — 半音単位のピッチシフト
| 値 | 効果 |
|----|------|
| -4 | 4半音低下 → 低い声 |
| **0** | **デフォルト (変化なし)** |
| +2 | 2半音上昇 → やや高い声 |
| +4 | 4半音上昇 → 高い声 |

**物理的意味:** F0 に `2^(st/12)` を乗算し、音の高さを音楽的に正確にシフトする。Variance Adaptor の Pitch Predictor が学習するパラメータに相当。

## 生成ファイル一覧 (sample1の場合)

| # | ファイル名 | 変化した特徴量 |
|---|-----------|---------------|
| 0 | `sample1__baseline.wav` | なし (全デフォルト) |
| 1 | `sample1__intensity=0.5.wav` | intensity=0.5 |
| 2 | `sample1__intensity=1.0.wav` | intensity=1.0 |
| 3 | `sample1__intensity=1.5.wav` | intensity=1.5 |
| 4 | `sample1__intensity=2.0.wav` | intensity=2.0 |
| 5 | `sample1__speed=0.6.wav` | speed=0.6 |
| 6 | `sample1__speed=0.8.wav` | speed=0.8 |
| 7 | `sample1__speed=1.0.wav` | speed=1.0 |
| 8 | `sample1__speed=1.4.wav` | speed=1.4 |
| 9 | `sample1__pitch_shift_st=-4.wav` | pitch_shift_st=-4 |
| 10 | `sample1__pitch_shift_st=0.wav` | pitch_shift_st=0 |
| 11 | `sample1__pitch_shift_st=2.wav` | pitch_shift_st=2 |
| 12 | `sample1__pitch_shift_st=4.wav` | pitch_shift_st=4 |

※ sample2, sample3 についても同一構成で生成。

## 比較の観点
- **intensity の段階的変化:** 音量のスケーリングが音質に与える影響（クリッピングの有無等）
- **speed の変化:** 音素の持続時間が変わっても、ピッチと声質が保持されるかの確認
- **pitch_shift_st の変化:** 音高のみが変化し、話速や音量が保持されるかの確認
- **3特徴量の独立性:** 各特徴量を独立に操作できることの聴感的検証
