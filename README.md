# 音声合成モデル比較検証 — 全体仕様書

## 概要
4つの最新音声合成・音声変換モデルについて、各モデルが内部で明示的に制御可能な音響・韻律特徴量の挙動を検証するための体系的比較実験を実施する。

## 実験設計

### 基本方針
- 各モデルについて、**制御可能な特徴量を1つずつ独立に変化**させ、他の特徴量はデフォルト値に固定する。
- 各特徴量は **4段階** の値で変化させる（デフォルト値を含む）。
- 特徴量が N 個のモデルでは、**N × 4 = 4N ファイル** が生成される。

### モデル別の生成ファイル数

| # | モデル | 論文 | 特徴量数 | 生成ファイル数 (per sample) |
|---|--------|------|---------|---------------------------|
| 01 | Kobayashi (2024) - Neural Formant | E2E-SiFi-NF | 3 | 12 + 1 baseline = 13 |
| 02 | Liu (2023) - Ai-TTS | FastSpeech 2 Variance Adaptor | 3 | 12 + 1 baseline = 13 |
| 03 | Sani (2025) - RT-PAD-VC | Prosody-Aware Decoder | 3 | 12 + 1 baseline = 13 |
| 04 | Torres (2025) - PitchFlower | Flow Matching | 2 | 8 + 1 baseline = 9 |

### 入力ファイル
- Script 01〜03: `sample1.mp3`, `sample2.mp3`, `sample3.wav` (各3サンプル)
- Script 04: `sample1.mp3` のみ (CPU上のODE計算負荷のため)

### ファイル命名規則
```
{サンプル名}__{特徴量名}={値}.wav
```

例:
```
sample1__formant_shift=0.8.wav
sample1__f0_scale=1.2.wav
sample1__breathiness=2.0.wav
sample1__baseline.wav
```

---

## 各モデルの制御特徴量一覧

### Model 01: Kobayashi (2024) — Neural Formant Synthesis
| 特徴量 | テスト値 | デフォルト | 意味 |
|--------|---------|-----------|------|
| `formant_shift` | 0.8, 0.9, 1.0, 1.2 | 1.0 | フォルマント周波数の伸縮 |
| `f0_scale` | 0.8, 0.9, 1.0, 1.2 | 1.0 | 基本周波数の倍率 |
| `breathiness` | 0.5, 1.0, 1.5, 2.0 | 1.0 | 気息性(非周期性)の強度 |

→ 詳細: [spec_01_kobayashi.md](spec_01_kobayashi.md)

### Model 02: Liu (2023) — Ai-TTS
| 特徴量 | テスト値 | デフォルト | 意味 |
|--------|---------|-----------|------|
| `intensity` | 0.5, 1.0, 1.5, 2.0 | 1.0 | 音量の倍率 |
| `speed` | 0.6, 0.8, 1.0, 1.4 | 1.0 | 話速の倍率 |
| `pitch_shift_st` | -4, 0, 2, 4 | 0 | 半音単位のピッチシフト |

→ 詳細: [spec_02_liu.md](spec_02_liu.md)

### Model 03: Sani (2025) — RT-PAD-VC
| 特徴量 | テスト値 | デフォルト | 意味 |
|--------|---------|-----------|------|
| `pitch_shift_st` | -6, -3, 0, 4 | 0 | 半音単位のピッチシフト |
| `vtlp_alpha` | 0.85, 0.95, 1.0, 1.1 | 1.0 | 声道長変換係数 (VTLP) |
| `breathiness` | 0.5, 1.0, 1.5, 2.0 | 1.0 | 気息性の強度 |

→ 詳細: [spec_03_sani.md](spec_03_sani.md)

### Model 04: Torres (2025) — PitchFlower
| 特徴量 | テスト値 | デフォルト | 意味 |
|--------|---------|-----------|------|
| `pitch_shift_st` | -6, -3, 0, 4 | 0 | 半音単位のピッチシフト |
| `guidance_w` | 0.0, 0.5, 1.0, 2.0 | 1.0 | F0条件付けの強さ |

→ 詳細: [spec_04_pitchflower.md](spec_04_pitchflower.md)

---

## 出力ディレクトリ構造
```
otamesi/
├── output_01_kobayashi_comparison/    ← Script 01 の出力
├── output_02_liu_comparison/          ← Script 02 の出力
├── output_03_sani_comparison/         ← Script 03 の出力
├── output_04_pitchflower_comparison/  ← Script 04 の出力
├── spec_00_overview.md                ← 本ファイル (全体仕様)
├── spec_01_kobayashi.md
├── spec_02_liu.md
├── spec_03_sani.md
└── spec_04_pitchflower.md
```

## 合成エンジン
| Script | 分析 | 操作 | 合成 | 品質向上 |
|--------|------|------|------|---------|
| 01 | WORLD (Harvest/CheapTrick/D4C) | スペクトル包絡伸縮 | WORLD | Vocos |
| 02 | WORLD | F0シフト・フレーム伸縮・振幅倍率 | WORLD | Vocos |
| 03 | WORLD | VTLP・F0シフト・AP倍率 | WORLD | Vocos |
| 04 | PitchFlower AutoEncoder | log2(F0)シフト | Flow Matching ODE | Vocos |
