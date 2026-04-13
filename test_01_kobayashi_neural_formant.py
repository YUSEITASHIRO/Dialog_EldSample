#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Test Script 1: Kobayashi (2024) - Neural Formant Synthesis
=============================================================================
制御可能な特徴量:
  1. formant_shift  - フォルマント周波数の伸縮 (声のキャラクター)
  2. f0_scale       - 基本周波数の倍率 (声の高さ)
  3. breathiness    - 気息性の強度 (息の混じり具合)

各特徴量を4段階で変化させ、他の特徴量はデフォルト(1.0)に固定。
合計: 3特徴量 × 4段階 = 12ファイル + 1ベースライン = 13ファイル (per sample)
=============================================================================
"""

import numpy as np
import torch
import librosa
import soundfile as sf
import pyworld as pw
from vocos import Vocos
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_FILES = ["sample1.mp3", "sample2.mp3", "sample3.wav"]
OUTPUT_DIR = SCRIPT_DIR / "output_01_kobayashi_comparison"
SR = 24000

# ===================== 特徴量の定義 =====================
FEATURES = {
    "formant_shift": {
        "values": [0.8, 0.9, 1.0, 1.2],
        "default": 1.0,
        "description": "フォルマント周波数の伸縮係数 (0.8=低い声, 1.2=高い声)",
    },
    "f0_scale": {
        "values": [0.8, 0.9, 1.0, 1.2],
        "default": 1.0,
        "description": "基本周波数(F0)の倍率 (0.8=低い, 1.2=高い)",
    },
    "breathiness": {
        "values": [0.5, 1.0, 1.5, 2.0],
        "default": 1.0,
        "description": "気息性の強度 (0.5=澄んだ声, 2.0=息まじりの声)",
    },
}

def synthesize(f0, sp, ap, sr, formant_shift=1.0, f0_scale=1.0, breathiness=1.0):
    """WORLD パラメータを操作して再合成"""
    # F0 スケーリング
    f0_mod = f0.copy()
    f0_mod[f0 > 0] *= f0_scale

    # フォルマントシフト (スペクトル包絡の周波数軸伸縮)
    sp_mod = np.zeros_like(sp)
    freq_axis = np.arange(sp.shape[1])
    for i in range(sp.shape[0]):
        sp_mod[i] = np.interp(freq_axis / formant_shift, freq_axis, sp[i])

    # 気息性 (非周期性指標のスケーリング)
    ap_mod = np.clip(ap * breathiness, 0.0, 1.0)

    return pw.synthesize(f0_mod, sp_mod, ap_mod, sr)


def vocos_enhance(wav, vocos_model, device):
    """Vocos ニューラルボコーダーで音質を向上"""
    t_wav = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        mel = vocos_model.feature_extractor(t_wav)
        enhanced = vocos_model.decode(mel)
    return enhanced.squeeze().cpu().numpy()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 72)
    print("  Kobayashi (2024) - Neural Formant Synthesis")
    print("  Systematic Comparison: formant_shift / f0_scale / breathiness")
    print("=" * 72)

    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)

    for fname in INPUT_FILES:
        fpath = SCRIPT_DIR / fname
        if not fpath.exists():
            continue

        stem = Path(fname).stem
        print(f"\n>> Analyzing: {fname}")
        y, _ = librosa.load(str(fpath), sr=SR, mono=True)

        # WORLD 分析
        f0, t = pw.harvest(y.astype(np.float64), SR)
        sp = pw.cheaptrick(y.astype(np.float64), f0, t, SR)
        ap = pw.d4c(y.astype(np.float64), f0, t, SR)

        # ベースライン (全デフォルト値)
        wav_base = synthesize(f0, sp, ap, SR)
        wav_base = vocos_enhance(wav_base, vocos, device)
        sf.write(str(OUTPUT_DIR / f"{stem}__baseline.wav"), wav_base, SR)
        print(f"   [OK] {stem}__baseline.wav")

        # 各特徴量を独立に変化
        defaults = {k: v["default"] for k, v in FEATURES.items()}
        for feat_name, feat_info in FEATURES.items():
            for val in feat_info["values"]:
                params = dict(defaults)
                params[feat_name] = val
                
                wav_mod = synthesize(f0, sp, ap, SR, **params)
                wav_mod = vocos_enhance(wav_mod, vocos, device)
                
                out_name = f"{stem}__{feat_name}={val}.wav"
                sf.write(str(OUTPUT_DIR / out_name), wav_mod, SR)
                print(f"   [OK] {out_name}")

    print(f"\n  All files saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
