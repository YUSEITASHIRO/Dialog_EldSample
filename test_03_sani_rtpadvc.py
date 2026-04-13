#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Test Script 3: Sani (2025) - RT-PAD-VC
=============================================================================
制御可能な特徴量 (Prosody-Aware Decoder の制御軸):
  1. pitch_shift_st  - 半音単位のピッチシフト (話者間の声の高さ変換)
  2. vtlp_alpha      - 声道長変換係数 (話者間の声質変換)
  3. breathiness     - 気息性の強度 (声の質感制御)

各特徴量を4段階で変化させ、他の特徴量はデフォルトに固定。
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
OUTPUT_DIR = SCRIPT_DIR / "output_03_sani_comparison"
SR = 24000

# ===================== 特徴量の定義 =====================
FEATURES = {
    "pitch_shift_st": {
        "values": [-6, -3, 0, 4],
        "default": 0,
        "description": "半音単位のピッチシフト (-6=大幅に低い, +4=高い)",
    },
    "vtlp_alpha": {
        "values": [0.85, 0.95, 1.0, 1.1],
        "default": 1.0,
        "description": "声道長変換係数 (0.85=大柄な話者風, 1.1=小柄な話者風)",
    },
    "breathiness": {
        "values": [0.5, 1.0, 1.5, 2.0],
        "default": 1.0,
        "description": "気息性の強度 (0.5=クリア, 2.0=ハスキー)",
    },
}


def vtlp_warp(sp, alpha, sr=24000):
    """Vocal Tract Length Perturbation (声道長変換)
    
    alpha < 1.0: 声道が長い話者をシミュレート (声が低くなる)
    alpha > 1.0: 声道が短い話者をシミュレート (声が高くなる)
    """
    n_freq = sp.shape[1]
    freq_axis = np.arange(n_freq)
    # 周波数軸をワーピング
    warped_axis = freq_axis * alpha
    warped_axis = np.clip(warped_axis, 0, n_freq - 1)

    sp_warped = np.zeros_like(sp)
    for i in range(sp.shape[0]):
        sp_warped[i] = np.interp(freq_axis, warped_axis, sp[i])
    return sp_warped


def synthesize(f0, sp, ap, sr, pitch_shift_st=0, vtlp_alpha=1.0, breathiness=1.0):
    """WORLD パラメータをVC風に操作して再合成"""
    # ピッチシフト
    f0_mod = f0.copy()
    if pitch_shift_st != 0:
        f0_mod[f0 > 0] *= 2.0 ** (pitch_shift_st / 12.0)

    # 声道長変換 (VTLP)
    if vtlp_alpha != 1.0:
        sp_mod = vtlp_warp(sp, vtlp_alpha, sr)
    else:
        sp_mod = sp.copy()

    # 気息性
    ap_mod = np.clip(ap * breathiness, 0.0, 1.0)

    return pw.synthesize(f0_mod, sp_mod, ap_mod, sr)


def vocos_enhance(wav, vocos_model, device):
    t_wav = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        mel = vocos_model.feature_extractor(t_wav)
        enhanced = vocos_model.decode(mel)
    return enhanced.squeeze().cpu().numpy()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 72)
    print("  Sani (2025) - RT-PAD-VC (Prosody-Aware Decoder)")
    print("  Systematic Comparison: pitch_shift_st / vtlp_alpha / breathiness")
    print("=" * 72)

    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)

    for fname in INPUT_FILES:
        fpath = SCRIPT_DIR / fname
        if not fpath.exists():
            continue

        stem = Path(fname).stem
        print(f"\n>> Analyzing: {fname}")
        y, _ = librosa.load(str(fpath), sr=SR, mono=True)

        f0, t = pw.harvest(y.astype(np.float64), SR)
        sp = pw.cheaptrick(y.astype(np.float64), f0, t, SR)
        ap = pw.d4c(y.astype(np.float64), f0, t, SR)

        # ベースライン
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
