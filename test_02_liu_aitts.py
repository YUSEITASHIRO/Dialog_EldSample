#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Test Script 2: Liu (2023) - Ai-TTS
=============================================================================
制御可能な特徴量 (Variance Adaptor の3要素):
  1. intensity     - 音量(エネルギー)の倍率
  2. speed         - 話速の倍率 (大きいほど速い)
  3. pitch_shift_st - 半音単位のピッチシフト

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
OUTPUT_DIR = SCRIPT_DIR / "output_02_liu_comparison"
SR = 24000

# ===================== 特徴量の定義 =====================
FEATURES = {
    "intensity": {
        "values": [0.5, 1.0, 1.5, 2.0],
        "default": 1.0,
        "description": "音量(エネルギー)の倍率 (0.5=小声, 2.0=大声)",
    },
    "speed": {
        "values": [0.6, 0.8, 1.0, 1.4],
        "default": 1.0,
        "description": "話速の倍率 (0.6=ゆっくり, 1.4=早口)",
    },
    "pitch_shift_st": {
        "values": [-4, 0, 2, 4],
        "default": 0,
        "description": "半音単位のピッチシフト (-4=低い, +4=高い)",
    },
}


def synthesize(f0, sp, ap, sr, intensity=1.0, speed=1.0, pitch_shift_st=0):
    """WORLD パラメータを操作して再合成"""
    n_frames = len(f0)

    # ピッチシフト
    f0_mod = f0.copy()
    if pitch_shift_st != 0:
        f0_mod[f0 > 0] *= 2.0 ** (pitch_shift_st / 12.0)

    # 話速変更 (WORLD フレーム列の伸縮)
    if speed != 1.0:
        n_new = max(1, int(n_frames / speed))
        indices = np.linspace(0, n_frames - 1, n_new)
        f0_mod = np.interp(indices, np.arange(n_frames), f0_mod)
        sp_new = np.zeros((n_new, sp.shape[1]))
        ap_new = np.zeros((n_new, ap.shape[1]))
        for j in range(sp.shape[1]):
            sp_new[:, j] = np.interp(indices, np.arange(n_frames), sp[:, j])
            ap_new[:, j] = np.interp(indices, np.arange(n_frames), ap[:, j])
    else:
        sp_new = sp.copy()
        ap_new = ap.copy()

    wav = pw.synthesize(f0_mod, sp_new, ap_new, sr)

    # 音量スケーリング
    if intensity != 1.0:
        wav = wav * intensity

    return wav


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
    print("  Liu (2023) - Ai-TTS (Variance Adaptor)")
    print("  Systematic Comparison: intensity / speed / pitch_shift_st")
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
