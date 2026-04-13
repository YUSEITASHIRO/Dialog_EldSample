#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Test Script 4: Torres (2025) - PitchFlower
=============================================================================
制御可能な特徴量 (Flow Matching の制御軸):
  1. pitch_shift_st  - 半音単位のピッチシフト (精密な音高操作)
  2. guidance_w      - F0条件付けの強さ (0.0=無条件, 2.0=強い条件付け)

各特徴量を4段階で変化させ、他の特徴量はデフォルトに固定。
合計: 2特徴量 × 4段階 = 8ファイル + 1ベースライン = 9ファイル (sample1のみ)
注: CPU上でのFlow計算は1ファイルあたり数分かかります。
=============================================================================
"""

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import pyworld as pw
import soundfile as sf
from pathlib import Path

try:
    from pitchflower.synthesizer import PitchFlowerSynthesizer
except ImportError:
    print("Error: pip install -e PitchFlower")
    import sys; sys.exit(1)

from vocos import Vocos

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_FILES = ["sample1.mp3"]  # CPU負荷のためsample1のみ
OUTPUT_DIR = SCRIPT_DIR / "output_04_pitchflower_comparison"
ODE_STEPS = 10  # 速度と品質のバランス

# ===================== 特徴量の定義 =====================
FEATURES = {
    "pitch_shift_st": {
        "values": [-6, -3, 0, 4],
        "default": 0,
        "description": "半音単位のピッチシフト (-6=大幅に低い, +4=高い)",
    },
    "guidance_w": {
        "values": [0.0, 0.5, 1.0, 2.0],
        "default": 1.0,
        "description": "F0条件付けの強さ (0.0=音高無視, 2.0=音高に忠実)",
    },
}


def get_f0(audio_np, fs=24000):
    frame_period = (256 / 24000) * 1000
    _f0, t = pw.harvest(audio_np.astype(np.float64), fs=fs,
                         frame_period=frame_period, f0_floor=40, f0_ceil=800)
    return pw.stonemask(audio_np.astype(np.float64), _f0, t, fs=fs)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 72)
    print("  Torres (2025) - PitchFlower (Flow Matching)")
    print("  Systematic Comparison: pitch_shift_st / guidance_w")
    print(f"  ODE Steps: {ODE_STEPS} | Device: {device}")
    print("=" * 72)

    synth = PitchFlowerSynthesizer.from_pretrained('diegotg343/PitchFlower').to(device)
    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)

    for fname in INPUT_FILES:
        fpath = SCRIPT_DIR / fname
        if not fpath.exists():
            continue

        stem = Path(fname).stem
        print(f"\n>> Processing: {fname}")

        y, sr = torchaudio.load(str(fpath))
        y = y.mean(dim=0, keepdim=True)
        if sr != 24000:
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=24000)

        # 10秒にトリミング (CPU速度のため)
        max_samples = 24000 * 10
        if y.shape[1] > max_samples:
            y = y[:, :max_samples]
            print(f"     (Trimmed to 10 seconds for CPU efficiency)")

        audio_np = y.squeeze().numpy()
        t_y = y.to(device)

        # 共通: セマンティックトークン抽出 (1回だけ)
        print("     1. Extracting Semantic Tokens...")
        with torch.no_grad():
            codes = synth.analyze_audios(t_y)
            features = synth.autoencoder.quantizer.decode(codes)
            features = synth.autoencoder.quantizer_out(features)
            rec_mel = synth.autoencoder.decoder(features)
            T_target = rec_mel.shape[-1]

        # 共通: F0 抽出
        print(f"     2. Extracting F0 ({T_target} frames)...")
        f0_raw = get_f0(audio_np, fs=24000)
        log_f0_np = np.zeros_like(f0_raw)
        mask = f0_raw > 0
        log_f0_np[mask] = np.log2(f0_raw[mask])

        t_f0_raw = torch.tensor(log_f0_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        t_f0_base = F.interpolate(t_f0_raw, size=(T_target,), mode='linear', align_corners=False).squeeze(1)

        # ベースライン (pitch_shift=0, w=1.0)
        print("     3. Generating baseline...")
        with torch.no_grad():
            mel_base = synth.flow_matcher.improve_mels(rec_mel, t_f0_base, steps=ODE_STEPS, w=1.0)
            wav_base = vocos.decode(mel_base).squeeze().cpu().numpy()
        sf.write(str(OUTPUT_DIR / f"{stem}__baseline.wav"), wav_base, 24000)
        print(f"   [OK] {stem}__baseline.wav")

        # 各特徴量を独立に変化
        defaults = {k: v["default"] for k, v in FEATURES.items()}
        for feat_name, feat_info in FEATURES.items():
            for val in feat_info["values"]:
                params = dict(defaults)
                params[feat_name] = val

                # F0の準備
                t_f0 = t_f0_base.clone()
                shift_st = params["pitch_shift_st"]
                if shift_st != 0:
                    f0_mask = t_f0 > 0
                    t_f0[f0_mask] += (shift_st / 12.0)

                w = params["guidance_w"]

                print(f"     Generating: pitch_shift_st={shift_st}, guidance_w={w}...")
                with torch.no_grad():
                    mel_mod = synth.flow_matcher.improve_mels(rec_mel, t_f0, steps=ODE_STEPS, w=w)
                    wav_mod = vocos.decode(mel_mod).squeeze().cpu().numpy()

                out_name = f"{stem}__{feat_name}={val}.wav"
                sf.write(str(OUTPUT_DIR / out_name), wav_mod, 24000)
                print(f"   [OK] {out_name}")

    print(f"\n  All files saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
