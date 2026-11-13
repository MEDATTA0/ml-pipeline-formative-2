"""Audio loading and feature extraction for voiceprint verification."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

PHRASES = ("yes_approve", "confirm_transaction")
SAMPLE_RATE = 22_050
N_MFCC = 13


def load_audio(path: Path, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y, sr


def extract_features(y: np.ndarray, sr: int) -> dict[str, float]:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)

    out: dict[str, float] = {}
    for i in range(N_MFCC):
        out[f"mfcc_{i}_mean"] = float(np.mean(mfcc[i]))
        out[f"mfcc_{i}_std"] = float(np.std(mfcc[i]))
    out["spectral_rolloff_mean"] = float(np.mean(rolloff))
    out["spectral_rolloff_std"] = float(np.std(rolloff))
    out["energy_mean"] = float(np.mean(rms))
    out["energy_std"] = float(np.std(rms))
    out["duration_sec"] = float(len(y) / sr)
    return out


from .features_schema import feature_column_names  # noqa: F401
