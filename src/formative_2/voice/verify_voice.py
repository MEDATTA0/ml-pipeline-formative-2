#!/usr/bin/env python3
"""CLI voice gate: exit 0 = approved, 1 = denied."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np

from .audio_utils import (
    SAMPLE_RATE,
    extract_features,
    feature_column_names,
    load_audio,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = ROOT / "models" / "voice_verifier.joblib"


def predict_speaker(wav_path: Path, model_path: Path = DEFAULT_MODEL) -> tuple[str, float]:
    if not wav_path.is_file():
        raise FileNotFoundError(f"Missing audio: {wav_path}")

    bundle = joblib.load(model_path)
    clf = bundle["model"]
    le = bundle["label_encoder"]

    y, sr = load_audio(wav_path, sr=SAMPLE_RATE)
    if y.size == 0 or not np.isfinite(y).all() or not np.any(y):
        raise ValueError("Expected a non-silent audio recording.")
    feats = extract_features(y, sr)
    row = np.array([[feats[c] for c in feature_column_names()]])
    if not np.isfinite(row).all():
        raise ValueError("Audio produced invalid voice features.")
    proba = clf.predict_proba(row)[0]
    best_idx = int(np.argmax(proba))
    member = str(le.inverse_transform([clf.classes_[best_idx]])[0])
    confidence = float(proba[best_idx])

    return member, confidence


def verify(wav_path: Path, model_path: Path = DEFAULT_MODEL, threshold: float = 0.5) -> int:
    if not model_path.exists():
        print(f"Missing model: {model_path}", file=sys.stderr)
        print("Train via Colab or: python -m formative_2.voice.train_voice_model", file=sys.stderr)
        return 2

    try:
        member, confidence = predict_speaker(wav_path, model_path)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Cannot verify voice: {error}", file=sys.stderr)
        return 2

    print(f"Predicted member: {member} (confidence={confidence:.3f})")
    if confidence >= threshold:
        print("Voice verification: APPROVED")
        return 0
    print("Voice verification: DENIED")
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Voiceprint verification for CLI demo")
    parser.add_argument("wav", type=Path, help="Path to voice sample WAV")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    sys.exit(verify(args.wav, args.model, args.threshold))


if __name__ == "__main__":
    main()
