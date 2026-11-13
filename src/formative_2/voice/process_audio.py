#!/usr/bin/env python3
"""Augment raw WAVs, extract features, write audio_features.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

from .audio_utils import (
    SAMPLE_RATE,
    extract_features,
    feature_column_names,
    load_audio,
)

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "audio" / "raw"
AUG_DIR = ROOT / "data" / "audio" / "augmented"
FIG_DIR = ROOT / "outputs" / "figures" / "audio"


def parse_filename(path: Path) -> tuple[str, str]:
    stem = path.stem.lower()
    for phrase in ("yes_approve", "confirm_transaction"):
        if f"_{phrase}" in stem:
            return stem.split(f"_{phrase}")[0], phrase
    raise ValueError(f"Filename must include yes_approve or confirm_transaction: {path.name}")


def save_plot(y: np.ndarray, sr: int, title: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 5))
    librosa.display.waveshow(y, sr=sr, ax=axes[0])
    axes[0].set_title(f"Waveform — {title}")
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(S, sr=sr, x_axis="time", y_axis="hz", ax=axes[1])
    axes[1].set_title(f"Spectrogram — {title}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args()

    exts = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}
    paths = sorted(p for p in args.raw_dir.iterdir() if p.suffix.lower() in exts)
    if not paths:
        raise SystemExit(f"No audio in {args.raw_dir}")

    rows: list[dict] = []
    AUG_DIR.mkdir(parents=True, exist_ok=True)
    for path in paths:
        member, phrase = parse_filename(path)
        base = f"{member}_{phrase}"
        y, sr = load_audio(path)
        save_plot(y, sr, f"{base} (original)", FIG_DIR / f"{base}_original.png")
        rows.append(
            {
                "member_id": member,
                "phrase": phrase,
                "augmentation": "original",
                "file_path": str(path.relative_to(ROOT)),
                **extract_features(y, sr),
            }
        )
        for aug_name, y_aug in (
            ("pitch_up", librosa.effects.pitch_shift(y, sr=sr, n_steps=2)),
            ("time_stretch", librosa.effects.time_stretch(y, rate=1.12)),
        ):
            aug_path = AUG_DIR / f"{base}_{aug_name}.wav"
            sf.write(aug_path, y_aug, sr)
            save_plot(y_aug, sr, f"{base} ({aug_name})", FIG_DIR / f"{base}_{aug_name}.png")
            rows.append(
                {
                    "member_id": member,
                    "phrase": phrase,
                    "augmentation": aug_name,
                    "file_path": str(aug_path.relative_to(ROOT)),
                    **extract_features(y_aug, sr),
                }
            )

    df = pd.DataFrame(rows)
    cols = ["member_id", "phrase", "augmentation", "file_path", *feature_column_names()]
    df = df[cols]
    df.to_csv(ROOT / "audio_features.csv", index=False)
    print(f"Wrote {ROOT / 'audio_features.csv'} ({len(df)} rows)")


if __name__ == "__main__":
    main()
