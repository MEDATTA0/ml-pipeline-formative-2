#!/usr/bin/env python3
"""Train voiceprint classifier; saves models/voice_verifier.joblib."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .features_schema import feature_column_names

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=ROOT / "audio_features.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    X = df[feature_column_names()].values
    y = df["member_id"].astype(str).values
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.25, random_state=42, stratify=y_enc
    )
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1_weighted": float(f1_score(y_test, pred, average="weighted")),
        "loss": float(log_loss(y_test, proba)),
        "members": list(le.classes_),
    }
    out_model = ROOT / "models" / "voice_verifier.joblib"
    out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": clf, "label_encoder": le, "feature_columns": feature_column_names()},
        out_model,
    )
    out_metrics = ROOT / "outputs" / "voice_model_metrics.json"
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"Model → {out_model}")


if __name__ == "__main__":
    main()
