from pathlib import Path

import numpy as np
import pandas as pd

CUSTOMER_COLUMN = "customer_id_new"
PLATFORM_COLUMN = "social_media_platform"
SENTIMENT_COLUMN = "review_sentiment"
SCORE_COLUMNS = ["engagement_score", "purchase_interest_score"]
SENTIMENTS = ("Negative", "Neutral", "Positive")


def prepare_customer_features(
    profiles_path: Path, customer_id: int, feature_columns: list[str]
) -> pd.DataFrame:
    """Build the model's feature row from a customer's raw social profiles."""
    profiles = pd.read_csv(profiles_path, dtype={CUSTOMER_COLUMN: "string"})
    required_columns = {CUSTOMER_COLUMN, PLATFORM_COLUMN, SENTIMENT_COLUMN, *SCORE_COLUMNS}
    missing_columns = required_columns.difference(profiles.columns)
    if missing_columns:
        raise ValueError(f"Missing profile columns: {', '.join(sorted(missing_columns))}")

    profile_ids = pd.to_numeric(profiles[CUSTOMER_COLUMN].str.lstrip("A"), errors="coerce")
    customer_profiles = profiles.loc[profile_ids == customer_id].drop_duplicates()
    if customer_profiles.empty:
        raise ValueError(f"No social profiles found for customer {customer_id}.")

    scores = customer_profiles[SCORE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(scores.to_numpy(dtype=float, na_value=np.nan)).all():
        raise ValueError(f"Customer {customer_id} has missing or invalid scores.")

    platforms = customer_profiles[PLATFORM_COLUMN]
    if platforms.isna().any() or platforms.astype(str).str.strip().eq("").any():
        raise ValueError(f"Customer {customer_id} has a missing social media platform.")

    sentiments = customer_profiles[SENTIMENT_COLUMN]
    if not sentiments.isin(SENTIMENTS).all():
        raise ValueError(f"Customer {customer_id} has a missing or invalid review sentiment.")

    features = scores.mean().to_dict()
    features["n_platforms"] = platforms.nunique()
    for sentiment in SENTIMENTS:
        features[f"n_{sentiment.lower()}"] = int(sentiments.eq(sentiment).sum())

    unknown_features = set(feature_columns).difference(features)
    if unknown_features:
        raise ValueError(f"Unsupported model features: {', '.join(sorted(unknown_features))}")
    return pd.DataFrame([features], columns=feature_columns)
