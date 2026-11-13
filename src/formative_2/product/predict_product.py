import argparse
from pathlib import Path

import joblib

from .process_profile import prepare_customer_features

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "product_model.joblib"
DEFAULT_PROFILES = PROJECT_ROOT / "customer_social_profiles.csv"


def predict_product(
    customer_id: int,
    profiles_path: Path = DEFAULT_PROFILES,
    model_path: Path = DEFAULT_MODEL,
) -> str:
    artifact = joblib.load(model_path)
    customer_features = prepare_customer_features(
        profiles_path, customer_id, artifact["feature_cols"]
    )
    return str(artifact["model"].predict(customer_features)[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a product category from social profiles.")
    parser.add_argument("customer_id", type=int, help="Numeric customer ID, such as 151")
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    try:
        category = predict_product(args.customer_id, args.profiles, args.model)
    except (ValueError, OSError) as error:
        parser.exit(2, f"{error}\n")
    print(f"Predicted product category: {category}")


if __name__ == "__main__":
    main()
