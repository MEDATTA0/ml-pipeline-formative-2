import argparse
from pathlib import Path

import torch
from torch import nn
from torchvision.models import mobilenet_v3_small

from .process_image import CropBox, prepare_face_image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "face_mobilenet_v3_small.pt"


def predict_face(
    image_path: Path,
    model_path: Path = DEFAULT_MODEL,
    crop_box: CropBox | None = None,
) -> tuple[str, float]:
    image_batch = prepare_face_image(image_path, crop_box)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    member_ids = checkpoint["members"]

    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(member_ids))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.inference_mode():
        probabilities = model(image_batch).softmax(dim=1)[0]
    predicted_class = int(probabilities.argmax())
    return member_ids[predicted_class], float(probabilities[predicted_class])


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a face among the enrolled members.")
    parser.add_argument("image", type=Path, help="Path to a single face image")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--crop", nargs=4, type=float, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
        help="Optional face box as fractions of the oriented image width and height",
    )
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    try:
        member_id, score = predict_face(
            args.image, args.model, tuple(args.crop) if args.crop is not None else None
        )
    except (ValueError, OSError) as error:
        parser.exit(2, f"{error}\n")
    print(f"Predicted member: {member_id} (score={score:.3f})")


if __name__ == "__main__":
    main()
