import argparse
import sys
from pathlib import Path

import torch

from .face.predict_face import predict_face
from .face.process_image import CropBox
from .product.predict_product import predict_product
from .voice.verify_voice import predict_speaker

# Voice label -> (face label, demo customer ID)
DEMO_ACCOUNTS = {
    "kelvin": ("member_1", 151),
    "beulla": ("member_2", 192),
    "ibrahim": ("member_3", 114),
}
FACE_THRESHOLD = 0.90
VOICE_THRESHOLD = 0.50


def run_transaction(
    account: str,
    image_path: Path,
    audio_path: Path,
    crop_box: CropBox | None = None,
) -> int:
    expected_face, customer_id = DEMO_ACCOUNTS[account]
    face_member, face_score = predict_face(image_path, crop_box=crop_box)
    if face_member != expected_face or not face_score >= FACE_THRESHOLD:
        print("Access denied: face check failed.")
        return 1
    print("Face check passed.")

    recommendation = predict_product(customer_id)

    speaker, voice_score = predict_speaker(audio_path)
    if speaker != account or not voice_score >= VOICE_THRESHOLD:
        print("Access denied: voice check failed.")
        return 1

    print("Voice check passed. Transaction approved.")
    print(f"Recommended product category: {recommendation}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Formative 2 transaction demo.")
    parser.add_argument("account", choices=DEMO_ACCOUNTS, help="Claimed demo account")
    parser.add_argument("image", type=Path, help="Face photo")
    parser.add_argument("audio", type=Path, help="Voice sample WAV")
    parser.add_argument(
        "--crop", nargs=4, type=float, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
        help="Optional face box as fractions of the oriented image size",
    )
    args = parser.parse_args()
    crop_box = tuple(args.crop) if args.crop is not None else None
    torch.set_num_threads(min(4, torch.get_num_threads()))

    try:
        exit_code = run_transaction(args.account, args.image, args.audio, crop_box)
    except (ValueError, OSError, RuntimeError, EOFError) as error:
        print(f"Transaction could not be completed: {error}", file=sys.stderr)
        exit_code = 2
    raise SystemExit(exit_code)
