from pathlib import Path

from PIL import Image, ImageOps
from torch import Tensor
from torchvision.models import MobileNet_V3_Small_Weights

FACE_WEIGHTS = MobileNet_V3_Small_Weights.IMAGENET1K_V1
CropBox = tuple[float, float, float, float]


def prepare_face_image(image_path: Path, crop_box: CropBox | None = None) -> Tensor:
    """Prepare a face crop as a batch of one, using the notebook's transforms."""
    try:
        with Image.open(image_path) as source:
            if getattr(source, "n_frames", 1) != 1:
                raise ValueError("Expected a single still image.")
            image = ImageOps.exif_transpose(source).convert("RGB")
    except (Image.DecompressionBombError, OSError) as error:
        raise ValueError(f"Cannot read image {image_path}: {error}") from error

    if crop_box is not None:
        left, top, right, bottom = crop_box
        if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
            raise ValueError("Crop coordinates must define a box within 0 and 1.")
        pixel_box = (
            round(left * image.width),
            round(top * image.height),
            round(right * image.width),
            round(bottom * image.height),
        )
        if pixel_box[0] == pixel_box[2] or pixel_box[1] == pixel_box[3]:
            raise ValueError("Crop is smaller than one pixel.")
        image = image.crop(pixel_box)

    padded_face = ImageOps.pad(
        image, (256, 256), Image.Resampling.BILINEAR, color=(128, 128, 128)
    )
    return FACE_WEIGHTS.transforms()(padded_face).unsqueeze(0)
