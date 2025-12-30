"""Image processing utilities for RF-DETR inference."""

import base64
from io import BytesIO
from typing import Union

import numpy as np
from PIL import Image


def encode_image_to_base64(image: Union[Image.Image, np.ndarray]) -> str:
    """Encode PIL Image or numpy array to base64 string.

    Args:
        image: PIL Image or numpy array in RGB format

    Returns:
        Base64 encoded string
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()


def decode_base64_to_image(base64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image.

    Args:
        base64_string: Base64 encoded image string

    Returns:
        PIL Image in RGB format
    """
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return image.convert("RGB")


def resize_image_maintain_aspect(
    image: Image.Image, max_size: int = 1024
) -> Image.Image:
    """Resize image while maintaining aspect ratio.

    Args:
        image: PIL Image to resize
        max_size: Maximum dimension (width or height)

    Returns:
        Resized PIL Image
    """
    width, height = image.size

    if max(width, height) <= max_size:
        return image

    if width > height:
        new_width = max_size
        new_height = int(height * max_size / width)
    else:
        new_height = max_size
        new_width = int(width * max_size / height)

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def validate_image_dimensions(image: Image.Image, max_dimension: int = 10000) -> None:
    """Validate image dimensions are within acceptable limits.

    Args:
        image: PIL Image to validate
        max_dimension: Maximum allowed dimension

    Raises:
        ValueError: If image dimensions exceed limits
    """
    width, height = image.size
    if width > max_dimension or height > max_dimension:
        raise ValueError(
            f"Image dimensions ({width}x{height}) exceed maximum allowed "
            f"({max_dimension}x{max_dimension})"
        )
