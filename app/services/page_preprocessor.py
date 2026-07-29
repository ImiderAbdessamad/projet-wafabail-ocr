"""Prétraitement léger des pages Vision."""
from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image, ImageChops


@dataclass
class PageRegion:
    region_id: str
    left: int
    top: int
    right: int
    bottom: int


@dataclass
class PreprocessedPage:
    image_bytes: bytes
    orientation: int
    width: int
    height: int
    crop_box: tuple[int, int, int, int]
    regions: list[PageRegion]


def preprocess_page_image(image_bytes: bytes, *, orientation: int = 0) -> PreprocessedPage:
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        if orientation:
            img = img.rotate(-orientation, expand=True)
        crop_box = detect_content_box(img)
        cropped = img.crop(crop_box) if crop_box != (0, 0, img.width, img.height) else img
        output = io.BytesIO()
        cropped.save(output, format="JPEG", quality=85)
        regions = build_regions(cropped.width, cropped.height)
        return PreprocessedPage(
            image_bytes=output.getvalue(),
            orientation=orientation,
            width=cropped.width,
            height=cropped.height,
            crop_box=crop_box,
            regions=regions,
        )


def detect_content_box(image: Image.Image) -> tuple[int, int, int, int]:
    background = Image.new(image.mode, image.size, image.getpixel((0, 0)))
    diff = ImageChops.difference(image, background)
    bbox = diff.getbbox()
    if not bbox:
        return (0, 0, image.width, image.height)
    left, top, right, bottom = bbox
    pad = 8
    return (
        max(0, left - pad),
        max(0, top - pad),
        min(image.width, right + pad),
        min(image.height, bottom + pad),
    )


def build_regions(width: int, height: int) -> list[PageRegion]:
    half = max(height // 2, 1)
    overlap = max(height // 12, 1)
    return [
        PageRegion("full", 0, 0, width, height),
        PageRegion("top", 0, 0, width, min(height, half + overlap)),
        PageRegion("bottom", 0, max(0, half - overlap), width, height),
    ]


def crop_region(image_bytes: bytes, region: PageRegion) -> bytes:
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        cropped = img.crop((region.left, region.top, region.right, region.bottom))
        output = io.BytesIO()
        cropped.save(output, format="JPEG", quality=82)
        return output.getvalue()


def rotate_image_bytes(
    image_bytes: bytes,
    angle: int,
    *,
    quality: int = 85,
) -> bytes:
    """Tourne une image JPEG et retourne les nouveaux octets."""
    with Image.open(io.BytesIO(image_bytes)) as img:
        rotated = img.convert("RGB").rotate(-angle, expand=True)
        output = io.BytesIO()
        rotated.save(output, format="JPEG", quality=quality)
        return output.getvalue()
