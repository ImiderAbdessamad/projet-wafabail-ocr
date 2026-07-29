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


def detect_content_box(
    image: Image.Image,
) -> tuple[int, int, int, int]:
    """Détecte une zone utile sans accepter un crop anormalement petit."""
    full_box = (0, 0, image.width, image.height)

    if image.width < 10 or image.height < 10:
        return full_box

    sample_points = (
        (0, 0),
        (image.width - 1, 0),
        (0, image.height - 1),
        (image.width - 1, image.height - 1),
    )

    colors = [image.getpixel(point) for point in sample_points]

    if image.mode == "RGB":
        background_color = tuple(
            sorted(channel_values)[len(channel_values) // 2]
            for channel_values in zip(*colors)
        )
    else:
        sorted_colors = sorted(colors)
        background_color = sorted_colors[len(sorted_colors) // 2]

    background = Image.new(image.mode, image.size, background_color)
    diff = ImageChops.difference(image, background)
    bbox = diff.getbbox()

    if not bbox:
        return full_box

    left, top, right, bottom = bbox
    detected_width = right - left
    detected_height = bottom - top

    if detected_width < image.width * 0.55:
        return full_box

    if detected_height < image.height * 0.55:
        return full_box

    pad_x = max(12, int(image.width * 0.01))
    pad_y = max(12, int(image.height * 0.01))

    return (
        max(0, left - pad_x),
        max(0, top - pad_y),
        min(image.width, right + pad_x),
        min(image.height, bottom + pad_y),
    )


def build_regions(width: int, height: int) -> list[PageRegion]:
    half = max(height // 2, 1)
    overlap = max(height // 12, 1)
    return [
        PageRegion("full", 0, 0, width, height),
        PageRegion("top", 0, 0, width, min(height, half + overlap)),
        PageRegion("bottom", 0, max(0, half - overlap), width, height),
    ]


def build_content_regions(
    width: int,
    height: int,
) -> list[PageRegion]:
    """Construit deux régions verticales avec chevauchement."""
    overlap = max(int(height * 0.10), 40)
    split = height // 2

    return [
        PageRegion(
            region_id="top",
            left=0,
            top=0,
            right=width,
            bottom=min(height, split + overlap),
        ),
        PageRegion(
            region_id="bottom",
            left=0,
            top=max(0, split - overlap),
            right=width,
            bottom=height,
        ),
    ]


def crop_region(image_bytes: bytes, region: PageRegion) -> bytes:
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        cropped = img.crop((region.left, region.top, region.right, region.bottom))
        output = io.BytesIO()
        cropped.save(output, format="JPEG", quality=82)
        return output.getvalue()


def crop_content_regions(
    image_bytes: bytes,
) -> list[tuple[str, bytes]]:
    """Découpe une page en régions haute et basse."""
    with Image.open(io.BytesIO(image_bytes)) as img:
        width, height = img.size

    regions = build_content_regions(width, height)

    return [
        (
            region.region_id,
            crop_region(image_bytes, region),
        )
        for region in regions
    ]


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
