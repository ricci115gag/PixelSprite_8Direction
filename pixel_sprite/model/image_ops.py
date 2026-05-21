from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def get_palette(img: Image.Image, alpha_threshold: int = 127) -> tuple[tuple[int, int, int], ...]:
    rgba = np.array(img.convert("RGBA"))
    opaque = rgba[:, :, 3] > alpha_threshold
    return tuple({tuple(map(int, rgb)) for rgb in rgba[:, :, :3][opaque]})


def apply_palette_and_binarize(
    source_img: Image.Image,
    target_img: Image.Image,
    alpha_threshold: int = 127,
) -> Image.Image:
    rgba = np.array(source_img.convert("RGBA"))
    source_palette = get_palette(source_img, alpha_threshold)
    target_palette = get_palette(target_img, alpha_threshold)
    if not source_palette or not target_palette:
        return _binarize_alpha_array(rgba, alpha_threshold)

    color_mapping = {
        source_color: min(
            target_palette,
            key=lambda target: sum((source_color[i] - target[i]) ** 2 for i in range(3)),
        )
        for source_color in source_palette
    }

    h, w = rgba.shape[:2]
    for y in range(h):
        for x in range(w):
            if rgba[y, x, 3] <= alpha_threshold:
                rgba[y, x, :4] = (0, 0, 0, 0)
                continue
            rgb = tuple(map(int, rgba[y, x, :3]))
            rgba[y, x, :3] = color_mapping.get(rgb, rgb)
            rgba[y, x, 3] = 255
    return Image.fromarray(rgba, mode="RGBA")


def process_image_alpha(
    input_path: str | Path,
    output_path: str | Path | None = None,
    alpha_threshold: int = 127,
) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else input_path.with_name(
        f"{input_path.stem}_processed{input_path.suffix}"
    )
    rgba = np.array(Image.open(input_path).convert("RGBA"))
    _binarize_alpha_array(rgba, alpha_threshold).save(output_path)
    return output_path


def _binarize_alpha_array(rgba: np.ndarray, alpha_threshold: int) -> Image.Image:
    rgba[:, :, 3] = np.where(rgba[:, :, 3] >= alpha_threshold, 255, 0)
    rgba[rgba[:, :, 3] == 0, :3] = 0
    return Image.fromarray(rgba, mode="RGBA")

