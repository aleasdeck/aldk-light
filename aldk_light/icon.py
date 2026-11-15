"""Значок приложения: радужное кольцо. Используется в трее и при сборке exe."""

from __future__ import annotations

import numpy as np
from PIL import Image


def make_icon(size: int = 64) -> Image.Image:
    big = size * 4  # рисуем крупнее и уменьшаем — так края получаются гладкими
    ys, xs = np.mgrid[0:big, 0:big].astype(np.float32)
    centre = (big - 1) / 2
    dx, dy = xs - centre, ys - centre
    radius = np.hypot(dx, dy) / (big / 2)
    hue = (np.arctan2(dy, dx) / (2 * np.pi)) % 1.0

    # HSV → RGB при насыщенности 0.85 и яркости 1.
    channels = []
    for offset in (5, 3, 1):
        k = (offset + hue * 6) % 6
        channels.append(1.0 - 0.85 * np.clip(np.minimum(k, 4 - k), 0, 1))
    alpha = ((radius <= 0.92) & (radius >= 0.45)).astype(np.float32)
    pixels = np.dstack([*channels, alpha]) * 255
    return Image.fromarray(pixels.astype(np.uint8), "RGBA").resize((size, size), Image.LANCZOS)


def save_ico(path: str) -> None:
    make_icon(256).save(path, format="ICO", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (256, 256)])
