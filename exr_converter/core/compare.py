"""Viewer helpers that need no GUI: clip overlays, difference images, preview encoding."""

from __future__ import annotations

import io
from typing import Optional, Tuple

import numpy as np

from .output import quantize

CLIP_HIGH = np.array([255, 40, 40], np.float32)    # hits display white in a channel
CLIP_LOW = np.array([40, 110, 255], np.float32)    # hits display black in a channel
SRC_BAD = np.array([255, 0, 255], np.float32)      # negative / NaN / inf in the source EXR


def to8(display: np.ndarray) -> np.ndarray:
    return quantize(display, 8)


def clip_masks(display: np.ndarray, eps: float = 0.5 / 255) -> Tuple[np.ndarray, np.ndarray]:
    d = np.nan_to_num(display)
    high = (d >= 1.0 - eps).any(axis=-1)
    low = (d <= eps).any(axis=-1) & ~high
    return high, low


def overlay(img8: np.ndarray, display: np.ndarray, src_bad: Optional[np.ndarray] = None,
            transparent: Optional[np.ndarray] = None) -> np.ndarray:
    """Tint clipped pixels: red = white clip, blue = black clip, magenta = bad source values.
    `transparent` (alpha == 0) pixels are not marked as black clip."""
    out = img8.astype(np.float32)
    high, low = clip_masks(display)
    if transparent is not None:
        low = low & ~transparent
    for mask, col in ((high, CLIP_HIGH), (low, CLIP_LOW), (src_bad, SRC_BAD)):
        if mask is not None and mask.any():
            out[mask] = out[mask] * 0.35 + col * 0.65
    return out.astype(np.uint8)


def clip_stats(display: np.ndarray, src_bad: Optional[np.ndarray] = None) -> str:
    high, low = clip_masks(display)
    n = high.size
    parts = [f"white clip {100 * high.sum() / n:.2f}%", f"black clip {100 * low.sum() / n:.2f}%"]
    if src_bad is not None:
        parts.append(f"negative/invalid source {100 * src_bad.sum() / n:.2f}%")
    return " · ".join(parts)


def difference(a: np.ndarray, b: np.ndarray, gain: float = 10.0) -> Tuple[np.ndarray, float, float]:
    """|A - B| * gain as uint8, plus max and mean difference in 8-bit code values."""
    if a.shape != b.shape:
        raise ValueError("A and B have different sizes")
    d = np.abs(np.clip(np.nan_to_num(a), 0, 1) - np.clip(b, 0, 1))
    img = np.clip(d * gain * 255.0, 0, 255).astype(np.uint8)
    return img, float(d.max() * 255.0), float(d.mean() * 255.0)


def preview_encode(display: np.ndarray, fmt: str, jpeg_quality: int = 95) -> np.ndarray:
    """What the file will contain once written (float 0..1), without touching the disk."""
    if fmt == "png16":
        return quantize(display, 16).astype(np.float32) / 65535.0
    q8 = quantize(display, 8)
    if fmt == "jpeg":
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(q8, "RGB").save(buf, "JPEG", quality=int(jpeg_quality), subsampling=0)
        buf.seek(0)
        with Image.open(buf) as im:
            q8 = np.asarray(im.convert("RGB"))
    return q8.astype(np.float32) / 255.0


def alpha_view(alpha: np.ndarray) -> np.ndarray:
    """Alpha as a grey uint8 image."""
    a8 = quantize(alpha, 8)
    return np.repeat(a8[..., None], 3, axis=-1)


def preview_alpha(alpha: np.ndarray, fmt: str) -> np.ndarray:
    """Alpha as it will be stored in a PNG of this format (float 0..1)."""
    bits = 16 if fmt == "png16" else 8
    return quantize(alpha, bits).astype(np.float32) / ((1 << bits) - 1)
