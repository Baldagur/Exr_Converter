"""Still image output: naming, quantising and PNG/JPEG writing (and reading back for the viewer)."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

FORMATS = {
    "png16": ("PNG 16-bit", ".png"),
    "png8": ("PNG 8-bit", ".png"),
    "jpeg": ("JPEG", ".jpg"),
}


def output_path(src: Path, suffix: str, ext: str, out_dir: Optional[Path] = None) -> Path:
    """name.exr -> name_<suffix><ext>. Never returns an existing file or the source itself."""
    src = Path(src)
    folder = Path(out_dir) if out_dir else src.parent
    base = f"{src.stem}_{suffix}"
    p = folder / f"{base}{ext}"
    n = 2
    while p.exists() or p.resolve() == src.resolve():
        p = folder / f"{base}_{n}{ext}"
        n += 1
    return p


def quantize(display: np.ndarray, bits: int) -> np.ndarray:
    top = (1 << bits) - 1
    q = np.rint(np.clip(np.nan_to_num(display), 0.0, 1.0) * top)
    return q.astype(np.uint16 if bits > 8 else np.uint8)


# ---------------------------------------------------------------------------- PNG
def _chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def write_png(path: Path, img: np.ndarray, srgb_tag: bool = False) -> None:
    """Write an RGB PNG, 8-bit (uint8) or 16-bit (uint16)."""
    h, w = img.shape[:2]
    bits = 16 if img.dtype == np.uint16 else 8
    rows = img.astype(">u2" if bits == 16 else np.uint8).reshape(h, -1).view(np.uint8)
    raw = np.concatenate([np.zeros((h, 1), np.uint8), rows], axis=1).tobytes()  # filter 0 per row
    out = bytearray(b"\x89PNG\r\n\x1a\n")
    out += _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, bits, 2, 0, 0, 0))
    if srgb_tag:
        out += _chunk(b"sRGB", b"\x00")
    out += _chunk(b"IDAT", zlib.compress(raw, 6))
    out += _chunk(b"IEND", b"")
    Path(path).write_bytes(bytes(out))


def _read_png_own(data: bytes) -> Optional[np.ndarray]:
    """Read the simple PNGs this tool writes (RGB, filter 0). Returns None for anything else."""
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return None
    pos, idat, hdr = 8, b"", None
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        tag = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + length]
        if tag == b"IHDR":
            hdr = struct.unpack(">IIBBBBB", body)
        elif tag == b"IDAT":
            idat += body
        pos += 12 + length
    if hdr is None or hdr[3] != 2 or hdr[6] != 0:
        return None
    w, h, bits = hdr[0], hdr[1], hdr[2]
    bpp = 6 if bits == 16 else 3
    raw = np.frombuffer(zlib.decompress(idat), np.uint8).reshape(h, 1 + w * bpp)
    if raw[:, 0].any():
        return None
    px = raw[:, 1:]
    if bits == 16:
        return px.copy().view(">u2").reshape(h, w, 3).astype(np.uint16)
    return px.reshape(h, w, 3).copy()


def read_image(path: Path) -> Tuple[np.ndarray, int]:
    """Read a PNG/JPEG as float 0..1 RGB plus its bit depth."""
    path = Path(path)
    data = path.read_bytes()
    own = _read_png_own(data)
    if own is not None:
        bits = 16 if own.dtype == np.uint16 else 8
        return own.astype(np.float32) / ((1 << bits) - 1), bits
    from PIL import Image

    with Image.open(path) as im:
        if im.mode in ("I;16", "I;16B", "I"):
            arr = np.asarray(im, dtype=np.float32) / 65535.0
            return np.repeat(arr[..., None], 3, axis=-1), 16
        arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        return arr, 8


def write_still(path: Path, display: np.ndarray, fmt: str, output: str, jpeg_quality: int = 95) -> np.ndarray:
    """Quantise and write. Returns the integer image that was written."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    if fmt == "png16":
        img = quantize(display, 16)
        write_png(path, img, srgb_tag=(output == "srgb"))
    elif fmt == "png8":
        img = quantize(display, 8)
        write_png(path, img, srgb_tag=(output == "srgb"))
    elif fmt == "jpeg":
        from PIL import Image

        img = quantize(display, 8)
        Image.fromarray(img, "RGB").save(path, "JPEG", quality=int(jpeg_quality), subsampling=0, optimize=True)
    else:
        raise ValueError(f"Unknown format {fmt}")
    return img
