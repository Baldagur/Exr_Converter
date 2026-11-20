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
    """name.exr -> name_<suffix><ext>; shot.1001.exr -> shot_<suffix>.1001<ext> (frame number stays
    last so Nuke/RV still see a sequence). Never returns an existing file or the source itself."""
    from .sequence import split_frame

    src = Path(src)
    folder = Path(out_dir) if out_dir else src.parent
    parts = split_frame(src.name)
    if parts:
        head, digits, _tail = parts
        sep = head[-1]
        stem = head[:-1].rstrip("._-") or "image"

        def name(n: int) -> str:
            extra = f"_{n}" if n > 1 else ""
            return f"{stem}_{suffix}{extra}{sep}{digits}{ext}"
    else:
        def name(n: int) -> str:
            extra = f"_{n}" if n > 1 else ""
            return f"{src.stem}_{suffix}{extra}{ext}"

    n = 1
    p = folder / name(n)
    while p.exists() or p.resolve() == src.resolve():
        n += 1
        p = folder / name(n)
    return p


def quantize(display: np.ndarray, bits: int) -> np.ndarray:
    top = (1 << bits) - 1
    q = np.rint(np.clip(np.nan_to_num(display), 0.0, 1.0) * top)
    return q.astype(np.uint16 if bits > 8 else np.uint8)


# ---------------------------------------------------------------------------- PNG
def _chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def png_bytes(img: np.ndarray, srgb_tag: bool = False) -> bytes:
    """Encode an RGB or RGBA PNG (straight alpha), 8-bit (uint8) or 16-bit (uint16)."""
    h, w, nc = img.shape
    color_type = 6 if nc == 4 else 2
    bits = 16 if img.dtype == np.uint16 else 8
    rows = img.astype(">u2" if bits == 16 else np.uint8).reshape(h, -1).view(np.uint8)
    raw = np.concatenate([np.zeros((h, 1), np.uint8), rows], axis=1).tobytes()  # filter 0 per row
    out = bytearray(b"\x89PNG\r\n\x1a\n")
    out += _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, bits, color_type, 0, 0, 0))
    if srgb_tag:
        out += _chunk(b"sRGB", b"\x00")
    out += _chunk(b"IDAT", zlib.compress(raw, 6))
    out += _chunk(b"IEND", b"")
    return bytes(out)


def write_png(path: Path, img: np.ndarray, srgb_tag: bool = False) -> None:
    """Write a PNG; fails with FileExistsError instead of overwriting."""
    with open(path, "xb") as fh:
        fh.write(png_bytes(img, srgb_tag))


def _read_png_own(data: bytes) -> Optional[np.ndarray]:
    """Read the simple PNGs this tool writes (RGB/RGBA, filter 0). Returns None for anything else."""
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
    if hdr is None or hdr[3] not in (2, 6) or hdr[6] != 0:
        return None
    w, h, bits = hdr[0], hdr[1], hdr[2]
    nc = 4 if hdr[3] == 6 else 3
    bpp = nc * (2 if bits == 16 else 1)
    raw = np.frombuffer(zlib.decompress(idat), np.uint8).reshape(h, 1 + w * bpp)
    if raw[:, 0].any():
        return None
    px = raw[:, 1:]
    if bits == 16:
        return px.copy().view(">u2").reshape(h, w, nc).astype(np.uint16)
    return px.reshape(h, w, nc).copy()


def read_image_rgba(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray], int]:
    """Read a PNG/JPEG as float 0..1 RGB, alpha (or None) and bit depth."""
    path = Path(path)
    data = path.read_bytes()
    own = _read_png_own(data)
    if own is not None:
        bits = 16 if own.dtype == np.uint16 else 8
        f = own.astype(np.float32) / ((1 << bits) - 1)
        return f[..., :3], (f[..., 3] if f.shape[-1] == 4 else None), bits
    from PIL import Image

    with Image.open(path) as im:
        if im.mode in ("I;16", "I;16B", "I"):
            arr = np.asarray(im, dtype=np.float32) / 65535.0
            return np.repeat(arr[..., None], 3, axis=-1), None, 16
        if im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info:
            arr = np.asarray(im.convert("RGBA"), dtype=np.float32) / 255.0
            return arr[..., :3], arr[..., 3], 8
        return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0, None, 8


def read_image(path: Path) -> Tuple[np.ndarray, int]:
    """Read a PNG/JPEG as float 0..1 RGB plus its bit depth (alpha dropped)."""
    rgb, _alpha, bits = read_image_rgba(path)
    return rgb, bits


def write_still(path: Path, display: np.ndarray, fmt: str, output: str, jpeg_quality: int = 95,
                alpha: Optional[np.ndarray] = None) -> np.ndarray:
    """Quantise and write. `alpha` (0..1, straight) is written for PNG only.
    Returns the integer image that was written."""
    path = Path(path)
    if fmt in ("png16", "png8"):
        bits = 16 if fmt == "png16" else 8
        img = quantize(display, bits)
        if alpha is not None:
            img = np.concatenate([img, quantize(alpha[..., None], bits)], axis=-1)
        write_png(path, img, srgb_tag=(output == "srgb"))
    elif fmt == "jpeg":
        from PIL import Image

        img = quantize(display, 8)
        with open(path, "xb") as fh:  # exclusive create: never overwrites
            try:
                Image.fromarray(img, "RGB").save(fh, "JPEG", quality=int(jpeg_quality), subsampling=0,
                                                 optimize=True)
            except BaseException:
                fh.close()
                path.unlink()
                raise
    else:
        raise ValueError(f"Unknown format {fmt}")
    return img
