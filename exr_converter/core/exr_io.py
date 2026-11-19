"""Read EXR files into float32 RGB arrays (display window size) plus header info.

Uses the OpenEXR Python bindings (3.3+). If they are missing, falls back to decoding with the
bundled ffmpeg, which reads PIZ/ZIP/DWA EXRs but gives less header information.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .resources import ffmpeg_path

AP1_CHROMATICITIES = (0.713, 0.293, 0.165, 0.830, 0.128, 0.044, 0.32168, 0.33767)


@dataclass
class ExrInfo:
    path: Path
    width: int
    height: int
    channels: str = ""
    pixel_type: str = ""
    compression: str = ""
    fps: Optional[float] = None
    timecode: Optional[str] = None
    chromaticities: Optional[Tuple[float, ...]] = None
    data_window: Optional[Tuple[int, int, int, int]] = None
    display_window: Optional[Tuple[int, int, int, int]] = None
    reader: str = ""
    notes: list = field(default_factory=list)

    @property
    def chroma_label(self) -> str:
        if self.chromaticities is None:
            return "not in header (assumed ACEScg)"
        if np.allclose(self.chromaticities, AP1_CHROMATICITIES, atol=2e-3):
            return "AP1 (ACEScg)"
        if np.allclose(self.chromaticities[:6], (0.7347, 0.2653, 0.0, 1.0, 0.0001, -0.077), atol=2e-3):
            return "AP0 (ACES2065-1), not ACEScg!"
        if np.allclose(self.chromaticities[:6], (0.64, 0.33, 0.30, 0.60, 0.15, 0.06), atol=2e-3):
            return "Rec.709 primaries, not ACEScg!"
        return "custom: " + ", ".join(f"{c:.3f}" for c in self.chromaticities)


def _box(value: Any) -> Optional[Tuple[int, int, int, int]]:
    try:
        flat = np.asarray(value, dtype=np.int64).ravel()
        if flat.size == 4:
            return tuple(int(v) for v in flat)  # xmin, ymin, xmax, ymax
    except Exception:  # noqa: BLE001
        pass
    for attrs in (("min", "max"),):
        try:
            mn, mx = getattr(value, attrs[0]), getattr(value, attrs[1])
            return (int(mn.x), int(mn.y), int(mx.x), int(mx.y))
        except Exception:  # noqa: BLE001
            pass
    return None


def _fps(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if isinstance(value, (tuple, list)) and len(value) == 2:
            return float(Fraction(int(value[0]), int(value[1])))
        if hasattr(value, "numerator") and hasattr(value, "denominator"):
            return float(Fraction(int(value.numerator), int(value.denominator)))
        if hasattr(value, "n") and hasattr(value, "d"):
            return float(Fraction(int(value.n), int(value.d)))
        return float(value)
    except Exception:  # noqa: BLE001
        return None


def _timecode(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        h, m, s, f = (int(getattr(value, k)) for k in ("hours", "minutes", "seconds", "frame"))
        sep = ";" if getattr(value, "dropFrame", False) else ":"
        return f"{h:02d}:{m:02d}:{s:02d}{sep}{f:02d}"
    except Exception:  # noqa: BLE001
        pass
    if isinstance(value, (tuple, list)) and value:
        return None
    return None


def _chroma(value: Any) -> Optional[Tuple[float, ...]]:
    if value is None:
        return None
    try:
        flat = np.asarray(value, dtype=np.float64).ravel()
        if flat.size == 8:
            return tuple(float(v) for v in flat)
    except Exception:  # noqa: BLE001
        pass
    try:
        pts = [value.red, value.green, value.blue, value.white]
        return tuple(float(c) for p in pts for c in (p.x, p.y))
    except Exception:  # noqa: BLE001
        return None


def _pick_rgb(chans: Dict[str, Any]) -> Tuple[np.ndarray, str]:
    def px(key):
        return np.asarray(chans[key].pixels if hasattr(chans[key], "pixels") else chans[key])

    keys = list(chans.keys())
    for k in ("RGBA", "RGB"):
        if k in chans:
            arr = px(k)
            return arr[..., :3], k
    for k in keys:
        if k.endswith(".RGBA") or k.endswith(".RGB"):
            arr = px(k)
            return arr[..., :3], k
    if all(c in chans for c in "RGB"):
        return np.stack([px("R"), px("G"), px("B")], axis=-1), "R,G,B"
    if "Y" in chans:
        y = px("Y")
        return np.stack([y, y, y], axis=-1), "Y"
    raise ValueError(f"No RGB channels found (channels: {', '.join(keys)})")


def read_exr_openexr(path: Path) -> Tuple[np.ndarray, ExrInfo]:
    import OpenEXR  # noqa: N813

    with OpenEXR.File(str(path)) as f:
        hdr = dict(f.header())
        chans = f.channels()
        rgb, chan_label = _pick_rgb(chans)
        pixel_type = str(rgb.dtype)
        rgb = rgb.astype(np.float32)
        names = ", ".join(chans.keys())

    dw = _box(hdr.get("dataWindow"))
    disp = _box(hdr.get("displayWindow")) or dw
    if disp is None:
        disp = (0, 0, rgb.shape[1] - 1, rgb.shape[0] - 1)
    width, height = disp[2] - disp[0] + 1, disp[3] - disp[1] + 1
    info = ExrInfo(
        path=path,
        width=width,
        height=height,
        channels=f"{names}  (using {chan_label})",
        pixel_type={"float16": "half", "float32": "float", "uint32": "uint"}.get(pixel_type, pixel_type),
        compression=str(hdr.get("compression", "")).split(".")[-1].replace("_COMPRESSION", ""),
        fps=_fps(hdr.get("framesPerSecond")),
        timecode=_timecode(hdr.get("timeCode")),
        chromaticities=_chroma(hdr.get("chromaticities")),
        data_window=dw,
        display_window=disp,
        reader="OpenEXR",
    )
    if dw is not None and dw != disp:
        canvas = np.zeros((height, width, 3), np.float32)
        x0, y0 = dw[0] - disp[0], dw[1] - disp[1]
        sx0, sy0 = max(0, -x0), max(0, -y0)
        dx0, dy0 = max(0, x0), max(0, y0)
        w = min(rgb.shape[1] - sx0, width - dx0)
        h = min(rgb.shape[0] - sy0, height - dy0)
        if w > 0 and h > 0:
            canvas[dy0 : dy0 + h, dx0 : dx0 + w] = rgb[sy0 : sy0 + h, sx0 : sx0 + w]
        rgb = canvas
        info.notes.append("Data window differs from display window; output uses the display window.")
    return rgb, info


_STREAM_RE = re.compile(r"Video: exr.*?, (\w+)(?:\([^)]*\))?, (\d+)x(\d+)")


def read_exr_ffmpeg(path: Path) -> Tuple[np.ndarray, ExrInfo]:
    ff = ffmpeg_path()
    probe = subprocess.run([ff, "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    m = _STREAM_RE.search(probe.stderr)
    if not m:
        raise ValueError(f"Could not read EXR: {path}\n{probe.stderr[-400:]}")
    w, h = int(m.group(2)), int(m.group(3))
    run = subprocess.run(
        [ff, "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "gbrpf32le", "-"],
        capture_output=True,
    )
    if run.returncode != 0:
        raise ValueError(run.stderr.decode(errors="replace")[-400:])
    planes = np.frombuffer(run.stdout, dtype="<f4").reshape(3, h, w)
    rgb = np.stack([planes[2], planes[0], planes[1]], axis=-1).astype(np.float32)
    info = ExrInfo(path=path, width=w, height=h, channels="RGB", pixel_type=m.group(1), reader="ffmpeg")
    info.notes.append("Read with ffmpeg: header details (fps, timecode) unavailable.")
    return rgb, info


def read_exr(path) -> Tuple[np.ndarray, ExrInfo]:
    path = Path(path)
    try:
        import OpenEXR  # noqa: F401,N813
    except ImportError:
        return read_exr_ffmpeg(path)
    return read_exr_openexr(path)


def write_exr(path, rgb: np.ndarray) -> None:
    """Write a half-float RGB EXR (used by the self-test)."""
    path = Path(path)
    try:
        import OpenEXR  # noqa: N813

        header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
        with OpenEXR.File(header, {"RGB": rgb.astype(np.float16)}) as f:
            f.write(str(path))
        return
    except ImportError:
        pass
    h, w = rgb.shape[:2]
    planes = np.stack([rgb[..., 1], rgb[..., 2], rgb[..., 0]]).astype("<f4")
    run = subprocess.run(
        [ffmpeg_path(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gbrpf32le", "-s", f"{w}x{h}",
         "-i", "-", "-c:v", "exr", "-compression", "zip1", "-format", "half", str(path)],
        input=planes.tobytes(), capture_output=True,
    )
    if run.returncode != 0:
        raise RuntimeError(run.stderr.decode(errors="replace"))
