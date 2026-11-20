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
        def get(k):
            v = getattr(value, k)
            return v() if callable(v) else v

        h, m, s, f = (int(get(k)) for k in ("hours", "minutes", "seconds", "frame"))
        sep = ";" if get("dropFrame") else ":"
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


# ---------------------------------------------------------------------------------------------
# Picking the main image out of multichannel / multipart CG renders
# ---------------------------------------------------------------------------------------------
_COMP = {"R": "R", "G": "G", "B": "B", "A": "A", "Y": "Y",
         "RED": "R", "GREEN": "G", "BLUE": "B", "ALPHA": "A"}
# Layer names that mean "the main image" (last path segment, case-insensitive). "" = no prefix.
_BEAUTY = ("", "beauty", "rgba", "rgb", "combined", "main", "color", "colour", "default", "image")


def _layers(names) -> Dict[str, Dict[str, str]]:
    """{'': {'R': 'R', ...}, 'diffuse': {'R': 'diffuse.R', ...}} in order of appearance."""
    out: Dict[str, Dict[str, str]] = {}
    for n in names:
        layer, _, comp = str(n).rpartition(".")
        c = _COMP.get(comp.upper())
        if c:
            out.setdefault(layer, {}).setdefault(c, str(n))
    return out


def choose_beauty(parts) -> Tuple[int, str, Dict[str, str]]:
    """parts: [(part_name, [channel names]), ...] -> (part index, layer, {'R': name, 'G': ..., 'B': ...}).

    Picks the main RGB(A) and ignores every other AOV. Order of preference: unprefixed R/G/B, then a
    layer called beauty/rgba/combined/..., then (multipart) a part with such a name, then the
    first layer that has R, G and B, then a luminance-only Y channel.
    """
    best = None
    for pi, (pname, names) in enumerate(parts):
        pname_l = (pname or "").lower().rpartition(".")[2]
        for li, (layer, comps) in enumerate(_layers(names).items()):
            seg = layer.lower().rpartition(".")[2]
            if all(c in comps for c in "RGB"):
                if layer == "" and pname_l in _BEAUTY:  # unnamed part counts ("" is in _BEAUTY)
                    score = 0
                elif layer == "":
                    score = 1
                elif seg in _BEAUTY:
                    score = 2
                elif pname_l in _BEAUTY:
                    score = 3
                else:
                    score = 4
            elif "Y" in comps and layer == "":
                score = 5
                comps = {"R": comps["Y"], "G": comps["Y"], "B": comps["Y"], **({"A": comps["A"]} if "A" in comps else {})}
            else:
                continue
            key = (score, pi, li)
            if best is None or key < best[0]:
                best = (key, pi, layer, comps)
    if best is None:
        listing = "; ".join(f"{n or f'part {i}'}: {', '.join(map(str, ch[:12]))}{' …' if len(ch) > 12 else ''}"
                            for i, (n, ch) in enumerate(parts))
        raise ValueError(f"No RGB channels found. Channels in the file: {listing}")
    return best[1], best[2], best[3]


def _describe(parts, pi, layer, comps) -> str:
    total = sum(len(ch) for _, ch in parts)
    where = f"part '{parts[pi][0] or pi}', " if len(parts) > 1 else ""
    used = ", ".join(comps[c] for c in "RGBA" if c in comps)
    extra = f"{total} channels" + (f" in {len(parts)} parts" if len(parts) > 1 else "")
    return f"{extra}; using {where}{used}" + ("" if total <= 4 else " (other channels ignored)")


def _read_legacy(path: Path):
    """Fast path: read only the beauty channels (single-part files) with the classic API."""
    import Imath  # noqa: N813
    import OpenEXR  # noqa: N813

    f = OpenEXR.InputFile(str(path))
    try:
        hdr = f.header()
        names = list(hdr["channels"].keys())
        parts = [(str(hdr.get("name", "") or ""), names)]
        pi, layer, comps = choose_beauty(parts)
        dw = _box(hdr["dataWindow"])
        w, h = dw[2] - dw[0] + 1, dw[3] - dw[1] + 1
        ft = Imath.PixelType(Imath.PixelType.FLOAT)
        planes = [np.frombuffer(f.channel(comps[c], ft), dtype=np.float32).reshape(h, w) for c in "RGB"]
        ptype = hdr["channels"][comps["R"]].type
        pixel_type = "half" if "HALF" in str(ptype) else ("float" if "FLOAT" in str(ptype) else str(ptype))
    finally:
        f.close()
    return np.stack(planes, axis=-1), dict(hdr), _describe(parts, pi, layer, comps), pixel_type


def _read_new(path: Path):
    """Any EXR the bindings can open, including multipart: reads everything, keeps the beauty."""
    import OpenEXR  # noqa: N813

    with OpenEXR.File(str(path), separate_channels=True) as f:
        nparts = len(f.parts) if hasattr(f, "parts") else 1
        headers = [dict(f.header(i)) for i in range(nparts)]
        chans = [f.channels(i) for i in range(nparts)]
        parts = [(str(headers[i].get("name", "") or ""), list(chans[i].keys())) for i in range(nparts)]
        pi, layer, comps = choose_beauty(parts)

        def px(name):
            ch = chans[pi][name]
            return np.asarray(ch.pixels if hasattr(ch, "pixels") else ch)

        planes = [px(comps[c]) for c in "RGB"]
        pixel_type = {"float16": "half", "float32": "float", "uint32": "uint"}.get(str(planes[0].dtype), "")
        rgb = np.stack(planes, axis=-1).astype(np.float32)
    return rgb, headers[pi], _describe(parts, pi, layer, comps), pixel_type


def read_exr_openexr(path: Path) -> Tuple[np.ndarray, ExrInfo]:
    import OpenEXR  # noqa: N813

    rgb = None
    if hasattr(OpenEXR, "InputFile"):
        try:
            rgb, hdr, chan_label, pixel_type = _read_legacy(path)
        except Exception:  # noqa: BLE001  (multipart, odd types: use the full reader below)
            rgb = None
    if rgb is None:
        rgb, hdr, chan_label, pixel_type = _read_new(path)
    rgb = rgb.astype(np.float32, copy=False)

    dw = _box(hdr.get("dataWindow"))
    disp = _box(hdr.get("displayWindow")) or dw
    if disp is None:
        disp = (0, 0, rgb.shape[1] - 1, rgb.shape[0] - 1)
    width, height = disp[2] - disp[0] + 1, disp[3] - disp[1] + 1
    info = ExrInfo(
        path=path,
        width=width,
        height=height,
        channels=chan_label,
        pixel_type=pixel_type,
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
    try:
        return read_exr_openexr(path)
    except Exception as err:  # noqa: BLE001
        try:
            rgb, info = read_exr_ffmpeg(path)
        except Exception:  # noqa: BLE001
            raise ValueError(f"Could not read {path.name}: {err}") from err
        info.notes.append(f"OpenEXR could not read it ({err}); decoded with ffmpeg instead.")
        return rgb, info


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


def write_cg_test_files(folder) -> Dict[str, Tuple[Path, np.ndarray]]:
    """Write multichannel/multipart EXRs like CG renders produce (self-test and unit tests).

    Returns {label: (path, expected beauty RGB)}. Needs the OpenEXR bindings.
    """
    import OpenEXR  # noqa: N813

    folder = Path(folder)
    h, w = 24, 33
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    beauty = np.stack([xx / w * 4, yy / h, np.full_like(xx, 0.18)], axis=-1)
    other = np.stack([np.full_like(xx, 0.9), np.full_like(xx, 0.1), np.full_like(xx, 0.5)], axis=-1)
    alpha = np.ones_like(xx)

    def hdr():
        return {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}

    def rgba(prefix, img, a=True):
        d = {f"{prefix}R": img[..., 0].astype(np.float16), f"{prefix}G": img[..., 1].astype(np.float16),
             f"{prefix}B": img[..., 2].astype(np.float16)}
        if a:
            d[f"{prefix}A"] = alpha.astype(np.float16)
        return d

    expected = beauty.astype(np.float16).astype(np.float32)
    out: Dict[str, Tuple[Path, np.ndarray]] = {}

    # Arnold/Redshift style: plain RGBA plus AOV layers, a float Z and cryptomatte-like channels
    chans = {**rgba("", beauty), **rgba("diffuse.", other, False), **rgba("specular.", other, False),
             "Z": (yy * 10).astype(np.float32), "N.X": xx.astype(np.float32),
             "crypto00.R": xx.astype(np.float32), "crypto00.G": yy.astype(np.float32)}
    p = folder / "cg_aovs.exr"
    with OpenEXR.File(hdr(), chans) as f:
        f.write(str(p))
    out["AOVs + Z + crypto"] = (p, expected)

    # Blender style: no unprefixed channels
    chans = {**rgba("ViewLayer.Combined.", beauty), **rgba("ViewLayer.DiffCol.", other, False),
             "ViewLayer.Depth.Z": (yy * 10).astype(np.float32)}
    p = folder / "cg_blender.exr"
    with OpenEXR.File(hdr(), chans) as f:
        f.write(str(p))
    out["Blender ViewLayer.Combined"] = (p, expected)

    # Multipart with the beauty in the second part
    parts = [OpenEXR.Part(hdr(), rgba("", other, False), "diffuse"),
             OpenEXR.Part(hdr(), rgba("", beauty), "beauty")]
    p = folder / "cg_multipart.exr"
    with OpenEXR.File(parts) as f:
        f.write(str(p))
    out["multipart, beauty in part 2"] = (p, expected)
    return out
