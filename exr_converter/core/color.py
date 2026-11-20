"""Colour pipeline: ACEScg scene-linear -> display code values (0..1 float).

Two looks:
  * ARRI K1S1  : ACEScg -> ARRI LogC3 (EI 800) / AWG3 -> K1S1 3D LUT (Rec.709 gamma 2.4, full range)
  * ACES 1.0   : the ACES 1.x output transform (RRT + ODT) from OCIO's built-in ACES studio config

Two outputs:
  * rec709 : Rec.709 primaries, gamma 2.4 (BT.1886) encoding
  * srgb   : sRGB primaries (same as Rec.709), sRGB piecewise encoding

OCIO does the work when it is installed (the normal case). A small numpy implementation of the
K1S1 chain is kept as a reference for tests and as a fallback; ACES always needs OCIO.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from .resources import k1s1_lut

LOOKS: Dict[str, str] = {
    "k1s1": "ARRI K1S1 (LogC3 EI800)",
    "aces": "ACES 1.0 SDR (RRT + ODT)",
}
OUTPUTS: Dict[str, str] = {
    "rec709": "Rec.709 (gamma 2.4)",
    "srgb": "sRGB",
}
# File-name suffix per output, as requested: name_rec709.png / name_sRGB.png
OUTPUT_SUFFIX: Dict[str, str] = {"rec709": "rec709", "srgb": "sRGB"}
LOOK_SUFFIX: Dict[str, str] = {"k1s1": "K1S1", "aces": "ACES"}

# --------------------------------------------------------------------------------------------
# Reference maths (numpy)
# --------------------------------------------------------------------------------------------
AP1_TO_AP0 = np.array(
    [
        [0.6954522414, 0.1406786965, 0.1638690622],
        [0.0447945634, 0.8596711185, 0.0955343182],
        [-0.0055258826, 0.0040252103, 1.0015006723],
    ]
)
# ARRI's AWG3 -> ACES AP0 (the matrix used by the ACES IDT and the OCIO studio config)
AWG3_TO_AP0 = np.array(
    [
        [0.680205505106279, 0.236136601606481, 0.083658023297240],
        [0.085415040775843, 1.017470870763540, -0.102885911539384],
        [0.002056997564970, -0.062563321311908, 1.060506323746940],
    ]
)
ACESCG_TO_AWG3 = np.linalg.inv(AWG3_TO_AP0) @ AP1_TO_AP0

# LogC3, EI 800
_LC3 = dict(cut=0.010591, a=5.555556, b=0.052272, c=0.247190, d=0.385537, e=5.367655, f=0.092809)


def logc3_encode(x: np.ndarray) -> np.ndarray:
    p = _LC3
    lin = np.maximum(x, -1e9)
    with np.errstate(invalid="ignore", divide="ignore"):
        log = p["c"] * np.log10(np.maximum(p["a"] * lin + p["b"], 1e-12)) + p["d"]
    return np.where(lin > p["cut"], log, p["e"] * lin + p["f"]).astype(np.float32)


def bt1886_to_srgb(v: np.ndarray) -> np.ndarray:
    """Re-encode gamma-2.4 display values as sRGB (same primaries). Sign is preserved."""
    s = np.sign(v)
    lin = np.abs(v) ** 2.4
    enc = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
    return (s * enc).astype(np.float32)


@dataclass
class Lut3D:
    size: int
    table: np.ndarray  # (size, size, size, 3) indexed [b, g, r]
    domain_min: np.ndarray
    domain_max: np.ndarray

    @classmethod
    def load_cube(cls, path: Path) -> "Lut3D":
        size = 0
        dmin = np.zeros(3)
        dmax = np.ones(3)
        rows = []
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                key = line.split()[0].upper()
                if key == "LUT_3D_SIZE":
                    size = int(line.split()[1])
                elif key == "DOMAIN_MIN":
                    dmin = np.array([float(x) for x in line.split()[1:4]])
                elif key == "DOMAIN_MAX":
                    dmax = np.array([float(x) for x in line.split()[1:4]])
                elif key in ("TITLE", "LUT_1D_SIZE", "LUT_1D_INPUT_RANGE", "LUT_3D_INPUT_RANGE"):
                    if key.startswith("LUT_1D"):
                        raise ValueError("1D LUTs are not supported")
                    continue
                else:
                    rows.append([float(x) for x in line.split()[:3]])
        data = np.asarray(rows, dtype=np.float32)
        if size == 0 or data.shape[0] != size**3:
            raise ValueError(f"Bad .cube file: {path}")
        return cls(size, data.reshape(size, size, size, 3), dmin, dmax)

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        """Tetrahedral interpolation (what OCIO and Nuke use for 3D LUTs)."""
        n = self.size - 1
        shape = rgb.shape
        x = ((rgb.reshape(-1, 3) - self.domain_min) / (self.domain_max - self.domain_min)) * n
        x = np.clip(np.nan_to_num(x), 0, n)
        i0 = np.minimum(np.floor(x).astype(np.int64), n - 1)
        f = x - i0
        r0, g0, b0 = i0[:, 0], i0[:, 1], i0[:, 2]
        fr, fg, fb = f[:, 0:1], f[:, 1:2], f[:, 2:3]
        T = self.table

        def c(dr, dg, db):
            return T[b0 + db, g0 + dg, r0 + dr]

        c000, c111 = c(0, 0, 0), c(1, 1, 1)
        out = np.empty_like(x, dtype=np.float32)
        conds = [
            (fr >= fg) & (fg >= fb),
            (fr >= fb) & (fb >= fg),
            (fb >= fr) & (fr >= fg),
            (fg >= fr) & (fr >= fb),
            (fg >= fb) & (fb >= fr),
            (fb >= fg) & (fg >= fr),
        ]
        # (first corner offset, second corner offset, weights)
        res = np.zeros_like(out)
        done = np.zeros((x.shape[0], 1), dtype=bool)
        cases = [
            ((1, 0, 0), (1, 1, 0), fr, fg, fb),
            ((1, 0, 0), (1, 0, 1), fr, fb, fg),
            ((0, 0, 1), (1, 0, 1), fb, fr, fg),
            ((0, 1, 0), (1, 1, 0), fg, fr, fb),
            ((0, 1, 0), (0, 1, 1), fg, fb, fr),
            ((0, 0, 1), (0, 1, 1), fb, fg, fr),
        ]
        for cond, (a, b, w1, w2, w3) in zip(conds, cases):
            m = cond & ~done
            if not m.any():
                continue
            ca, cb = c(*a), c(*b)
            val = (1 - w1) * c000 + (w1 - w2) * ca + (w2 - w3) * cb + w3 * c111
            res = np.where(m, val, res)
            done |= m
        return res.reshape(shape).astype(np.float32)


_REF_LUT: Optional[Lut3D] = None


def reference_k1s1(acescg: np.ndarray) -> np.ndarray:
    """Pure numpy ACEScg -> K1S1 Rec.709 gamma 2.4 display values."""
    global _REF_LUT
    if _REF_LUT is None:
        _REF_LUT = Lut3D.load_cube(k1s1_lut())
    awg = acescg.reshape(-1, 3) @ ACESCG_TO_AWG3.T
    log = logc3_encode(awg)
    return _REF_LUT.apply(log).reshape(acescg.shape)


# --------------------------------------------------------------------------------------------
# OCIO engine
# --------------------------------------------------------------------------------------------
def _pick(names, *wanted, contains: Tuple[str, ...] = ()) -> Optional[str]:
    names = list(names)
    for w in wanted:
        if w in names:
            return w
    for n in names:
        if all(c.lower() in n.lower() for c in contains):
            return n
    return None


class ColorEngine:
    """Builds and caches the transforms. Call `convert(acescg, look, output)`."""

    def __init__(self, use_ocio: bool = True):
        self.ocio = None
        self.config = None
        self.config_name = "numpy reference (K1S1 only)"
        self._cache: Dict[Tuple[str, str], Callable[[np.ndarray], np.ndarray]] = {}
        if use_ocio:
            try:
                import PyOpenColorIO as OCIO  # noqa: N813

                self.ocio = OCIO
                self.config, self.config_name = self._load_config()
            except Exception as err:  # noqa: BLE001  (keep K1S1 working without OCIO)
                self.ocio = None
                self.config = None
                self.config_name = f"numpy reference (K1S1 only) - OCIO unavailable: {err}"

    # -- config -----------------------------------------------------------------------------
    def _load_config(self):
        OCIO = self.ocio
        names = []
        try:
            for item in OCIO.BuiltinConfigRegistry().getBuiltinConfigs():
                names.append(item[0] if isinstance(item, (tuple, list)) else str(item))
        except Exception:
            pass
        # ACES 1.3 studio configs, newest first. "latest" may point at ACES 2.0, so pick explicitly.
        aces13 = sorted([n for n in names if "studio-config" in n and "aces-v1.3" in n], reverse=True)
        candidates = aces13 + [
            "studio-config-v2.2.0_aces-v1.3_ocio-v2.4",
            "studio-config-v2.1.0_aces-v1.3_ocio-v2.3",
            "studio-config-v1.0.0_aces-v1.3_ocio-v2.1",
        ]
        last_err = None
        for name in candidates:
            try:
                return OCIO.Config.CreateFromBuiltinConfig(name), name
            except Exception as err:  # noqa: BLE001
                last_err = err
        raise RuntimeError(f"No ACES 1.3 studio config found in OCIO ({last_err})")

    @property
    def has_aces(self) -> bool:
        return self.config is not None

    def describe(self) -> str:
        return f"OCIO {self.ocio.__version__} · {self.config_name}" if self.ocio else self.config_name

    # -- processors -------------------------------------------------------------------------
    def _cpu(self, transform):
        proc = self.config.getProcessor(transform)
        return proc.getOptimizedCPUProcessor(self.ocio.OPTIMIZATION_DEFAULT)

    def _apply_cpu(self, cpu, rgb: np.ndarray) -> np.ndarray:
        buf = np.ascontiguousarray(rgb, dtype=np.float32).copy()
        cpu.applyRGB(buf)
        return buf

    def _build(self, look: str, output: str) -> Callable[[np.ndarray], np.ndarray]:
        if look not in LOOKS or output not in OUTPUTS:
            raise ValueError(f"Unknown look/output: {look}/{output}")
        if self.config is None:
            if look != "k1s1":
                raise RuntimeError("ACES output needs OpenColorIO, which is not available.")
            if output == "srgb":
                return lambda x: bt1886_to_srgb(reference_k1s1(x))
            return reference_k1s1

        OCIO = self.ocio
        cfg = self.config
        cs_names = [cs.getName() for cs in cfg.getColorSpaces()]
        src = _pick(cs_names, "ACEScg", contains=("acescg",))
        if src is None:
            raise RuntimeError("ACEScg colour space not found in the OCIO config")

        if look == "k1s1":
            logc = _pick(cs_names, "ARRI LogC3 (EI800)", contains=("logc3", "800"))
            if logc is None:
                raise RuntimeError("ARRI LogC3 (EI800) colour space not found in the OCIO config")
            group = OCIO.GroupTransform()
            group.appendTransform(OCIO.ColorSpaceTransform(src=src, dst=logc))
            ft = OCIO.FileTransform(src=str(k1s1_lut()), interpolation=OCIO.INTERP_TETRAHEDRAL)
            group.appendTransform(ft)
            cpu = self._cpu(group)
            if output == "srgb":
                return lambda x: bt1886_to_srgb(self._apply_cpu(cpu, x))
            return lambda x: self._apply_cpu(cpu, x)

        # ACES 1.0 output transform
        displays = list(cfg.getDisplays())
        if output == "rec709":
            disp = _pick(displays, "Rec.1886 Rec.709 - Display", contains=("rec.1886",)) or _pick(
                displays, contains=("709",)
            )
        else:
            disp = _pick(displays, "sRGB - Display", contains=("srgb",))
        if disp is None:
            raise RuntimeError(f"No display for {output} in the OCIO config")
        views = list(cfg.getViews(disp))
        view = _pick(views, "ACES 1.0 - SDR Video", contains=("aces 1.0", "sdr"))
        if view is None:
            raise RuntimeError(f"No ACES 1.0 SDR view for display {disp}")
        dvt = OCIO.DisplayViewTransform(src=src, display=disp, view=view)
        cpu = self._cpu(dvt)
        return lambda x: self._apply_cpu(cpu, x)

    def transform(self, look: str, output: str) -> Callable[[np.ndarray], np.ndarray]:
        key = (look, output)
        if key not in self._cache:
            self._cache[key] = self._build(look, output)
        return self._cache[key]

    def convert(self, acescg: np.ndarray, look: str, output: str) -> np.ndarray:
        """ACEScg float (H, W, 3) -> display float (H, W, 3) in 0..1.

        The LUT and the ACES ODT already end in 0..1; OCIO's ACES sRGB display can leave small
        negatives for out-of-gamut colours, which the ACES CTL clamps, so clamp here too.
        """
        clean = sanitize(acescg)
        return np.clip(self.transform(look, output)(clean), 0.0, 1.0)


def sanitize(rgb: np.ndarray) -> np.ndarray:
    """Replace NaN with 0 and +/-inf with the half-float limits."""
    return np.nan_to_num(rgb.astype(np.float32, copy=False), nan=0.0, posinf=65504.0, neginf=-65504.0)


def source_problem_mask(rgb: np.ndarray) -> np.ndarray:
    """Pixels that are NaN/inf or negative (outside the ACEScg gamut) in any channel."""
    return (~np.isfinite(rgb)).any(axis=-1) | (np.nan_to_num(rgb) < 0).any(axis=-1)


def unpremultiply(rgb: np.ndarray, alpha: np.ndarray, threshold: float = 1e-6) -> np.ndarray:
    """Divide premultiplied RGB by alpha (like Nuke's Unpremult); alpha 0 pixels are left alone."""
    a = alpha[..., None].astype(np.float32)
    safe = np.where(a > threshold, a, 1.0)
    return np.where(a > threshold, rgb / safe, rgb).astype(np.float32)
