"""Encode a converted EXR sequence to QuickTime (ProRes 422 / 422 HQ / H.264) with ffmpeg."""

from __future__ import annotations

import errno
import os
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from .color import ColorEngine, OUTPUT_SUFFIX, LOOK_SUFFIX
from .exr_io import read_exr
from .output import quantize
from .resources import ffmpeg_path
from .sequence import Sequence

CODECS = {
    "prores422": "ProRes 422",
    "prores422hq": "ProRes 422 HQ",
    "h264": "H.264",
}
FPS_CHOICES = ["23.976", "24", "25", "29.97", "30", "48", "50", "59.94", "60"]
_FPS_EXACT = {"23.976": "24000/1001", "23.98": "24000/1001", "29.97": "30000/1001",
              "47.952": "48000/1001", "47.95": "48000/1001", "59.94": "60000/1001", "59.95": "60000/1001"}


def fps_rational(fps: str) -> Fraction:
    fps = str(fps).strip()
    if fps in _FPS_EXACT:
        return Fraction(_FPS_EXACT[fps])
    val = Fraction(fps).limit_denominator(1001)
    for k, v in _FPS_EXACT.items():
        if abs(float(val) - float(Fraction(v))) < 0.005:
            return Fraction(v)
    return val


def fps_label(fps: Optional[float]) -> Optional[str]:
    if fps is None:
        return None
    for label in FPS_CHOICES:
        if abs(float(fps_rational(label)) - fps) < 0.002:
            return label
    return f"{fps:g}"


def frame_to_timecode(frame: int, fps: Fraction) -> str:
    nominal = max(1, round(float(fps)))
    ff = frame % nominal
    s = frame // nominal
    return f"{s // 3600 % 24:02d}:{s // 60 % 60:02d}:{s % 60:02d}:{ff:02d}"


def output_size(w: int, h: int, half: bool, codec: str) -> tuple:
    ow, oh = (round(w / 2), round(h / 2)) if half else (w, h)
    if half or codec == "h264":  # 4:2:0 needs even sizes; keep halves even too
        ow -= ow % 2
        oh -= oh % 2
    return max(2, ow), max(2, oh)


def movie_path(seq: Sequence, output: str, half: bool, look: str, add_look: bool,
               out_dir: Optional[Path] = None) -> Path:
    folder = Path(out_dir) if out_dir else seq.folder
    parts = [seq.base_name]
    if add_look:
        parts.append(LOOK_SUFFIX[look])
    parts.append(OUTPUT_SUFFIX[output])
    if half:
        parts.append("half")
    base = "_".join(parts)
    p = folder / f"{base}.mov"
    n = 2
    while p.exists():
        p = folder / f"{base}_{n}.mov"
        n += 1
    return p


def build_command(ffmpeg: str, w: int, h: int, fps: Fraction, codec: str, half: bool, output: str,
                  out_path: Path, timecode: Optional[str] = None) -> List[str]:
    ow, oh = output_size(w, h, half, codec)
    rate = f"{fps.numerator}/{fps.denominator}"
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", f"{w}x{h}", "-framerate", rate, "-i", "-"]
    pix = "yuv420p" if codec == "h264" else "yuv422p10le"
    filters = []
    if not half and (ow, oh) != (w, h):
        filters.append(f"crop={ow}:{oh}:0:0")  # parity fix only: crop 1px, don't resample
    filters.append(f"scale={ow}:{oh}:flags=lanczos+accurate_rnd+full_chroma_int:"
                   f"in_range=pc:out_range=tv:out_color_matrix=bt709")
    filters.append(f"format={pix}")
    cmd += ["-vf", ",".join(filters)]
    if codec == "h264":
        cmd += ["-c:v", "libx264", "-preset", "slow", "-crf", "16", "-profile:v", "high",
                "-movflags", "+faststart+write_colr"]
    else:
        profile = "3" if codec == "prores422hq" else "2"
        cmd += ["-c:v", "prores_ks", "-profile:v", profile, "-vendor", "apl0", "-movflags", "+write_colr"]
    trc = "iec61966-2-1" if output == "srgb" else "bt709"
    cmd += ["-color_primaries", "bt709", "-color_trc", trc, "-colorspace", "bt709", "-color_range", "tv"]
    if timecode:
        cmd += ["-timecode", timecode]
    cmd += ["-r", rate, "-an", str(out_path)]
    return cmd


@dataclass
class MovieJob:
    seq: Sequence
    start: int
    end: int
    look: str
    output: str
    codec: str
    half: bool
    fps: str
    out_path: Path
    use_header_timecode: bool = True


class MovieCancelled(Exception):
    pass


def encode_movie(job: MovieJob, engine: ColorEngine,
                 progress: Optional[Callable[[int, int], None]] = None,
                 cancel: Optional[threading.Event] = None) -> Path:
    if job.end < job.start:
        raise ValueError("Out frame is before in frame.")
    missing = job.seq.missing(job.start, job.end)
    if missing:
        shown = ", ".join(str(f) for f in missing[:10]) + (" …" if len(missing) > 10 else "")
        raise FileNotFoundError(f"{len(missing)} frame(s) missing in {job.start}-{job.end}: {shown}")
    if job.out_path.exists():
        raise FileExistsError(f"Refusing to overwrite {job.out_path}")

    fps = fps_rational(job.fps)
    first, info = read_exr(job.seq.path(job.start))
    w, h = info.width, info.height
    tc = info.timecode if (job.use_header_timecode and info.timecode) else frame_to_timecode(job.start, fps)
    # Encode to a private temp name, then link it into place without overwriting. On failure only
    # the temp file (ours) is removed, never a file someone else created under the final name.
    tmp = job.out_path.with_name(f".{job.out_path.stem}.{os.getpid()}.{threading.get_ident()}.partial.mov")
    cmd = build_command(ffmpeg_path(), w, h, fps, job.codec, job.half, job.output, tmp, tc)
    total = job.end - job.start + 1
    with tempfile.TemporaryFile() as errlog:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=errlog)

        def ffmpeg_error() -> str:
            errlog.seek(0)
            return errlog.read().decode(errors="replace")[-800:]

        try:
            for i, frame in enumerate(range(job.start, job.end + 1)):
                if cancel is not None and cancel.is_set():
                    raise MovieCancelled()
                rgb = first if i == 0 else read_exr(job.seq.path(frame))[0]
                if rgb.shape[:2] != (h, w):
                    raise ValueError(f"Frame {frame} is {rgb.shape[1]}x{rgb.shape[0]}, expected {w}x{h}.")
                disp = engine.convert(rgb, job.look, job.output)
                try:
                    proc.stdin.write(quantize(disp, 16).astype("<u2").tobytes())
                except BrokenPipeError:
                    proc.wait()
                    raise RuntimeError(f"ffmpeg stopped:\n{ffmpeg_error()}")
                if progress:
                    progress(i + 1, total)
            proc.stdin.close()
            if proc.wait() != 0:
                raise RuntimeError(f"ffmpeg failed:\n{ffmpeg_error()}")
            _link_into_place(tmp, job.out_path)
        except BaseException:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            _remove(tmp)
            raise
    return job.out_path


def _link_into_place(tmp: Path, final: Path) -> None:
    try:
        os.link(tmp, final)  # atomic, raises FileExistsError instead of overwriting
    except FileExistsError:
        raise FileExistsError(f"{final.name} appeared while encoding; your movie was not saved over it.")
    except OSError as err:
        if err.errno not in (errno.EPERM, errno.EXDEV, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EMLINK):
            raise
        if final.exists():  # filesystem without hard links: best effort
            raise FileExistsError(f"{final.name} appeared while encoding; your movie was not saved over it.")
        os.replace(tmp, final)
        return
    tmp.unlink()


def _remove(p: Path) -> None:
    try:
        Path(p).unlink()
    except FileNotFoundError:
        pass


def probe(path: Path) -> str:
    """ffmpeg's stream description of a file (used by the self-test)."""
    r = subprocess.run([ffmpeg_path(), "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    return r.stderr
