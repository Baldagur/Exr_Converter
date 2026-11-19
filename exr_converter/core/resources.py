"""Find files bundled with the app, both from source and inside the PyInstaller build."""

import os
import shutil
import sys
from pathlib import Path

K1S1_LUT_NAME = "ARRI_K1S1_LogC3-EI800_AWG3_to_Rec709-g24_full_33.cube"


def base_dir() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent.parent


def resource(*parts: str) -> Path:
    return base_dir().joinpath("resources", *parts)


def k1s1_lut() -> Path:
    return resource("luts", K1S1_LUT_NAME)


def ffmpeg_path() -> str:
    """Bundled ffmpeg first, then EXRCONV_FFMPEG, then whatever is on PATH."""
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    bundled = base_dir() / "bin" / exe
    if bundled.is_file():
        if not os.access(bundled, os.X_OK):
            try:
                bundled.chmod(0o755)
            except OSError:
                pass
        return str(bundled)
    env = os.environ.get("EXRCONV_FFMPEG")
    if env and Path(env).is_file():
        return env
    found = shutil.which("ffmpeg")
    if found:
        return found
    raise FileNotFoundError("ffmpeg was not found (it should be bundled with the app).")
