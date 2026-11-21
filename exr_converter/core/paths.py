"""Turn pasted text (or dropped paths) into EXR files to open.

Understands what gets copied from Nuke, a terminal or a file browser:
  /jobs/shot/render/shot.1001.exr          a file
  "/jobs/shot/render/shot.1001.exr"        quoted (also 'single quotes')
  file:///jobs/shot/render/shot.1001.exr   a URL from a file browser (%20 etc. decoded)
  ~/renders/shot.1001.exr, $SHOW/...       home and environment variables
  /jobs/shot/render/shot.####.exr          a sequence pattern: #### / %04d / @@@@ (Nuke, RV, Houdini)
  /jobs/shot/render/shot.####.exr 1001-1100   a Nuke-style frame range after the pattern
  /jobs/shot/render/                       a folder: its sequence, or all its EXRs
Several paths can be pasted at once, one per line.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Tuple
from urllib.parse import unquote, urlparse

from .sequence import detect_sequence

_RANGE_RE = re.compile(r"\s+-?\d+\s*-\s*-?\d+\s*$")          # "  1001-1100" at the end
_PATTERN_RE = re.compile(r"(#+|@+|%0?(\d*)d|\$F(\d*))")      # frame tokens
_WINDOWS_RE = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\)")


def _clean(line: str) -> str:
    s = line.strip().strip("​").strip()
    s = _RANGE_RE.sub("", s).strip()  # '"shot.####.exr" 1001-1100': drop the range before the quotes
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    if s.lower().startswith("file:"):
        u = urlparse(s)
        s = unquote(u.path)
    s = _RANGE_RE.sub("", s)
    return os.path.expanduser(os.path.expandvars(s))


def _first_of_pattern(path: Path) -> Tuple[List[Path], str]:
    """shot.####.exr -> the first existing frame of that sequence."""
    name = path.name
    m = _PATTERN_RE.search(name)
    head, tail = name[: m.start()], name[m.end():]
    token = m.group(0)
    if token.startswith("%"):
        pad = int(m.group(2) or 0)
    elif token.startswith("$F"):
        pad = int(m.group(3) or 0)
    else:
        pad = len(token)
    digits = rf"\d{{{pad},}}" if pad > 1 else r"\d+"
    rx = re.compile("^" + re.escape(head) + "(" + digits + ")" + re.escape(tail) + "$")
    folder = path.parent
    if not folder.is_dir():
        return [], f"Folder not found: {folder}"
    frames = sorted((int(mm.group(1)), p) for p in folder.iterdir()
                    if p.is_file() and (mm := rx.match(p.name)))
    if not frames:
        return [], f"No frames match {name} in {folder}"
    return [frames[0][1]], ""


def _from_folder(folder: Path) -> Tuple[List[Path], str]:
    exrs = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".exr")
    if not exrs:
        return [], f"No EXR files in {folder}"
    seq = detect_sequence(exrs[0])
    if seq is not None and len(seq.files) == len(exrs):
        return [seq.path(seq.first)], ""  # one sequence: open its first frame
    return exrs, ""


def resolve_pasted(text: str) -> Tuple[List[Path], List[str]]:
    """Return (EXR files to open, problems to show)."""
    files: List[Path] = []
    problems: List[str] = []
    for raw in re.split(r"[\r\n]+", text or ""):
        if not raw.strip():
            continue
        s = _clean(raw)
        if _WINDOWS_RE.match(s):
            problems.append(f"Windows path, not valid on Linux: {s}")
            continue
        p = Path(s)
        if _PATTERN_RE.search(p.name) and not p.exists():
            found, err = _first_of_pattern(p)
        elif p.is_dir():
            found, err = _from_folder(p)
        elif p.is_file():
            if p.suffix.lower() == ".exr":
                found, err = [p], ""
            else:
                found, err = [], f"Not an EXR: {p.name}"
        else:
            found, err = [], f"Not found: {s}"
        files.extend(f for f in found if f not in files)
        if err:
            problems.append(err)
    return files, problems
