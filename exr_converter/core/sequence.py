"""Detect an image sequence from one of its frames (shot.1001.exr, shot_1001.exr, shot1001.exr)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

_FRAME_RE = re.compile(r"^(?P<head>.*?)(?P<frame>\d+)(?P<tail>\.[A-Za-z0-9]+)$")


@dataclass
class Sequence:
    folder: Path
    head: str
    tail: str
    padding: int
    frames: List[int]

    def path(self, frame: int) -> Path:
        return self.folder / f"{self.head}{str(frame).zfill(self.padding)}{self.tail}"

    @property
    def first(self) -> int:
        return self.frames[0]

    @property
    def last(self) -> int:
        return self.frames[-1]

    def missing(self, start: Optional[int] = None, end: Optional[int] = None) -> List[int]:
        start = self.first if start is None else start
        end = self.last if end is None else end
        have = set(self.frames)
        return [f for f in range(start, end + 1) if f not in have]

    @property
    def base_name(self) -> str:
        """Sequence name without frame number or trailing separator: 'shot.' -> 'shot'."""
        base = self.head.rstrip("._- ")
        return base or "sequence"

    @property
    def label(self) -> str:
        pad = "#" * self.padding
        return f"{self.head}{pad}{self.tail}  [{self.first}-{self.last}, {len(self.frames)} frames]"


def detect_sequence(path) -> Optional[Sequence]:
    """Return the sequence `path` belongs to, or None if it has no frame number / is alone."""
    path = Path(path)
    m = _FRAME_RE.match(path.name)
    if not m:
        return None
    head, tail, digits = m.group("head"), m.group("tail"), m.group("frame")
    pat = re.compile("^" + re.escape(head) + r"(\d+)" + re.escape(tail) + "$", re.IGNORECASE)
    frames, pads = [], set()
    for p in path.parent.iterdir():
        mm = pat.match(p.name)
        if mm and p.is_file():
            frames.append(int(mm.group(1)))
            pads.add(len(mm.group(1)))
    if len(frames) < 2:
        return None
    padding = len(digits) if digits.startswith("0") or len(pads) == 1 else min(pads)
    return Sequence(path.parent, head, tail, padding, sorted(set(frames)))
