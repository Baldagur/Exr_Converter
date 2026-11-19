"""Detect an image sequence from one of its frames (shot.1001.exr, shot_1001.exr, shot-1001.exr).

The frame number must follow '.', '_' or '-', so versioned single files (shot_v001.exr,
shot_v002.exr) are not mistaken for frames. Matching is case-sensitive (Linux filesystems are),
and every frame keeps its real file name, so mixed padding can't produce wrong paths.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

_FRAME_RE = re.compile(r"^(?P<head>.*[._-])(?P<frame>\d+)(?P<tail>\.[A-Za-z0-9]+)$")


def split_frame(name: str):
    """'shot.1001.exr' -> ('shot.', '1001', '.exr'), or None."""
    m = _FRAME_RE.match(name)
    return (m.group("head"), m.group("frame"), m.group("tail")) if m else None


@dataclass
class Sequence:
    folder: Path
    head: str
    tail: str
    padding: int
    files: Dict[int, Path] = field(default_factory=dict)

    @property
    def frames(self) -> List[int]:
        return sorted(self.files)

    def path(self, frame: int) -> Path:
        return self.files.get(frame) or self.folder / f"{self.head}{str(frame).zfill(self.padding)}{self.tail}"

    @property
    def first(self) -> int:
        return self.frames[0]

    @property
    def last(self) -> int:
        return self.frames[-1]

    def missing(self, start: Optional[int] = None, end: Optional[int] = None) -> List[int]:
        start = self.first if start is None else start
        end = self.last if end is None else end
        return [f for f in range(start, end + 1) if f not in self.files]

    @property
    def base_name(self) -> str:
        """Sequence name without frame number or trailing separator: 'shot.' -> 'shot'."""
        return self.head.rstrip("._- ") or "sequence"

    @property
    def label(self) -> str:
        pad = "#" * self.padding
        return f"{self.head}{pad}{self.tail}  [{self.first}-{self.last}, {len(self.files)} frames]"


def detect_sequence(path) -> Optional[Sequence]:
    """Return the sequence `path` belongs to, or None if it has no frame number / is alone."""
    path = Path(path)
    parts = split_frame(path.name)
    if not parts:
        return None
    head, digits, tail = parts
    pat = re.compile("^" + re.escape(head) + r"(\d+)" + re.escape(tail) + "$")
    files: Dict[int, Path] = {}
    for p in path.parent.iterdir():
        mm = pat.match(p.name)
        if mm and p.is_file():
            frame = int(mm.group(1))
            # Same frame number with different padding: prefer the padding of the opened file.
            if frame not in files or len(mm.group(1)) == len(digits):
                files[frame] = p
    if len(files) < 2:
        return None
    return Sequence(path.parent, head, tail, len(digits), files)
