# EXR Converter

Converts ACEScg EXRs (16-bit half) to Rec.709 or sRGB PNG / JPEG, and EXR sequences to QuickTime.
Standalone Linux app: no install, no internet needed. Unpack the folder anywhere and run
`ExrConverter/ExrConverter` (double-click, or from a terminal). Keep the folder together.

## Opening files
- **Open EXR(s)…**, drag and drop (files or folders), or **paste a path** in the box under the
  buttons and press Enter. Ctrl+V anywhere in the window (outside a text field) does the same.
- Pasted paths can be quoted, `file://` links, use `~` or `$VARS`, a folder (opens its sequence,
  or all its EXRs), or a sequence pattern like `shot.####.exr`, `shot.%04d.exr`, `shot.@@@@.exr`,
  with or without a Nuke frame range after it (`shot.####.exr 1001-1100`). One path per line for
  several files.

## Looks
| Look | Chain |
|---|---|
| ARRI K1S1 | ACEScg → ARRI LogC3 (EI 800) / AWG3 → K1S1 3D LUT (tetrahedral) |
| ACES 1.0 SDR | ACES 1.x output transform (RRT + ODT) from OCIO's built-in ACES 1.3 studio config |

Output **Rec.709** = gamma 2.4 (BT.1886). Output **sRGB** = sRGB curve. For K1S1 the LUT's Rec.709
result is re-encoded to sRGB (same primaries, same picture on an sRGB display).

The K1S1 LUT (`exr_converter/resources/luts/`) is the one from the Colorspace viewer v3:
LogC3 EI 800 AWG3 in → Rec.709 gamma 2.4, full range, 33 point.

## Stills
- Formats: PNG 16-bit, PNG 8-bit, JPEG (4:4:4, quality setting).
- Naming: `shot_v001.exr` → `shot_v001_rec709.png` / `shot_v001_sRGB.jpg`, next to the original
  (or another folder). Optional look in the name: `shot_v001_K1S1_rec709.png`.
  Frames keep the number last so Nuke/RV still see a sequence: `shot.1001.exr` → `shot_rec709.1001.png`.
- Never overwrites: an existing name gets `_2`, `_3`…, and files are created exclusively, so a
  file that appears at the last moment is never written over.
- Open several EXRs at once to export them all with the same settings.
- **Alpha (PNG):** tick "Write alpha" to get an RGBA PNG (8 or 16-bit, straight alpha). CG renders
  are premultiplied, so the RGB is unpremultiplied before the look (like Nuke's Unpremult →
  colour transform → Write); untick "Unpremultiply" for EXRs that are already straight. JPEG and
  QuickTime have no alpha. Multichannel/multipart renders: the beauty's A is used.

## QuickTime
- Open any frame of a sequence (`shot.1001.exr`, `shot_1001.exr`); the whole sequence is found.
  The frame number must follow `.`, `_` or `-`, so `shot_v001.exr` / `shot_v002.exr` are versions,
  not frames. Matching is case-sensitive.
- ProRes 422, ProRes 422 HQ (10-bit 4:2:2) or H.264 (8-bit 4:2:0, CRF 16).
- Full or half resolution (half sizes are rounded to even numbers). H.264 needs even sizes, so an
  odd full-res frame loses one pixel row/column by cropping (never resampled).
- Encodes to a hidden `.partial.mov` and moves it into place at the end; a failed or cancelled
  export only ever removes its own partial file.
- In/out frames, frame rate (read from the EXR header when present), timecode from the header,
  otherwise from the first frame number.
- Missing frames: it stops and lists them instead of making a movie with a gap.
- Tagged Rec.709 (or sRGB transfer) so Flame, Resolve and RV read it right. QuickTime Player on
  macOS shows Rec.709 files slightly lighter; that's the player.
- Name: `shot_rec709.mov`, `shot_rec709_half.mov`.

## Viewer
- **A** = the original EXR shown live through a look (pick the export look or the other one).
- **B** = the result: a preview of what will be written, or the actual file read back after export
  (or load any PNG/JPEG with "Load a result file as B").
- Modes: wipe (drag the line), A only, B only, difference ×10 with max/mean error in 8-bit codes,
  alpha. With PNG alpha on, A and B are shown over black so they still compare 1:1.
- Clipping overlay: red = a channel hits display white, blue = hits black, magenta = negative or
  invalid values in the source EXR. Percentages are shown under the viewer controls.
- Exposure on A (±6 stops) to see how much highlight detail is left before the roll-off.
- Status bar shows ACEScg values, A values and B code values under the cursor.
- Mouse wheel zoom, drag to pan, double-click or F to fit, 1 for 1:1.

## Build
GitHub Actions builds on Rocky Linux 8 on every push; download `ExrConverter-linux-x86_64` from
the run's artifacts (or from Releases for `v*` tags). The build runs the unit tests and a full
self-test of the finished app. `ExrConverter/BUILD_INFO.txt` records the exact ffmpeg build
(URL + sha256) and Python package versions; set `FFMPEG_URL` / `FFMPEG_SHA256` to pin ffmpeg.

Local build (machine with internet): `PYTHON=python3.11 bash packaging/build_linux.sh`

Check an installed copy: `./ExrConverter/ExrConverter --selftest /tmp/exrconv_test`

Run from source (Rocky's system Python is too old for PySide6):
`python3.11 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt && python run.py`
