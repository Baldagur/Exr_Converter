# EXR Converter

Converts ACEScg EXRs (16-bit half) to Rec.709 or sRGB PNG / JPEG, and EXR sequences to QuickTime.
Standalone Linux app: no install, no internet needed. Double-click `ExrConverter`.

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
- Never overwrites: an existing name gets `_2`, `_3`…
- Open several EXRs at once to export them all with the same settings.

## QuickTime
- Open any frame of a sequence (`shot.1001.exr`, `shot_1001.exr`); the whole sequence is found.
- ProRes 422, ProRes 422 HQ (10-bit 4:2:2) or H.264 (8-bit 4:2:0, CRF 16).
- Full or half resolution (half sizes are rounded to even numbers).
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
- Modes: wipe (drag the line), A only, B only, difference ×10 with max/mean error in 8-bit codes.
- Clipping overlay: red = a channel hits display white, blue = hits black, magenta = negative or
  invalid values in the source EXR. Percentages are shown under the viewer controls.
- Exposure on A (±6 stops) to see how much highlight detail is left before the roll-off.
- Status bar shows ACEScg values, A values and B code values under the cursor.
- Mouse wheel zoom, drag to pan, double-click or F to fit, 1 for 1:1.

## Build
GitHub Actions builds on Rocky Linux 8 on every push; download `ExrConverter-linux-x86_64` from
the run's artifacts (or from Releases for `v*` tags). The build runs the unit tests and a full
self-test of the finished app.

Local build (machine with internet): `PYTHON=python3.11 bash packaging/build_linux.sh`

Check an installed copy: `./ExrConverter --selftest /tmp/exrconv_test`

Run from source: `pip install -r requirements.txt && python run.py`
