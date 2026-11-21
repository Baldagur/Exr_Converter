"""`ExrConverter --selftest [folder]`: checks a build end to end without user input.

Writes a small synthetic ACEScg EXR sequence, converts it with every look/output/format,
reads the files back, encodes every QuickTime variant and opens the GUI off-screen.
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np


def _frames(folder: Path, n: int = 4, w: int = 321, h: int = 181) -> list:
    from .core.exr_io import write_exr

    paths = []
    for i in range(n):
        img = np.zeros((h, w, 3), np.float32)
        img[:] = np.linspace(0, 8, w, dtype=np.float32)[None, :, None] * (1 + 0.05 * i)
        img[:40] = 0.18
        img[40:60, :] = [0.6, 0.05, 0.02]   # saturated red
        img[60:70, :10] = -0.01             # small negative values
        p = folder / f"selftest.{1001 + i:04d}.exr"
        write_exr(p, img)
        paths.append(p)
    return paths


def run(folder: str = "") -> int:
    import faulthandler

    faulthandler.enable()
    sys.stdout.reconfigure(line_buffering=True)
    from .core.color import LOOKS, OUTPUTS, ColorEngine, reference_k1s1
    from .core.exr_io import read_exr
    from .core.movie import CODECS, MovieJob, encode_movie, movie_path, probe
    from .core.output import FORMATS, output_path, read_image, write_still
    from .core.resources import ffmpeg_path, k1s1_lut
    from .core.sequence import detect_sequence

    out = Path(folder) if folder else Path(tempfile.mkdtemp(prefix="exrconv_selftest_"))
    out.mkdir(parents=True, exist_ok=True)
    failures = []

    def check(cond: bool, msg: str):
        print(("  ok   " if cond else "  FAIL ") + msg)
        if not cond:
            failures.append(msg)

    print(f"Self-test in {out}")
    print(f"  ffmpeg: {ffmpeg_path()}")
    check(k1s1_lut().is_file(), f"K1S1 LUT bundled ({k1s1_lut().name})")
    engine = ColorEngine()
    print(f"  colour: {engine.describe()}")
    check(engine.has_aces, "OCIO with an ACES 1.3 config is available")

    frames = _frames(out)
    rgb, info = read_exr(frames[0])
    check(info.width == 321 and info.height == 181, f"EXR read back at 321x181 via {info.reader}")
    check(abs(float(rgb[5, 5, 0]) - 0.18) < 1e-3, "EXR grey value preserved")

    k = engine.convert(rgb, "k1s1", "rec709")
    ref = reference_k1s1(rgb)
    err = float(np.abs(k - ref)[:60].max())
    check(err < 0.004, f"K1S1 via OCIO matches numpy reference (max diff {err:.5f})")
    check(0.37 < float(k[5, 5, 0]) < 0.43, f"K1S1 Rec.709 grey 0.18 -> {float(k[5, 5, 0]):.4f}")

    for look in LOOKS:
        for output in OUTPUTS:
            disp = engine.convert(rgb, look, output)
            g = float(disp[5, 5, 1])
            check(0.30 < g < 0.50, f"{look}/{output}: grey -> {g:.4f}")
            check(float(disp.max()) <= 1.0001 and float(disp.min()) >= -0.0001,
                  f"{look}/{output}: output inside 0..1 (min {disp.min():.4f}, max {disp.max():.4f})")
            for fmt, (_label, ext) in FORMATS.items():
                p = output_path(frames[0], f"{look}_{output}", ext)
                write_still(p, disp, fmt, output)
                back, bits = read_image(p)
                tol = 0.02 if fmt == "jpeg" else (0.6 / 255 if fmt == "png8" else 1e-4)
                diff = float(np.abs(back - np.clip(disp, 0, 1))[:40].max())
                check(diff < tol, f"  {p.name}: {bits}-bit, read-back diff {diff:.5f}")
            try:
                write_still(p, disp, "png8", output)
                check(False, "refuses to overwrite")
            except FileExistsError:
                pass

    try:
        from .core.exr_io import write_cg_test_files

        for label, (path, expected) in write_cg_test_files(out).items():
            got, cg_info = read_exr(path)
            diff = float(np.abs(got - expected).max())
            check(diff < 1e-3 and cg_info.reader == "OpenEXR", f"CG EXR {label}: {cg_info.channels}")
    except ImportError:
        print("  (CG multichannel test skipped: no OpenEXR bindings)")

    # Alpha: premultiplied CG-style EXR -> PNG with straight alpha
    from .core.color import unpremultiply
    from .core.exr_io import read_exr_rgba, write_exr
    from .core.output import read_image_rgba

    h, w = 64, 96
    a = np.clip(np.linspace(-0.2, 1.2, w, dtype=np.float32), 0, 1)[None, :].repeat(h, 0)
    straight = np.zeros((h, w, 3), np.float32)
    straight[:] = [0.6, 0.18, 0.05]
    alpha_exr = out / "alpha_test.exr"
    write_exr(alpha_exr, straight * a[..., None], a)
    rgb_a, al, _ = read_exr_rgba(alpha_exr)
    check(al is not None and float(np.abs(al - a).max()) < 1e-3, "EXR alpha read")
    for look in LOOKS:
        disp = engine.convert(unpremultiply(rgb_a, al), look, "rec709")
        pth = out / f"alpha_{look}.png"
        write_still(pth, disp, "png16", "rec709", alpha=al)
        back, back_a, bits = read_image_rgba(pth)
        ref = engine.convert(straight, look, "rec709")  # colour must not darken toward the edges
        visible = a > 0.05
        err_rgb = float(np.abs(back - ref)[visible].max())
        err_a = float(np.abs(back_a - a).max())
        check(back_a is not None and err_a < 1e-3 and err_rgb < 0.004,
              f"PNG alpha {look}: straight colour err {err_rgb:.5f}, alpha err {err_a:.5f}")

    seq = detect_sequence(frames[1])
    check(seq is not None and seq.frames == [1001, 1002, 1003, 1004], "sequence detected")
    for codec in CODECS:
        for half in (False, True):
            mp = movie_path(seq, "rec709", half, "k1s1", True)
            encode_movie(MovieJob(seq, 1001, 1004, "k1s1", "rec709", codec, half, "23.976", mp), engine)
            desc = [ln.strip() for ln in probe(mp).splitlines() if "Video:" in ln]
            size = ("160x90" if half else ("320x180" if codec == "h264" else "321x181"))
            check(bool(desc) and size in desc[0], f"{mp.name}: {desc[0] if desc else 'no video stream'}")

    import os

    if os.environ.get("EXRCONV_SELFTEST_NOGUI"):
        print("  (GUI test skipped)")
    else:
        _gui_test(out, frames, engine, check, alpha_exr)

    print("\nSELFTEST " + ("PASSED" if not failures else f"FAILED ({len(failures)})"))
    return 0 if not failures else 1


def _gui_test(out, frames, engine, check, alpha_exr):
    from .core.output import read_image_rgba

    try:
        from PySide6.QtWidgets import QApplication

        from .ui.main_window import MainWindow

        app = QApplication.instance() or QApplication([])
        win = MainWindow(engine)
        win.show()
        win.open_files([frames[0]])
        app.processEvents()
        for i in range(win.cmb_mode.count()):
            win.cmb_mode.setCurrentIndex(i)
            app.processEvents()
        win.chk_clip.setChecked(True)
        win.sld_exp.setValue(20)
        win.refresh_a()
        win.cmb_look.setCurrentIndex(win.cmb_look.findData("aces"))
        win.cmb_out.setCurrentIndex(win.cmb_out.findData("srgb"))
        win.cmb_fmt.setCurrentIndex(win.cmb_fmt.findData("png16"))
        app.processEvents()
        before = set(out.iterdir())
        win.export_stills()
        new = [p.name for p in set(out.iterdir()) - before]
        check(any("_sRGB." in n and n.endswith(".png") for n in new), f"GUI still export: {new}")
        check(win.seq is not None and win.btn_movie.isEnabled(), "GUI sees the sequence")
        win.ed_path.setText(f'"{frames[0].parent}/selftest.####.exr" 1001-1004')
        win.open_typed_path()
        app.processEvents()
        check(win.files == [frames[0]] and win.seq is not None and win.ed_path.text() == "",
              f"GUI pasted path opens the sequence: {[p.name for p in win.files]}")
        win.open_files([alpha_exr])
        app.processEvents()
        win.chk_alpha.setChecked(True)
        win.cmb_mode.setCurrentIndex(win.cmb_mode.findData("alpha"))
        app.processEvents()
        before = set(out.iterdir())
        win.export_stills()
        new = [p for p in set(out.iterdir()) - before if p.suffix == ".png"]
        has_alpha = bool(new) and read_image_rgba(new[0])[1] is not None
        check(has_alpha, f"GUI PNG export with alpha: {[p.name for p in new]}")
        win.close()
        win.deleteLater()
        app.processEvents()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        check(False, "GUI off-screen test")


if __name__ == "__main__":
    sys.exit(run(sys.argv[1] if len(sys.argv) > 1 else ""))
