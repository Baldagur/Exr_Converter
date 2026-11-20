from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from exr_converter.core import compare
from exr_converter.core.color import ColorEngine, Lut3D, bt1886_to_srgb, reference_k1s1
from exr_converter.core.movie import build_command, fps_rational, frame_to_timecode, output_size
from exr_converter.core.output import output_path, quantize, read_image, write_png
from exr_converter.core.resources import k1s1_lut
from exr_converter.core.sequence import detect_sequence


def test_output_name_and_no_overwrite(tmp_path):
    src = tmp_path / "shot_v001.exr"
    src.write_bytes(b"x")
    p = output_path(src, "rec709", ".png")
    assert p.name == "shot_v001_rec709.png"
    p.write_bytes(b"x")
    assert output_path(src, "rec709", ".png").name == "shot_v001_rec709_2.png"
    assert output_path(src, "sRGB", ".jpg").name == "shot_v001_sRGB.jpg"


def test_sequence_detection(tmp_path):
    for f in (1001, 1002, 1004):
        (tmp_path / f"plate_v002.{f:04d}.exr").write_bytes(b"x")
    (tmp_path / "other.1001.exr").write_bytes(b"x")
    seq = detect_sequence(tmp_path / "plate_v002.1002.exr")
    assert seq.frames == [1001, 1002, 1004]
    assert seq.missing() == [1003]
    assert seq.base_name == "plate_v002"
    assert seq.path(1003).name == "plate_v002.1003.exr"
    assert detect_sequence(tmp_path / "single.exr") is None


def test_versions_and_case_are_not_sequences(tmp_path):
    for n in ("shot_v001.exr", "shot_v002.exr", "Plate.1001.exr", "plate.1002.exr", "plate.1003.EXR"):
        (tmp_path / n).write_bytes(b"x")
    assert detect_sequence(tmp_path / "shot_v001.exr") is None
    assert detect_sequence(tmp_path / "plate.1002.exr") is None  # only one frame with that exact case


def test_still_name_keeps_frame_last(tmp_path):
    src = tmp_path / "shot.1001.exr"
    src.write_bytes(b"x")
    assert output_path(src, "rec709", ".png").name == "shot_rec709.1001.png"
    (tmp_path / "shot_rec709.1001.png").write_bytes(b"x")
    assert output_path(src, "rec709", ".png").name == "shot_rec709_2.1001.png"


def test_writers_never_overwrite(tmp_path):
    from exr_converter.core.output import write_still
    p = tmp_path / "a.png"
    p.write_bytes(b"keep")
    for fmt in ("png16", "png8", "jpeg"):
        with pytest.raises(FileExistsError):
            write_still(p, np.zeros((2, 2, 3), np.float32), fmt, "rec709")
    assert p.read_bytes() == b"keep"


@pytest.mark.parametrize("bits", [8, 16])
def test_png_roundtrip(tmp_path, bits):
    rng = np.random.default_rng(1)
    disp = rng.random((7, 9, 3)).astype(np.float32)
    img = quantize(disp, bits)
    write_png(tmp_path / "a.png", img)
    back, b = read_image(tmp_path / "a.png")
    assert b == bits
    assert np.array_equal(np.rint(back * ((1 << bits) - 1)).astype(img.dtype), img)


def test_lut_exact_on_lattice_and_monotonic_grey():
    lut = Lut3D.load_cube(k1s1_lut())
    idx = np.array([[0, 0, 0], [5, 9, 30], [32, 32, 32]])
    vals = lut.apply((idx / 32.0).astype(np.float32))
    assert np.allclose(vals, lut.table[idx[:, 2], idx[:, 1], idx[:, 0]])
    grey = reference_k1s1(np.geomspace(1e-4, 50, 64, dtype=np.float32)[:, None].repeat(3, 1))
    assert np.all(np.diff(grey[:, 1]) >= -1e-6)
    assert 0.37 < reference_k1s1(np.full((1, 3), 0.18, np.float32))[0, 0] < 0.43


def test_srgb_reencode():
    v = np.array([0.0, 1.0, 0.5], np.float32)
    s = bt1886_to_srgb(v)
    assert s[0] == 0 and abs(s[1] - 1) < 1e-6 and s[2] < 0.5


def test_fps_and_timecode():
    assert fps_rational("23.976") == Fraction(24000, 1001)
    assert fps_rational("25") == 25
    assert fps_rational("24000/1001") == Fraction(24000, 1001)
    assert fps_rational("23.98") == Fraction(24000, 1001)
    assert frame_to_timecode(1001, Fraction(24)) == "00:00:41:17"


def test_sizes_and_command(tmp_path):
    assert output_size(1919, 1081, False, "prores422") == (1919, 1081)
    assert output_size(1919, 1081, False, "h264") == (1918, 1080)
    assert output_size(3840, 2160, True, "prores422hq") == (1920, 1080)
    cmd = build_command("ffmpeg", 3840, 2160, Fraction(24), "prores422hq", True, "rec709", tmp_path / "o.mov",
                        "01:00:00:00")
    s = " ".join(cmd)
    assert "prores_ks" in s and "-profile:v 3" in s and "scale=1920:1080" in s and "-timecode 01:00:00:00" in s
    odd = build_command("ffmpeg", 1919, 1081, Fraction(24), "h264", False, "rec709", tmp_path / "o.mov")
    assert "crop=1918:1080:0:0" in " ".join(odd)  # parity fixed by cropping, not resampling


def test_compare_helpers():
    d = np.zeros((2, 2, 3), np.float32)
    d[0, 0] = 1.0
    hi, lo = compare.clip_masks(d)
    assert hi[0, 0] and not hi[1, 1] and lo[1, 1]
    img, mx, mean = compare.difference(d, d)
    assert mx == 0 and img.max() == 0


def test_ocio_matches_reference():
    pytest.importorskip("PyOpenColorIO")
    eng = ColorEngine()
    rgb = np.random.default_rng(3).random((32, 32, 3)).astype(np.float32) * 4
    diff = np.abs(eng.convert(rgb, "k1s1", "rec709") - reference_k1s1(rgb)).max()
    assert diff < 0.004, diff


@pytest.mark.parametrize("output", ["rec709", "srgb"])
def test_ocio_aces_grey(output):
    pytest.importorskip("PyOpenColorIO")
    eng = ColorEngine()
    g = eng.convert(np.full((1, 1, 3), 0.18, np.float32), "aces", output)[0, 0, 0]
    assert 0.3 < g < 0.5, g


def test_choose_beauty_ignores_aovs():
    from exr_converter.core.exr_io import choose_beauty
    assert choose_beauty([("", ["A", "B", "G", "R", "Z", "diffuse.R", "diffuse.G", "diffuse.B"])])[1] == ""
    pi, layer, comps = choose_beauty([("", ["ViewLayer.Combined.R", "ViewLayer.Combined.G", "ViewLayer.Combined.B",
                                            "ViewLayer.DiffCol.R", "ViewLayer.DiffCol.G", "ViewLayer.DiffCol.B"])])
    assert layer == "ViewLayer.Combined" and comps["R"] == "ViewLayer.Combined.R"
    assert choose_beauty([("diffuse", ["R", "G", "B"]), ("beauty", ["R", "G", "B", "A"])])[0] == 1
    with pytest.raises(ValueError, match="No RGB"):
        choose_beauty([("", ["Z", "N.X"])])


def test_reads_cg_multichannel_files(tmp_path):
    pytest.importorskip("OpenEXR")
    from exr_converter.core import exr_io
    files = exr_io.write_cg_test_files(tmp_path)
    for label, (path, expected) in files.items():
        rgb, info = exr_io.read_exr(path)
        assert info.reader == "OpenEXR", (label, info.notes)
        assert np.allclose(rgb, expected, atol=1e-3), label
        # the full reader must work too, not only the fast single-part path
        rgb2 = exr_io._read_new(path)[0]
        assert np.allclose(rgb2, expected, atol=1e-3), label
