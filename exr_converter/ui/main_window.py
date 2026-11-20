"""Main window: controls on the right, compare viewer on the left."""

from __future__ import annotations

import threading
import traceback
from pathlib import Path
from typing import List, Optional

import numpy as np
from PySide6.QtCore import QSettings, QThread, QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QMainWindow, QMessageBox, QProgressBar,
    QProgressDialog, QPushButton, QScrollArea, QSlider, QSpinBox, QSplitter, QVBoxLayout, QWidget,
)

from .. import APP_NAME, __version__
from ..core import compare
from ..core.color import (
    LOOK_SUFFIX, LOOKS, OUTPUT_SUFFIX, OUTPUTS, ColorEngine, sanitize, source_problem_mask, unpremultiply,
)
from ..core.exr_io import ExrInfo, read_exr_rgba
from ..core.movie import CODECS, FPS_CHOICES, MovieCancelled, MovieJob, encode_movie, fps_label, fps_rational, movie_path
from ..core.output import FORMATS, output_path, read_image_rgba, write_still
from ..core.sequence import Sequence, detect_sequence
from .viewer import ImageViewer

VIEW_MODES = {
    "wipe": "Wipe  A | B",
    "a": "A only (original, live)",
    "b": "B only (result)",
    "diff": "Difference ×10",
    "alpha": "Alpha",
}


class MovieThread(QThread):
    progress = Signal(int, int)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, job: MovieJob, engine: ColorEngine):
        super().__init__()
        self.job, self.engine = job, engine
        self.cancel_event = threading.Event()

    def run(self):
        try:
            out = encode_movie(self.job, self.engine, lambda i, n: self.progress.emit(i, n), self.cancel_event)
            self.finished_ok.emit(str(out))
        except MovieCancelled:
            self.failed.emit("Cancelled. The partial file was removed.")
        except Exception as err:  # noqa: BLE001
            self.failed.emit(str(err))


class MainWindow(QMainWindow):
    def __init__(self, engine: Optional[ColorEngine] = None):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {__version__}")
        self.resize(1500, 900)
        self.setAcceptDrops(True)
        self.settings = QSettings("Baldagur", "ExrConverter")

        self.engine = engine or ColorEngine()
        self.files: List[Path] = []
        self.src: Optional[np.ndarray] = None
        self.src_bad: Optional[np.ndarray] = None
        self.src_alpha: Optional[np.ndarray] = None
        self.b_alpha: Optional[np.ndarray] = None  # alpha as stored in the result (PNG), or None
        self.b_raw: Optional[np.ndarray] = None    # result RGB as stored (b_img is shown over black)
        self.info: Optional[ExrInfo] = None
        self.seq: Optional[Sequence] = None
        self.a_disp: Optional[np.ndarray] = None
        self.b_img: Optional[np.ndarray] = None  # float 0..1 as stored in the file
        self.b_bits = 8
        self.b_label = ""
        self.b_from_file = False
        self.movie_thread: Optional[MovieThread] = None

        self._refresh_timer = QTimer(self, singleShot=True, interval=120)
        self._refresh_timer.timeout.connect(self.refresh_a)

        self._build_ui()
        self._restore_settings()
        self._update_enabled()

    # ================================================================== UI
    def _build_ui(self):
        self.viewer = ImageViewer()
        self.viewer.hovered.connect(self._on_hover)

        panel = QWidget()
        col = QVBoxLayout(panel)
        col.setContentsMargins(8, 8, 8, 8)

        # --- source
        g_src = QGroupBox("Source (ACEScg EXR)")
        ls = QVBoxLayout(g_src)
        row = QHBoxLayout()
        self.btn_open = QPushButton("Open EXR(s)…")
        self.btn_open.clicked.connect(self.open_dialog)
        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self.clear_files)
        row.addWidget(self.btn_open)
        row.addWidget(self.btn_clear)
        ls.addLayout(row)
        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(90)
        self.file_list.currentRowChanged.connect(self._on_file_selected)
        ls.addWidget(self.file_list)
        self.lbl_info = QLabel("No file loaded.")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        ls.addWidget(self.lbl_info)
        col.addWidget(g_src)

        # --- conversion
        g_conv = QGroupBox("Conversion")
        fc = QFormLayout(g_conv)
        self.cmb_look = QComboBox()
        for k, v in LOOKS.items():
            self.cmb_look.addItem(v, k)
        self.cmb_out = QComboBox()
        for k, v in OUTPUTS.items():
            self.cmb_out.addItem(f"{v}   →  _{OUTPUT_SUFFIX[k]}", k)
        self.chk_lookname = QCheckBox("Add look to file name (e.g. _K1S1_rec709)")
        fc.addRow("Look", self.cmb_look)
        fc.addRow("Output", self.cmb_out)
        fc.addRow("", self.chk_lookname)
        self.chk_outdir = QCheckBox("Save to another folder")
        self.ed_outdir = QLineEdit()
        self.ed_outdir.setPlaceholderText("Next to the original")
        self.btn_outdir = QPushButton("…")
        self.btn_outdir.setFixedWidth(30)
        self.btn_outdir.clicked.connect(self._choose_outdir)
        r2 = QHBoxLayout()
        r2.addWidget(self.ed_outdir)
        r2.addWidget(self.btn_outdir)
        fc.addRow(self.chk_outdir)
        fc.addRow(r2)
        lbl_engine = QLabel(self.engine.describe())
        lbl_engine.setStyleSheet("color: gray; font-size: 10px;")
        lbl_engine.setWordWrap(True)
        fc.addRow(lbl_engine)
        col.addWidget(g_conv)

        # --- stills
        g_still = QGroupBox("Still export")
        fs = QFormLayout(g_still)
        self.cmb_fmt = QComboBox()
        for k, (label, _ext) in FORMATS.items():
            self.cmb_fmt.addItem(label, k)
        self.spn_quality = QSpinBox()
        self.spn_quality.setRange(50, 100)
        self.spn_quality.setValue(95)
        self.btn_still = QPushButton("Export still(s)")
        self.btn_still.clicked.connect(self.export_stills)
        fs.addRow("Format", self.cmb_fmt)
        fs.addRow("JPEG quality", self.spn_quality)
        self.chk_alpha = QCheckBox("Write alpha (PNG only)")
        self.chk_unpremult = QCheckBox("Unpremultiply before the look (CG renders)")
        self.chk_unpremult.setChecked(True)
        self.chk_unpremult.setToolTip("EXR renders are premultiplied. PNG stores straight alpha, so the RGB is\n"
                                      "divided by alpha before the colour transform, like Nuke's Unpremult.")
        fs.addRow(self.chk_alpha)
        fs.addRow(self.chk_unpremult)
        fs.addRow(self.btn_still)
        col.addWidget(g_still)

        # --- movie
        g_mov = QGroupBox("QuickTime from sequence")
        fm = QFormLayout(g_mov)
        self.lbl_seq = QLabel("Open one frame of a sequence.")
        self.lbl_seq.setWordWrap(True)
        self.cmb_codec = QComboBox()
        for k, v in CODECS.items():
            self.cmb_codec.addItem(v, k)
        self.cmb_size = QComboBox()
        self.cmb_size.addItem("Full resolution", False)
        self.cmb_size.addItem("Half resolution", True)
        self.cmb_fps = QComboBox()
        self.cmb_fps.setEditable(True)
        self.cmb_fps.addItems(FPS_CHOICES)
        self.cmb_fps.setCurrentText("24")
        self.spn_in = QSpinBox()
        self.spn_out = QSpinBox()
        for s in (self.spn_in, self.spn_out):
            s.setRange(-10_000_000, 10_000_000)
        rio = QHBoxLayout()
        rio.addWidget(QLabel("In"))
        rio.addWidget(self.spn_in)
        rio.addWidget(QLabel("Out"))
        rio.addWidget(self.spn_out)
        self.btn_movie = QPushButton("Export QuickTime")
        self.btn_movie.clicked.connect(self.export_movie)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.cancel_movie)
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        rb = QHBoxLayout()
        rb.addWidget(self.btn_movie)
        rb.addWidget(self.btn_cancel)
        fm.addRow(self.lbl_seq)
        fm.addRow("Codec", self.cmb_codec)
        fm.addRow("Size", self.cmb_size)
        fm.addRow("Frame rate", self.cmb_fps)
        fm.addRow("Range", rio)
        fm.addRow(rb)
        fm.addRow(self.progress)
        col.addWidget(g_mov)

        # --- viewer
        g_view = QGroupBox("Viewer")
        fv = QFormLayout(g_view)
        self.cmb_mode = QComboBox()
        for k, v in VIEW_MODES.items():
            self.cmb_mode.addItem(v, k)
        self.cmb_alook = QComboBox()
        self.cmb_alook.addItem("Same as export", "")
        for k, v in LOOKS.items():
            self.cmb_alook.addItem(v, k)
        self.sld_exp = QSlider(Qt.Horizontal)
        self.sld_exp.setRange(-60, 60)
        self.lbl_exp = QLabel("0.0")
        self.lbl_exp.setFixedWidth(40)
        rex = QHBoxLayout()
        rex.addWidget(self.sld_exp)
        rex.addWidget(self.lbl_exp)
        btn_exp0 = QPushButton("0")
        btn_exp0.setFixedWidth(26)
        btn_exp0.clicked.connect(lambda: self.sld_exp.setValue(0))
        rex.addWidget(btn_exp0)
        self.chk_clip = QCheckBox("Show clipping (red = white, blue = black, magenta = bad source)")
        self.chk_clip.setStyleSheet("QCheckBox { font-size: 11px; }")
        self.btn_loadb = QPushButton("Load a result file as B…")
        self.btn_loadb.clicked.connect(self.load_b_dialog)
        rz = QHBoxLayout()
        b_fit = QPushButton("Fit (F)")
        b_fit.clicked.connect(self.viewer.fit)
        b_11 = QPushButton("1:1 (1)")
        b_11.clicked.connect(self.viewer.one_to_one)
        rz.addWidget(b_fit)
        rz.addWidget(b_11)
        fv.addRow("Mode", self.cmb_mode)
        fv.addRow("A look", self.cmb_alook)
        fv.addRow("A exposure", rex)
        fv.addRow(self.chk_clip)
        fv.addRow(self.btn_loadb)
        fv.addRow(rz)
        self.lbl_stats = QLabel("")
        self.lbl_stats.setWordWrap(True)
        self.lbl_stats.setStyleSheet("font-size: 11px;")
        fv.addRow(self.lbl_stats)
        col.addWidget(g_view)
        col.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(380)

        split = QSplitter()
        split.addWidget(self.viewer)
        split.addWidget(scroll)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)
        split.setSizes([1100, 400])
        self.setCentralWidget(split)
        self.statusBar().showMessage("Ready")

        # signals
        for w in (self.cmb_look, self.cmb_out, self.cmb_fmt):
            w.currentIndexChanged.connect(self._on_export_settings_changed)
        self.spn_quality.valueChanged.connect(self._on_export_settings_changed)
        self.chk_alpha.toggled.connect(self._on_alpha_settings_changed)
        self.chk_unpremult.toggled.connect(self._on_alpha_settings_changed)
        self.cmb_alook.currentIndexChanged.connect(self.refresh_a)
        self.cmb_mode.currentIndexChanged.connect(self.compose)
        self.chk_clip.toggled.connect(self.compose)
        self.sld_exp.valueChanged.connect(self._on_exposure)
        self.chk_outdir.toggled.connect(self._update_enabled)

        if not self.engine.has_aces:
            idx = self.cmb_look.findData("aces")
            self.cmb_look.model().item(idx).setEnabled(False)
            self.cmb_alook.model().item(self.cmb_alook.findData("aces")).setEnabled(False)

    # ================================================================== settings
    def _restore_settings(self):
        s = self.settings
        for combo, key in ((self.cmb_look, "look"), (self.cmb_out, "output"), (self.cmb_fmt, "format"),
                           (self.cmb_codec, "codec")):
            i = combo.findData(s.value(key, combo.currentData()))
            if i >= 0:
                combo.setCurrentIndex(i)
        if not self.engine.has_aces:  # never restore a disabled choice
            for combo in (self.cmb_look, self.cmb_alook):
                if combo.currentData() == "aces":
                    combo.setCurrentIndex(combo.findData("k1s1"))
        self.chk_lookname.setChecked(s.value("lookname", False, type=bool))
        self.chk_alpha.setChecked(s.value("alpha", False, type=bool))
        self.chk_unpremult.setChecked(s.value("unpremult", True, type=bool))
        self.spn_quality.setValue(int(s.value("quality", 95)))

    def _save_settings(self):
        s = self.settings
        s.setValue("look", self.cmb_look.currentData())
        s.setValue("output", self.cmb_out.currentData())
        s.setValue("format", self.cmb_fmt.currentData())
        s.setValue("codec", self.cmb_codec.currentData())
        s.setValue("lookname", self.chk_lookname.isChecked())
        s.setValue("alpha", self.chk_alpha.isChecked())
        s.setValue("unpremult", self.chk_unpremult.isChecked())
        s.setValue("quality", self.spn_quality.value())

    def closeEvent(self, e):  # noqa: N802
        if self.movie_thread and self.movie_thread.isRunning():
            if QMessageBox.question(self, APP_NAME, "A QuickTime export is running. Cancel it and quit?") \
                    != QMessageBox.Yes:
                e.ignore()
                return
            self.movie_thread.cancel_event.set()
            self.statusBar().showMessage("Cancelling the QuickTime export…")
            while not self.movie_thread.wait(200):  # never destroy a running thread
                QApplication.processEvents()
        self._save_settings()
        super().closeEvent(e)

    # ================================================================== helpers
    @property
    def look(self) -> str:
        return self.cmb_look.currentData()

    @property
    def output(self) -> str:
        return self.cmb_out.currentData()

    @property
    def fmt(self) -> str:
        return self.cmb_fmt.currentData()

    @property
    def a_look(self) -> str:
        return self.cmb_alook.currentData() or self.look

    @property
    def alpha_on(self) -> bool:
        """Alpha is written (and shown) only for PNG, when asked for and the EXR has one."""
        return self.fmt in ("png16", "png8") and self.chk_alpha.isChecked() and self.src_alpha is not None

    def export_display(self, rgb: np.ndarray, alpha: Optional[np.ndarray], look: str,
                       gain: float = 1.0) -> np.ndarray:
        """Display RGB for export. With alpha on: unpremultiplied first (straight alpha)."""
        if alpha is not None and self.chk_unpremult.isChecked():
            rgb = unpremultiply(rgb, alpha)
        if gain != 1.0:
            rgb = rgb * gain
        return self.engine.convert(rgb, look, self.output)

    def suffix(self) -> str:
        s = OUTPUT_SUFFIX[self.output]
        return f"{LOOK_SUFFIX[self.look]}_{s}" if self.chk_lookname.isChecked() else s

    def out_dir(self) -> Optional[Path]:
        if self.chk_outdir.isChecked() and self.ed_outdir.text().strip():
            return Path(self.ed_outdir.text().strip())
        return None

    def _update_enabled(self):
        has = self.src is not None
        busy = bool(self.movie_thread and self.movie_thread.isRunning())
        self.btn_still.setEnabled(has and not busy)
        self.btn_movie.setEnabled(self.seq is not None and not busy)
        self.btn_cancel.setEnabled(busy)
        self.spn_quality.setEnabled(self.fmt == "jpeg")
        is_png = self.fmt in ("png16", "png8")
        self.chk_alpha.setEnabled(is_png)
        self.chk_unpremult.setEnabled(is_png and self.chk_alpha.isChecked())
        self.ed_outdir.setEnabled(self.chk_outdir.isChecked())
        self.btn_outdir.setEnabled(self.chk_outdir.isChecked())

    def _error(self, title: str, err) -> None:
        QApplication.restoreOverrideCursor()
        QMessageBox.critical(self, title, str(err))

    def _choose_outdir(self):
        d = QFileDialog.getExistingDirectory(self, "Output folder", self.ed_outdir.text() or str(Path.home()))
        if d:
            self.ed_outdir.setText(d)

    # ================================================================== files
    def dragEnterEvent(self, e):  # noqa: N802
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):  # noqa: N802
        paths = [Path(u.toLocalFile()) for u in e.mimeData().urls() if u.isLocalFile()]
        exrs = [p for p in paths if p.suffix.lower() == ".exr"]
        others = [p for p in paths if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
        if exrs:
            self.open_files(exrs)
        elif others and self.src is not None:
            self.load_b(others[0])

    def open_dialog(self):
        start = self.settings.value("last_dir", str(Path.home()))
        names, _ = QFileDialog.getOpenFileNames(self, "Open ACEScg EXR", start, "OpenEXR (*.exr *.EXR)")
        if names:
            self.open_files([Path(n) for n in names])

    def open_files(self, paths: List[Path]):
        self.files = list(paths)
        self.settings.setValue("last_dir", str(paths[0].parent))
        self.file_list.blockSignals(True)
        self.file_list.clear()
        for p in self.files:
            self.file_list.addItem(p.name)
        self.file_list.blockSignals(False)
        self.file_list.setCurrentRow(0)

    def clear_files(self):
        self.files = []
        self.file_list.clear()
        self.src = self.info = self.seq = self.a_disp = self.b_img = None
        self.src_alpha = self.b_alpha = self.b_raw = None
        self.lbl_info.setText("No file loaded.")
        self.lbl_seq.setText("Open one frame of a sequence.")
        self.viewer.clear()
        self._update_enabled()

    def _on_file_selected(self, row: int):
        if 0 <= row < len(self.files):
            self.load_source(self.files[row])

    def load_source(self, path: Path):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            rgb, alpha, info = read_exr_rgba(path)
        except Exception as err:  # noqa: BLE001
            self._error("Could not read EXR", err)
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.src_bad = source_problem_mask(rgb)
        self.src = sanitize(rgb)
        self.src_alpha = np.clip(np.nan_to_num(alpha), 0.0, 1.0) if alpha is not None else None
        self.info = info
        lines = [
            f"<b>{path.name}</b>",
            f"{info.width} × {info.height} · {info.pixel_type} · {info.compression or '—'}",
            f"Channels: {info.channels}",
            "Alpha: yes" if alpha is not None else "Alpha: none (PNG alpha option does nothing)",
            f"Primaries: {info.chroma_label}",
            f"FPS: {info.fps:g}" if info.fps else "FPS: not in header",
            f"Timecode: {info.timecode}" if info.timecode else "Timecode: not in header",
            f"Range: min {float(np.nanmin(rgb)):.4g}, max {float(np.nanmax(rgb)):.4g}",
        ] + [f"<i>{n}</i>" for n in info.notes]
        if "not ACEScg" in info.chroma_label:
            lines.append("<span style='color:#e07a00'>The header says this is not ACEScg. "
                         "Conversion still assumes ACEScg.</span>")
        self.lbl_info.setText("<br>".join(lines))

        self.seq = detect_sequence(path)
        if self.seq:
            missing = self.seq.missing()
            txt = self.seq.label
            if missing:
                txt += f"<br><span style='color:#e07a00'>{len(missing)} missing frame(s) in the range</span>"
            self.lbl_seq.setText(txt)
            self.spn_in.setValue(self.seq.first)
            self.spn_out.setValue(self.seq.last)
            label = fps_label(info.fps)
            if label:
                self.cmb_fps.setCurrentText(label)
        else:
            self.lbl_seq.setText("This file is not part of a numbered sequence.")
        self.b_from_file = False
        self.refresh_a()
        self.refresh_b_preview()
        self._update_enabled()

    # ================================================================== processing
    def _on_exposure(self, v: int):
        self.lbl_exp.setText(f"{v / 10:+.1f}" if v else "0.0")
        self._refresh_timer.start()

    def _on_export_settings_changed(self, *_):
        self._update_enabled()
        if self.src is None:
            return
        if not self.cmb_alook.currentData():
            self.refresh_a(compose=False)
        self.b_from_file = False
        self.refresh_b_preview()

    def _on_alpha_settings_changed(self, *_):
        self._update_enabled()
        if self.src is None:
            return
        self.b_from_file = False
        self.refresh_a(compose=False)
        self.refresh_b_preview()

    def refresh_a(self, *_, compose: bool = True):
        if self.src is None:
            return
        try:
            gain = 2.0 ** (self.sld_exp.value() / 10.0)
            if self.alpha_on:
                # Same path as the export, shown over black so A and B line up.
                disp = self.export_display(self.src, self.src_alpha, self.a_look, gain)
                self.a_disp = disp * self.src_alpha[..., None]
            else:
                src = self.src * gain if gain != 1.0 else self.src
                self.a_disp = self.engine.convert(src, self.a_look, self.output)
        except Exception as err:  # noqa: BLE001
            self._error("Colour conversion failed", err)
            return
        if compose:
            self.compose()

    def refresh_b_preview(self):
        if self.src is None:
            return
        try:
            if self.alpha_on:
                disp = self.export_display(self.src, self.src_alpha, self.look)
                self.b_raw = compare.preview_encode(disp, self.fmt, self.spn_quality.value())
                self.b_alpha = compare.preview_alpha(self.src_alpha, self.fmt)
                self.b_img = self.b_raw * self.b_alpha[..., None]
            else:
                disp = self.engine.convert(self.src, self.look, self.output)
                self.b_raw = self.b_img = compare.preview_encode(disp, self.fmt, self.spn_quality.value())
                self.b_alpha = None
        except Exception as err:  # noqa: BLE001
            self._error("Colour conversion failed", err)
            return
        self.b_bits = 16 if self.fmt == "png16" else 8
        extra = " + alpha, shown over black" if self.alpha_on else ""
        self.b_label = f"B: preview of {FORMATS[self.fmt][0]}{extra} (not saved yet)"
        self.compose()

    def load_b_dialog(self):
        start = str(self.files[0].parent) if self.files else str(Path.home())
        name, _ = QFileDialog.getOpenFileName(self, "Load result", start, "Images (*.png *.jpg *.jpeg)")
        if name:
            self.load_b(Path(name))

    def load_b(self, path: Path):
        try:
            img, alpha, bits = read_image_rgba(path)
        except Exception as err:  # noqa: BLE001
            self._error("Could not read image", err)
            return
        if self.src is not None and img.shape[:2] != self.src.shape[:2]:
            self._error("Size mismatch", f"{path.name} is {img.shape[1]}×{img.shape[0]}, "
                                         f"the EXR is {self.src.shape[1]}×{self.src.shape[0]}.")
            return
        self.b_raw, self.b_alpha, self.b_bits, self.b_from_file = img, alpha, bits, True
        self.b_img = img * alpha[..., None] if alpha is not None else img
        extra = " + alpha, shown over black" if alpha is not None else ""
        self.b_label = f"B: {path.name} ({bits}-bit{extra}, read from disk)"
        self.compose()

    def compose(self, *_):
        if self.a_disp is None:
            return
        mode = self.cmb_mode.currentData()
        show_clip = self.chk_clip.isChecked()
        exp = self.sld_exp.value() / 10.0
        a_label = f"A: original · {LOOKS[self.a_look]}" + (f" · {exp:+.1f} stops" if exp else "")
        a8 = compare.to8(self.a_disp)
        clear_a = (self.src_alpha <= 0) if self.alpha_on else None
        if show_clip:
            a8 = compare.overlay(a8, self.a_disp, self.src_bad, clear_a)
        b8 = None
        if self.b_img is not None:
            b8 = compare.to8(self.b_img)
            if show_clip:
                clear_b = (self.b_alpha <= 0) if self.b_alpha is not None else None
                b8 = compare.overlay(b8, self.b_img, transparent=clear_b)

        stats = [f"A {compare.clip_stats(self.a_disp, self.src_bad)}"]
        if mode == "alpha":
            if self.b_alpha is not None:
                self.viewer.set_view(compare.alpha_view(self.b_alpha), left_label="Alpha of B (as stored in the PNG)")
            elif self.src_alpha is not None:
                self.viewer.set_view(compare.alpha_view(self.src_alpha), left_label="Alpha of the EXR (not exported)")
            else:
                self.viewer.set_view(np.zeros_like(a8), left_label="No alpha in this EXR")
        elif mode == "diff" and self.b_img is not None:
            img, mx, mean = compare.difference(self.a_disp, self.b_img)
            self.viewer.set_view(img, left_label="|A − B| × 10")
            stats.append(f"A vs B: max {mx:.2f}, mean {mean:.3f} (8-bit code values)")
            if exp or self.a_look != self.look:
                stats.append("Note: A uses a different look or exposure than the export.")
        elif mode == "b" and b8 is not None:
            self.viewer.set_view(b8, left_label=self.b_label)
        elif mode == "wipe" and b8 is not None:
            self.viewer.set_view(a8, b8, a_label, self.b_label, wipe=True)
        else:
            self.viewer.set_view(a8, left_label=a_label)
        self.lbl_stats.setText("<br>".join(stats))

    def _on_hover(self, x: int, y: int):
        if x < 0 or self.src is None:
            self.statusBar().clearMessage()
            return
        s = self.src[y, x]
        parts = [f"x {x}  y {y}", f"ACEScg {s[0]:.4f} {s[1]:.4f} {s[2]:.4f}"]
        if self.src_alpha is not None:
            parts[-1] += f"  A {self.src_alpha[y, x]:.4f}"
        if self.a_disp is not None:
            a = self.a_disp[y, x]
            parts.append(f"A {a[0]:.4f} {a[1]:.4f} {a[2]:.4f}")
        if self.b_raw is not None:
            top = (1 << self.b_bits) - 1
            b = np.rint(self.b_raw[y, x] * top).astype(int)
            txt = f"B {b[0]} {b[1]} {b[2]}"
            if self.b_alpha is not None:
                txt += f" α {int(np.rint(self.b_alpha[y, x] * top))}"
            parts.append(txt + f" /{top}")
        self.statusBar().showMessage("    ".join(parts))

    # ================================================================== export
    def export_stills(self):
        if not self.files:
            return
        ext = FORMATS[self.fmt][1]
        written, errors = [], []
        dlg = QProgressDialog("Exporting stills…", "Cancel", 0, len(self.files), self)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(400)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            current = self.files[self.file_list.currentRow()] if self.file_list.currentRow() >= 0 else None
            for i, p in enumerate(self.files):
                dlg.setValue(i)
                dlg.setLabelText(f"Exporting {p.name} ({i + 1}/{len(self.files)})")
                QApplication.processEvents()
                if dlg.wasCanceled():
                    errors.append("Cancelled; the remaining files were not exported.")
                    break
                try:
                    if p == current and self.src is not None:
                        rgb, alpha = self.src, self.src_alpha
                    else:
                        rgb, alpha, _info = read_exr_rgba(p)
                        rgb = sanitize(rgb)
                        alpha = np.clip(np.nan_to_num(alpha), 0.0, 1.0) if alpha is not None else None
                    write_alpha = (self.fmt in ("png16", "png8") and self.chk_alpha.isChecked()
                                   and alpha is not None)
                    if write_alpha:
                        disp = self.export_display(rgb, alpha, self.look)
                    else:
                        disp = self.engine.convert(rgb, self.look, self.output)
                    out = output_path(p, self.suffix(), ext, self.out_dir())
                    write_still(out, disp, self.fmt, self.output, self.spn_quality.value(),
                                alpha=alpha if write_alpha else None)
                    written.append(out)
                    if p == current:
                        self.load_b(out)
                except Exception as err:  # noqa: BLE001
                    errors.append(f"{p.name}: {err}")
        finally:
            dlg.setValue(len(self.files))
            QApplication.restoreOverrideCursor()
        msg = f"Wrote {len(written)} file(s)" + (f": {written[0].name}" if len(written) == 1 else "")
        self.statusBar().showMessage(msg, 8000)
        if errors:
            QMessageBox.warning(self, APP_NAME, msg + "\n\nProblems:\n" + "\n".join(errors))

    def export_movie(self):
        if self.seq is None:
            return
        start, end = self.spn_in.value(), self.spn_out.value()
        if end < start:
            self._error("Frame range", "The out frame is before the in frame.")
            return
        outside = start < self.seq.first or end > self.seq.last
        missing = self.seq.missing(start, end)
        if outside or missing:
            msg = []
            if outside:
                msg.append(f"The range {start}-{end} goes past the frames on disk "
                           f"({self.seq.first}-{self.seq.last}).")
            if missing and not outside:
                msg.append(f"{len(missing)} frame(s) are missing: " + ", ".join(map(str, missing[:10])))
            self._error("Frames missing", "\n".join(msg) + "\n\nNothing was exported.")
            return
        fps = self.cmb_fps.currentText().strip()
        try:
            if fps_rational(fps) <= 0:
                raise ValueError
        except Exception:  # noqa: BLE001
            self._error("Frame rate", f"'{self.cmb_fps.currentText()}' is not a valid frame rate.")
            return
        half = bool(self.cmb_size.currentData())
        out = movie_path(self.seq, self.output, half, self.look, self.chk_lookname.isChecked(), self.out_dir())
        job = MovieJob(self.seq, start, end, self.look, self.output, self.cmb_codec.currentData(), half, fps, out)
        self.progress.setRange(0, end - start + 1)
        self.progress.setValue(0)
        self.progress.setFormat(f"{out.name}  %v / %m")
        self.movie_thread = MovieThread(job, self.engine)
        self.movie_thread.progress.connect(lambda i, n: self.progress.setValue(i))
        self.movie_thread.finished_ok.connect(self._movie_done)
        self.movie_thread.failed.connect(self._movie_failed)
        self.movie_thread.finished.connect(self._update_enabled)
        self.movie_thread.start()
        self._update_enabled()

    def cancel_movie(self):
        if self.movie_thread:
            self.movie_thread.cancel_event.set()

    def _movie_done(self, path: str):
        self.progress.setFormat(f"Done: {Path(path).name}")
        self.statusBar().showMessage(f"Wrote {path}", 10000)

    def _movie_failed(self, msg: str):
        self.progress.setFormat("Failed" if "Cancelled" not in msg else "Cancelled")
        QMessageBox.warning(self, APP_NAME, msg)


def run_gui(argv) -> int:
    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName(APP_NAME)
    try:
        engine = ColorEngine()
    except Exception:  # noqa: BLE001
        QMessageBox.critical(None, APP_NAME, "Could not start the colour engine:\n\n" + traceback.format_exc())
        return 1
    win = MainWindow(engine)
    files = [Path(a) for a in argv[1:] if a.lower().endswith(".exr") and Path(a).is_file()]
    win.show()
    if files:
        win.open_files(files)
    return app.exec()
