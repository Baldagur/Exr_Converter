"""Zoomable image viewer with an optional A/B wipe."""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget


def to_qimage(img: np.ndarray) -> QImage:
    """uint8 (H, W, 3) -> QImage that owns its memory."""
    img = np.ascontiguousarray(img, dtype=np.uint8)
    h, w = img.shape[:2]
    return QImage(img.data, w, h, 3 * w, QImage.Format_RGB888).copy()


class ImageViewer(QWidget):
    hovered = Signal(int, int)  # image pixel under the mouse, (-1, -1) when outside

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(400, 300)
        self._left: Optional[QImage] = None
        self._right: Optional[QImage] = None
        self._left_label = ""
        self._right_label = ""
        self._wipe = False
        self.wipe_pos = 0.5  # fraction of the image width
        self._scale = 1.0
        self._offset = QPointF(0, 0)
        self._fitted = True
        self._drag = None  # "pan" | "wipe"
        self._last = QPointF()
        self._img_size = (0, 0)

    # ------------------------------------------------------------------ content
    def set_view(self, left: Optional[np.ndarray], right: Optional[np.ndarray] = None,
                 left_label: str = "", right_label: str = "", wipe: bool = False) -> None:
        new_size = (left.shape[1], left.shape[0]) if left is not None else (0, 0)
        self._left = to_qimage(left) if left is not None else None
        self._right = to_qimage(right) if (right is not None and wipe) else None
        self._left_label, self._right_label, self._wipe = left_label, right_label, wipe
        if new_size != self._img_size:
            self._img_size = new_size
            self._fitted = True
        if self._fitted:
            self.fit()
        self.update()

    def clear(self) -> None:
        self.set_view(None)

    # ------------------------------------------------------------------ geometry
    def fit(self) -> None:
        w, h = self._img_size
        if not w or not h:
            return
        s = min((self.width() - 20) / w, (self.height() - 20) / h)
        self._scale = max(s, 0.01)
        self._offset = QPointF((self.width() - w * self._scale) / 2, (self.height() - h * self._scale) / 2)
        self._fitted = True
        self.update()

    def one_to_one(self) -> None:
        w, h = self._img_size
        self._scale = 1.0
        self._offset = QPointF((self.width() - w) / 2, (self.height() - h) / 2)
        self._fitted = False
        self.update()

    def _to_image(self, p: QPointF) -> QPointF:
        return (p - self._offset) / self._scale

    def _wipe_x_widget(self) -> float:
        return self._offset.x() + self.wipe_pos * self._img_size[0] * self._scale

    def resizeEvent(self, e):  # noqa: N802
        if self._fitted:
            self.fit()
        super().resizeEvent(e)

    # ------------------------------------------------------------------ painting
    def paintEvent(self, e):  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(38, 38, 38))
        if self._left is None:
            p.setPen(QColor(150, 150, 150))
            p.drawText(self.rect(), Qt.AlignCenter, "Open or drop an ACEScg EXR")
            return
        w, h = self._img_size
        p.setRenderHint(QPainter.SmoothPixmapTransform, self._scale < 1.0)
        target = QRectF(self._offset.x(), self._offset.y(), w * self._scale, h * self._scale)
        p.drawImage(target, self._left)
        if self._wipe and self._right is not None:
            wx = self.wipe_pos * w
            src = QRectF(wx, 0, w - wx, h)
            dst = QRectF(self._offset.x() + wx * self._scale, self._offset.y(),
                         (w - wx) * self._scale, h * self._scale)
            p.drawImage(dst, self._right, src)
            x = self._wipe_x_widget()
            p.setPen(QPen(QColor(255, 255, 255, 220), 1))
            p.drawLine(QPointF(x, target.top()), QPointF(x, target.bottom()))
            p.setBrush(QColor(255, 255, 255, 220))
            p.drawEllipse(QPointF(x, target.center().y()), 5, 5)
        p.setFont(QFont(self.font().family(), 9))
        self._label(p, self._left_label, Qt.AlignLeft)
        if self._wipe and self._right_label:
            self._label(p, self._right_label, Qt.AlignRight)
        p.setPen(QColor(170, 170, 170))
        p.drawText(self.rect().adjusted(8, 0, -8, -6), Qt.AlignBottom | Qt.AlignRight,
                   f"{self._scale * 100:.0f}%")

    def _label(self, p: QPainter, text: str, align) -> None:
        if not text:
            return
        r = p.fontMetrics().boundingRect(text).adjusted(-6, -3, 6, 3)
        r.moveTop(8)
        if align == Qt.AlignLeft:
            r.moveLeft(8)
        else:
            r.moveRight(self.width() - 8)
        p.fillRect(r, QColor(0, 0, 0, 170))
        p.setPen(QColor(235, 235, 235))
        p.drawText(r, Qt.AlignCenter, text)

    # ------------------------------------------------------------------ interaction
    def wheelEvent(self, e):  # noqa: N802
        if self._left is None:
            return
        factor = 1.25 if e.angleDelta().y() > 0 else 0.8
        pos = e.position()
        before = self._to_image(pos)
        self._scale = min(max(self._scale * factor, 0.02), 64.0)
        self._offset = pos - before * self._scale
        self._fitted = False
        self.update()

    def mousePressEvent(self, e):  # noqa: N802
        self._last = e.position()
        if e.button() == Qt.LeftButton and self._wipe and abs(e.position().x() - self._wipe_x_widget()) < 8:
            self._drag = "wipe"
        elif e.button() in (Qt.LeftButton, Qt.MiddleButton):
            self._drag = "pan"

    def mouseMoveEvent(self, e):  # noqa: N802
        pos = e.position()
        if self._drag == "pan":
            self._offset += pos - self._last
            self._fitted = False
            self.update()
        elif self._drag == "wipe" and self._img_size[0]:
            self.wipe_pos = min(max(self._to_image(pos).x() / self._img_size[0], 0.0), 1.0)
            self.update()
        self._last = pos
        if self._wipe:
            near = abs(pos.x() - self._wipe_x_widget()) < 8
            self.setCursor(Qt.SplitHCursor if near or self._drag == "wipe" else Qt.CrossCursor)
        else:
            self.setCursor(Qt.CrossCursor)
        ip = self._to_image(pos)
        x, y = int(np.floor(ip.x())), int(np.floor(ip.y()))
        w, h = self._img_size
        self.hovered.emit(x if 0 <= x < w and 0 <= y < h else -1, y if 0 <= x < w and 0 <= y < h else -1)

    def mouseReleaseEvent(self, e):  # noqa: N802
        self._drag = None

    def mouseDoubleClickEvent(self, e):  # noqa: N802
        self.fit()

    def leaveEvent(self, e):  # noqa: N802
        self.hovered.emit(-1, -1)

    def keyPressEvent(self, e):  # noqa: N802
        if e.key() == Qt.Key_F:
            self.fit()
        elif e.key() == Qt.Key_1:
            self.one_to_one()
        else:
            super().keyPressEvent(e)
