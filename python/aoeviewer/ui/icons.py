# Author: Paulo R. A. Gilz
"""Toolbar icons, drawn rather than loaded.

Every icon here is painted with QPainter at request time. That is deliberate:

  * No asset files to ship, to lose, or to get wrong in a PyInstaller bundle --
    a missing .png in a frozen build is a blank toolbar with no error.
  * They follow the theme. A dark-on-dark icon set is the usual result of
    shipping fixed images, and this application has both a light and a dark
    theme; here the colour is a parameter.
  * They are rendered at whatever size is asked for, so a high-DPI display
    gets a sharp icon instead of an upscaled 16 px bitmap.

Each painter function receives a QPainter already scaled to a 100x100 box, so
the geometry below reads in percentages and is resolution-independent.
"""

from __future__ import annotations

from typing import Callable, Dict

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QIcon,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
)

CANVAS = 100.0


def _pen(color: QColor, width: float = 9.0) -> QPen:
    pen = QPen(color)
    pen.setWidthF(width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    return pen


# --------------------------------------------------------------- mouse modes


def _draw_select(painter: QPainter, color: QColor) -> None:
    """A standard arrow cursor."""
    arrow = QPolygonF([
        QPointF(26, 14), QPointF(26, 78), QPointF(42, 63),
        QPointF(52, 86), QPointF(64, 80), QPointF(54, 58),
        QPointF(74, 56),
    ])
    painter.setPen(_pen(color, 7.0))
    painter.setBrush(QBrush(color))
    painter.drawPolygon(arrow)


def _draw_crosshair(painter: QPainter, color: QColor) -> None:
    """A crosshair with a gap at the centre, so the target stays visible."""
    painter.setPen(_pen(color, 8.0))
    painter.drawLine(QPointF(50, 8), QPointF(50, 36))
    painter.drawLine(QPointF(50, 64), QPointF(50, 92))
    painter.drawLine(QPointF(8, 50), QPointF(36, 50))
    painter.drawLine(QPointF(64, 50), QPointF(92, 50))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(_pen(color, 6.0))
    painter.drawEllipse(QRectF(36, 36, 28, 28))


def _draw_pan(painter: QPainter, color: QColor) -> None:
    """A pointing hand: index finger up, thumb out, three fingers curled."""
    painter.setPen(_pen(color, 7.0))
    painter.setBrush(Qt.BrushStyle.NoBrush)

    palm = QPainterPath()
    palm.moveTo(38, 46)          # base of the index finger
    palm.lineTo(38, 16)          # index finger, pointing up
    palm.arcTo(QRectF(38, 8, 16, 16), 180, -180)
    palm.lineTo(54, 40)
    palm.lineTo(54, 34)          # middle finger
    palm.arcTo(QRectF(54, 26, 14, 14), 180, -180)
    palm.lineTo(68, 44)
    palm.lineTo(68, 38)          # ring finger
    palm.arcTo(QRectF(68, 30, 13, 13), 180, -180)
    palm.lineTo(81, 62)
    palm.cubicTo(81, 84, 66, 92, 52, 92)
    palm.cubicTo(38, 92, 30, 84, 26, 72)
    palm.lineTo(18, 52)          # thumb
    palm.cubicTo(15, 45, 24, 39, 29, 45)
    palm.lineTo(38, 58)
    palm.closeSubpath()
    painter.drawPath(palm)


def _draw_zoom(painter: QPainter, color: QColor) -> None:
    """A magnifier with a plus inside, so it reads as zoom rather than search."""
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(_pen(color, 9.0))
    painter.drawEllipse(QRectF(14, 12, 54, 54))
    painter.setPen(_pen(color, 12.0))
    painter.drawLine(QPointF(64, 62), QPointF(90, 88))
    painter.setPen(_pen(color, 8.0))
    painter.drawLine(QPointF(41, 27), QPointF(41, 51))
    painter.drawLine(QPointF(29, 39), QPointF(53, 39))


# ------------------------------------------------------------------ actions


def _draw_open(painter: QPainter, color: QColor) -> None:
    painter.setPen(_pen(color, 8.0))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    path = QPainterPath()
    path.moveTo(10, 78)
    path.lineTo(10, 24)
    path.lineTo(40, 24)
    path.lineTo(48, 36)
    path.lineTo(82, 36)
    path.lineTo(82, 48)
    painter.drawPath(path)
    lid = QPolygonF([QPointF(10, 78), QPointF(24, 48),
                     QPointF(96, 48), QPointF(82, 78)])
    painter.drawPolygon(lid)


def _draw_save(painter: QPainter, color: QColor) -> None:
    painter.setPen(_pen(color, 8.0))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    body = QPainterPath()
    body.moveTo(16, 14)
    body.lineTo(70, 14)
    body.lineTo(86, 30)
    body.lineTo(86, 86)
    body.lineTo(16, 86)
    body.closeSubpath()
    painter.drawPath(body)
    painter.drawRect(QRectF(32, 14, 34, 24))   # the shutter
    painter.drawRect(QRectF(30, 54, 42, 32))   # the label


def _draw_image(painter: QPainter, color: QColor) -> None:
    painter.setPen(_pen(color, 8.0))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawRect(QRectF(12, 20, 76, 60))
    painter.setBrush(QBrush(color))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(QRectF(26, 32, 14, 14))
    painter.setPen(_pen(color, 8.0))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    ridge = QPainterPath()
    ridge.moveTo(16, 72)
    ridge.lineTo(38, 50)
    ridge.lineTo(54, 64)
    ridge.lineTo(68, 48)
    ridge.lineTo(84, 70)
    painter.drawPath(ridge)


def _draw_table(painter: QPainter, color: QColor) -> None:
    painter.setPen(_pen(color, 7.0))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawRect(QRectF(12, 18, 76, 64))
    painter.drawLine(QPointF(12, 38), QPointF(88, 38))
    painter.drawLine(QPointF(12, 60), QPointF(88, 60))
    painter.drawLine(QPointF(42, 18), QPointF(42, 82))
    painter.drawLine(QPointF(66, 18), QPointF(66, 82))


def _draw_fit(painter: QPainter, color: QColor) -> None:
    """Four corner brackets: the universal "fit to view"."""
    painter.setPen(_pen(color, 9.0))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    for x_sign, y_sign, corner in (
        (1, 1, (14, 14)), (-1, 1, (86, 14)), (1, -1, (14, 86)), (-1, -1, (86, 86)),
    ):
        cx, cy = corner
        painter.drawLine(QPointF(cx, cy), QPointF(cx + 22 * x_sign, cy))
        painter.drawLine(QPointF(cx, cy), QPointF(cx, cy + 22 * y_sign))
    painter.setPen(_pen(color, 7.0))
    painter.drawLine(QPointF(34, 60), QPointF(48, 42))
    painter.drawLine(QPointF(48, 42), QPointF(60, 54))
    painter.drawLine(QPointF(60, 54), QPointF(70, 38))


def _draw_slope(painter: QPainter, color: QColor) -> None:
    painter.setPen(_pen(color, 8.0))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawLine(QPointF(16, 82), QPointF(84, 24))
    painter.setPen(_pen(color, 6.0))
    painter.drawLine(QPointF(16, 82), QPointF(84, 82))
    painter.drawLine(QPointF(84, 82), QPointF(84, 24))


def _draw_cube(painter: QPainter, color: QColor) -> None:
    """An isometric box, for the 3D tab."""
    painter.setPen(_pen(color, 7.0))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    top = QPolygonF([QPointF(50, 12), QPointF(88, 32),
                     QPointF(50, 52), QPointF(12, 32)])
    painter.drawPolygon(top)
    painter.drawLine(QPointF(12, 32), QPointF(12, 66))
    painter.drawLine(QPointF(88, 32), QPointF(88, 66))
    painter.drawLine(QPointF(50, 52), QPointF(50, 88))
    painter.drawLine(QPointF(12, 66), QPointF(50, 88))
    painter.drawLine(QPointF(88, 66), QPointF(50, 88))


_PAINTERS: Dict[str, Callable[[QPainter, QColor], None]] = {
    "select": _draw_select,
    "crosshair": _draw_crosshair,
    "pan": _draw_pan,
    "zoom": _draw_zoom,
    "open": _draw_open,
    "save": _draw_save,
    "image": _draw_image,
    "table": _draw_table,
    "fit": _draw_fit,
    "slope": _draw_slope,
    "cube": _draw_cube,
}

_CACHE: Dict[tuple, QIcon] = {}


def icon(name: str, color: str, size: int = 20) -> QIcon:
    """A themed icon by name. Cached, because toolbars ask repeatedly."""
    key = (name, color, size)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    painter_fn = _PAINTERS.get(name)
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)

    if painter_fn is not None:
        painter = QPainter(pixmap)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.scale(size / CANVAS, size / CANVAS)
            painter_fn(painter, QColor(color))
        finally:
            # Not optional: a QPainter still active when its QPixmap is
            # destroyed takes the process with it.
            painter.end()

    result = QIcon(pixmap)
    _CACHE[key] = result
    return result


def clear_cache() -> None:
    """Drop cached icons, so a theme change repaints them."""
    _CACHE.clear()
