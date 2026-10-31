# Author: Paulo R. A. Gilz
"""Dark and light themes for the Qt widgets and the plots.

Qt's default palette on Linux and Windows differ enough that an application
looking correct on one can be unreadable on the other, so the stylesheet sets
every colour it depends on rather than inheriting from the desktop.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    name: str
    window: str
    panel: str
    surface: str
    border: str
    text: str
    text_dim: str
    accent: str
    accent_text: str
    plot_background: str
    plot_foreground: str
    plot_grid: str
    selection: str


DARK = Theme(
    name="dark",
    window="#121416",
    panel="#1a1d21",
    surface="#22262b",
    border="#32383f",
    text="#e6e9ec",
    text_dim="#8b949e",
    accent="#3fa9f5",
    accent_text="#06121c",
    plot_background="#0e1013",
    plot_foreground="#c9d1d9",
    plot_grid="#2a3038",
    selection="#1f4a6b",
)

LIGHT = Theme(
    name="light",
    window="#f2f4f6",
    panel="#e9ecef",
    surface="#ffffff",
    border="#c7ced6",
    text="#1a1d21",
    text_dim="#5a636d",
    accent="#0b78d0",
    accent_text="#ffffff",
    plot_background="#ffffff",
    plot_foreground="#22262b",
    plot_grid="#d5dbe1",
    selection="#bcdcf5",
)


def theme_for(dark: bool) -> Theme:
    return DARK if dark else LIGHT


# ------------------------------------------------------------------ type scale
#
# Every size in the interface, in one place, because they only work as a set:
# shrinking the panels but not the tiles, or the text but not the padding
# around it, gives you a smaller-looking application that is no less crowded.
#
# The sizes below are deliberately compact. This is a tool for comparing plots,
# so the screen belongs to the plots: on a laptop a full page of tiles plus ten
# panels of controls was legible but dense, and every pixel spent on chrome is
# one the data does not get.
#
# Qt sizes widget text in *pixels* here rather than points. Points would be the
# usual choice, but a stylesheet that sets a pixel size makes QFont.pointSizeF()
# report -1, and code that reads a font back to adjust it -- the tab rail, the
# tile captions -- then has to guess. Pixels keep one unit throughout, and the
# constants below are the only places these numbers appear.
FONT_PX = 10          # body text: menus, panels, buttons, lists, tables
FONT_SMALL_PX = 9     # hints, section titles, table headers, the status bar

# The plot tiles. pyqtgraph wants CSS-ish strings and QFont point sizes rather
# than pixels, so these are points -- roughly 0.75 x the pixel figures above at
# the 96 dpi Qt assumes, which is what keeps a tile title in step with a panel
# heading instead of shouting over it.
PLOT_TITLE_PT = 7.5     # the tile title
PLOT_AXIS_PT = 7.0      # axis names and their tick numbers
PLOT_LEGEND_PT = 6.0    # legend rows -- smaller than the axes on purpose:
                        # a legend is a key that is read once and then only
                        # glanced at, and on a 3x3 grid it was eating the
                        # corner of every tile at the axis size.
PLOT_LABEL_PT = 6.5     # crosshair readout, slope figure, captions

# The vertical tab rail down the left edge, whose text is drawn rotated.
RAIL_FONT_PT = 6.5


# --------------------------------------------------------------- UI scale
#
# Every figure above is a *base* size, and everything that draws multiplies it
# by this. One knob rather than a set of them: text set two pixels smaller
# inside unchanged padding looks shrunken rather than roomier, so the padding,
# the tick boxes, the scrollbars and the rail move with the type or none of it
# should.
#
# It exists because the sizes above are one compromise, chosen for a 1080-high
# screen, and screens are not all that. A 1050-high Linux laptop with a desktop
# panel has less usable height than the panels wanted; a 4K screen at 100 %
# has more than they need.
UI_SCALE_MIN = 0.70
UI_SCALE_MAX = 1.60
UI_SCALE_STEP = 0.10

# What a session starts at, and the one place to change it.
#
# 0.90 rather than 1.00 since 2.2.1. The base figures above were chosen against
# a 1080-high screen and are a little generous on most of the screens this is
# actually used on: at 100 % the panels want more height than a 1050-high laptop
# with a desktop panel has to give, which is the problem the scroll areas were
# added to contain. Starting one step down fits the whole interface on every
# screen it has been run on, and the toolbar's two buttons move it either way
# for anyone who would rather have the larger type.
UI_SCALE_DEFAULT = 0.90

_ui_scale = UI_SCALE_DEFAULT


def ui_scale() -> float:
    """The current multiplier.

    1.0 means the base sizes above, unscaled; it is the unit of the scale, not
    the value a session starts at -- that is UI_SCALE_DEFAULT.
    """
    return _ui_scale


def set_ui_scale(value: float) -> float:
    """Set the multiplier, clamped. Returns what it actually became.

    Clamped rather than refused: the buttons that drive this disable
    themselves at the ends, and a workspace written by a future version with a
    wider range must still open.
    """
    global _ui_scale
    _ui_scale = max(UI_SCALE_MIN, min(UI_SCALE_MAX, round(float(value), 3)))
    return _ui_scale


def px(base: float) -> int:
    """A pixel figure at the current scale, never rounded away to nothing."""
    return max(1, int(round(base * _ui_scale)))


def pt(base: float) -> float:
    """A point size at the current scale, for Qt fonts and pyqtgraph strings."""
    return round(base * _ui_scale, 2)


def plot_font(size_pt: float, bold: bool = False):
    """A QFont at one of the sizes above.

    Here rather than in the drawing code so that the tiles cannot drift away
    from the panels: everything that paints text on a plot asks for it by name.
    """
    from PySide6.QtGui import QFont

    font = QFont()
    font.setPointSizeF(pt(size_pt))
    font.setBold(bold)
    return font


def stylesheet(theme: Theme) -> str:
    """Application-wide Qt stylesheet for the given theme.

    Sizes come from the type scale above rather than being written in here, so
    that the tiles and the panels stay in step. The padding figures are part of
    the same decision: text set two pixels smaller inside unchanged padding
    looks shrunken rather than roomier.
    """
    return """
    QWidget {{
        background: {window};
        color: {text};
        font-size: {font_px}px;
    }}
    QMainWindow::separator {{ background: {border}; width: {sep}px; height: {sep}px; }}

    QFrame#Panel, QWidget#Panel, QScrollArea#Panel {{ background: {panel}; }}
    /* The panels scroll, so their bar sits inside the panel rather than
       against the window, and needs the panel's ground to disappear into. */
    QScrollArea#Panel > QWidget > QWidget {{ background: {panel}; }}
    QLabel#SectionTitle {{
        color: {text_dim};
        font-size: {small_px}px;
        font-weight: 600;
        padding: {pad_b}px {pad_s}px {pad_t}px {pad_s}px;
        text-transform: uppercase;
    }}
    /* Hints are read once and then skipped over for the rest of the session,
       so they are the first thing that should give up space. */
    QLabel#Hint {{ color: {text_dim}; font-size: {small_px}px; }}
    QLabel#Error {{ color: #ff6b6b; font-size: {small_px}px; }}

    QPushButton {{
        background: {surface};
        border: 1px solid {border};
        border-radius: 3px;
        padding: {pad_v}px {pad_h}px;
    }}
    QPushButton:hover {{ border-color: {accent}; }}
    QPushButton:pressed {{ background: {selection}; }}
    QPushButton:disabled {{ color: {text_dim}; border-color: {border}; }}
    QPushButton:checked {{ background: {accent}; color: {accent_text}; border-color: {accent}; }}

    QToolButton {{
        background: transparent;
        border: 1px solid transparent;
        border-radius: 3px;
        padding: {pad_s}px;
    }}
    QToolButton:hover {{ background: {surface}; border-color: {border}; }}
    QToolButton:checked {{ background: {accent}; color: {accent_text}; }}
    /* Without this a disabled tool button is indistinguishable from an enabled
       one, so an unavailable action reads as an unresponsive one. */
    QToolButton:disabled {{ color: {text_dim}; background: transparent; }}
    QToolButton:disabled:hover {{ background: transparent; border-color: transparent; }}

    QLineEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
        background: {surface};
        border: 1px solid {border};
        border-radius: 3px;
        padding: {pad_s}px {pad_b}px;
        selection-background-color: {accent};
        selection-color: {accent_text};
    }}
    QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus {{ border-color: {accent}; }}
    QComboBox::drop-down {{ border: none; width: {drop}px; }}
    QComboBox QAbstractItemView {{
        background: {surface};
        border: 1px solid {border};
        selection-background-color: {accent};
        selection-color: {accent_text};
    }}

    QTreeWidget, QListWidget, QTableWidget {{
        background: {surface};
        border: 1px solid {border};
        alternate-background-color: {panel};
    }}
    QTreeWidget::item, QListWidget::item {{ padding: {pad_t}px {pad_t}px; }}
    QTreeWidget::item:selected, QListWidget::item:selected {{
        background: {selection};
        color: {text};
    }}
    QHeaderView::section {{
        background: {panel};
        border: none;
        border-bottom: 1px solid {border};
        padding: {pad_s}px;
        font-size: {small_px}px;
        color: {text_dim};
    }}

    /* The ISO tab's read-only "every filter" box. Panel-coloured rather than
       surface-coloured, so it reads as a report rather than as another list
       you could click in. */
    QTextBrowser#Overview {{
        background: {panel};
        border: 1px solid {border};
        font-size: {small_px}px;
    }}

    QTabWidget::pane {{ border: 1px solid {border}; }}
    QTabBar::tab {{
        background: {panel};
        border: 1px solid {border};
        padding: {pad_v}px {pad_h}px;
        margin-right: {pad_t}px;
    }}
    QTabBar::tab:selected {{ background: {surface}; border-bottom-color: {surface}; }}

    QScrollBar:vertical {{ background: {panel}; width: {bar}px; margin: 0; }}
    QScrollBar::handle:vertical {{
        background: {border}; border-radius: 5px; min-height: {barmin}px;
    }}
    QScrollBar::handle:vertical:hover {{ background: {accent}; }}
    QScrollBar:horizontal {{ background: {panel}; height: {bar}px; margin: 0; }}
    QScrollBar::handle:horizontal {{
        background: {border}; border-radius: 5px; min-width: {barmin}px;
    }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}

    QStatusBar {{
        background: {panel};
        color: {text_dim};
        font-size: {small_px}px;
    }}
    QMenuBar {{ background: {panel}; }}
    QMenuBar::item:selected {{ background: {accent}; color: {accent_text}; }}
    QMenu {{ background: {surface}; border: 1px solid {border}; }}
    QMenu::item:selected {{ background: {accent}; color: {accent_text}; }}

    /* Check and radio indicators are drawn explicitly rather than left to the
       platform style. Once the application sets a palette-changing stylesheet,
       Qt keeps drawing its own indicators against the *system* colours -- so on
       the dark theme a selected radio button was a dark dot on a dark ground
       and there was no way to tell which of Simulation / Parameter / ISO was
       active. These rules make the checked state a filled accent colour, which
       is unambiguous in both themes.

       The item views are listed alongside, and that is the part that was
       missing: the tick boxes beside a run in DATA, a parameter in PARAMS or a
       value in ISO are drawn by QTreeView/QListView, not by QCheckBox, so none
       of the rules below reached them. They kept the platform's own indicator,
       whose unchecked box is a dark outline on a dark ground -- invisible, on
       the tab whose whole job is choosing what to plot.

       The border is {text_dim} rather than {border}: an outline has to be
       findable against the surface behind it before it can be read as empty or
       filled, and the panel border colour is deliberately quiet. */
    QCheckBox, QRadioButton {{ spacing: {pad_b}px; color: {text}; }}
    QCheckBox::indicator, QRadioButton::indicator,
    QTreeView::indicator, QListView::indicator, QTableView::indicator {{
        width: {tick}px;
        height: {tick}px;
        border: 1px solid {text_dim};
        border-radius: 3px;
        background: {surface};
    }}
    QRadioButton::indicator {{ border-radius: {tickr}px; }}
    QCheckBox::indicator:hover, QRadioButton::indicator:hover,
    QTreeView::indicator:hover, QListView::indicator:hover,
    QTableView::indicator:hover {{
        border: 1px solid {accent};
    }}
    QCheckBox::indicator:checked,
    QTreeView::indicator:checked, QListView::indicator:checked,
    QTableView::indicator:checked {{
        background: {accent};
        border: 1px solid {accent};
    }}
    /* Partly-checked reads as neither on nor off: the accent, dimmed, on the
       same outline. */
    QTreeView::indicator:indeterminate, QListView::indicator:indeterminate {{
        background: {selection};
        border: 1px solid {accent};
    }}
    QRadioButton::indicator:checked {{
        border: 1px solid {accent};
        background: qradialgradient(cx:0.5, cy:0.5, radius:0.5,
                                    fx:0.5, fy:0.5,
                                    stop:0 {accent}, stop:0.55 {accent},
                                    stop:0.6 {surface}, stop:1 {surface});
    }}
    QCheckBox::indicator:disabled, QRadioButton::indicator:disabled,
    QTreeView::indicator:disabled, QListView::indicator:disabled {{
        border: 1px solid {border};
        background: {panel};
    }}
    QSplitter::handle {{ background: {border}; }}
    QSplitter::handle:hover {{ background: {accent}; }}
    QGroupBox {{
        border: 1px solid {border};
        border-radius: 3px;
        margin-top: {pad_h}px;
        padding-top: {pad_g}px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 8px;
        font-size: {small_px}px;
        color: {text_dim};
    }}
    """.format(
        font_px=px(FONT_PX), small_px=px(FONT_SMALL_PX),
        sep=px(3),        # the splitter grip between the panes
        pad_t=px(2),      # the tightest padding there is: list rows
        pad_s=px(3),      # tool buttons, header sections, field insets
        pad_v=px(4),      # the vertical half of a button's padding
        pad_b=px(5),      # the horizontal half of a field's, and tick spacing
        pad_g=px(6),      # a group box's inner top
        pad_h=px(9),      # the horizontal half of a button's padding
        bar=px(11),       # scrollbar thickness
        drop=px(18),      # a combo's arrow well
        barmin=px(24),    # the shortest a scrollbar handle may get
        tick=px(12),      # check and radio indicators
        tickr=px(7),      # which makes a radio indicator round
        **theme.__dict__)


def apply_pyqtgraph_theme(theme: Theme) -> None:
    """Configure pyqtgraph's globals to match.

    ``antialias`` stays off on purpose: it is the single most expensive
    pyqtgraph option, and with min/max decimation the curves are already at
    pixel resolution, so it buys almost nothing visually while costing a lot on
    a grid of tiles.
    """
    import pyqtgraph as pg

    pg.setConfigOption("background", theme.plot_background)
    pg.setConfigOption("foreground", theme.plot_foreground)
    pg.setConfigOption("antialias", False)
    pg.setConfigOption("useOpenGL", False)
    pg.setConfigOption("imageAxisOrder", "row-major")
