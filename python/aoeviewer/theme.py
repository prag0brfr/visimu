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


def stylesheet(theme: Theme) -> str:
    """Application-wide Qt stylesheet for the given theme."""
    return """
    QWidget {{
        background: {window};
        color: {text};
        font-size: 12px;
    }}
    QMainWindow::separator {{ background: {border}; width: 3px; height: 3px; }}

    QFrame#Panel, QWidget#Panel {{ background: {panel}; }}
    QLabel#SectionTitle {{
        color: {text_dim};
        font-weight: 600;
        padding: 6px 4px 2px 4px;
        text-transform: uppercase;
    }}
    QLabel#Hint {{ color: {text_dim}; }}
    QLabel#Error {{ color: #ff6b6b; }}

    QPushButton {{
        background: {surface};
        border: 1px solid {border};
        border-radius: 3px;
        padding: 5px 10px;
    }}
    QPushButton:hover {{ border-color: {accent}; }}
    QPushButton:pressed {{ background: {selection}; }}
    QPushButton:disabled {{ color: {text_dim}; border-color: {border}; }}
    QPushButton:checked {{ background: {accent}; color: {accent_text}; border-color: {accent}; }}

    QToolButton {{
        background: transparent;
        border: 1px solid transparent;
        border-radius: 3px;
        padding: 4px;
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
        padding: 4px 6px;
        selection-background-color: {accent};
        selection-color: {accent_text};
    }}
    QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus {{ border-color: {accent}; }}
    QComboBox::drop-down {{ border: none; width: 18px; }}
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
    QTreeWidget::item, QListWidget::item {{ padding: 3px 2px; }}
    QTreeWidget::item:selected, QListWidget::item:selected {{
        background: {selection};
        color: {text};
    }}
    QHeaderView::section {{
        background: {panel};
        border: none;
        border-bottom: 1px solid {border};
        padding: 4px;
        color: {text_dim};
    }}

    QTabWidget::pane {{ border: 1px solid {border}; }}
    QTabBar::tab {{
        background: {panel};
        border: 1px solid {border};
        padding: 5px 12px;
        margin-right: 2px;
    }}
    QTabBar::tab:selected {{ background: {surface}; border-bottom-color: {surface}; }}

    QScrollBar:vertical {{ background: {panel}; width: 11px; margin: 0; }}
    QScrollBar::handle:vertical {{
        background: {border}; border-radius: 5px; min-height: 24px;
    }}
    QScrollBar::handle:vertical:hover {{ background: {accent}; }}
    QScrollBar:horizontal {{ background: {panel}; height: 11px; margin: 0; }}
    QScrollBar::handle:horizontal {{
        background: {border}; border-radius: 5px; min-width: 24px;
    }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}

    QStatusBar {{ background: {panel}; color: {text_dim}; }}
    QMenuBar {{ background: {panel}; }}
    QMenuBar::item:selected {{ background: {accent}; color: {accent_text}; }}
    QMenu {{ background: {surface}; border: 1px solid {border}; }}
    QMenu::item:selected {{ background: {accent}; color: {accent_text}; }}

    /* Check and radio indicators are drawn explicitly rather than left to the
       platform style. Once the application sets a palette-changing stylesheet,
       Windows keeps drawing its own indicators against the *system* colours --
       so on the dark theme a selected radio button was a dark dot on a dark
       ground and there was no way to tell which of Simulation / Parameter /
       ISO was active. These rules make the checked state a filled accent
       colour, which is unambiguous in both themes. */
    QCheckBox, QRadioButton {{ spacing: 6px; color: {text}; }}
    QCheckBox::indicator, QRadioButton::indicator {{
        width: 14px;
        height: 14px;
        border: 1px solid {border};
        background: {surface};
    }}
    QCheckBox::indicator {{ border-radius: 3px; }}
    QRadioButton::indicator {{ border-radius: 8px; }}
    QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
        border: 1px solid {accent};
    }}
    QCheckBox::indicator:checked {{
        background: {accent};
        border: 1px solid {accent};
    }}
    QRadioButton::indicator:checked {{
        border: 1px solid {accent};
        background: qradialgradient(cx:0.5, cy:0.5, radius:0.5,
                                    fx:0.5, fy:0.5,
                                    stop:0 {accent}, stop:0.55 {accent},
                                    stop:0.6 {surface}, stop:1 {surface});
    }}
    QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{
        border: 1px solid {border};
        background: {panel};
    }}
    QSplitter::handle {{ background: {border}; }}
    QSplitter::handle:hover {{ background: {accent}; }}
    QGroupBox {{
        border: 1px solid {border};
        border-radius: 3px;
        margin-top: 10px;
        padding-top: 8px;
    }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 8px; color: {text_dim}; }}
    """.format(**theme.__dict__)


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
