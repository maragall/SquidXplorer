"""Display-derived window facts: the QSS type scale on resize, and which screen a window is on."""
from __future__ import annotations

import re
from typing import Optional

from qtpy.QtWidgets import QWidget

#: Width the type was authored against.
DESIGN_W = 1100

#: Below this, a resize is sub-pixel churn not worth restyling the tree for.
_SCALE_EPSILON = 0.02

_FONT_SIZE_RE = re.compile(r"(font-size\s*:\s*)(\d+(?:\.\d+)?)(px)", re.IGNORECASE)


def window_screen(widget: Optional[QWidget]):
    """The screen *widget* is actually on, falling back to the primary screen."""
    from qtpy.QtGui import QGuiApplication

    screen = None
    getter = getattr(widget, "screen", None) if widget is not None else None
    if callable(getter):
        try:
            screen = getter()
        except Exception:                    # noqa: BLE001 - a screen query is never worth a crash
            screen = None
    return screen if screen is not None else QGuiApplication.primaryScreen()


def scale_qss_fonts(qss: str, scale: float) -> str:
    """Multiply every ``font-size:Npx`` in *qss* by *scale*, leaving the rest untouched."""
    def _sub(m: "re.Match[str]") -> str:
        # Floor at 8px so shrinking cannot produce unreadable type.
        return f"{m.group(1)}{max(8, int(round(float(m.group(2)) * scale)))}{m.group(3)}"

    return _FONT_SIZE_RE.sub(_sub, qss)


def ui_scale(window: QWidget, design_w: int = DESIGN_W) -> float:
    """How much bigger *window* is than the shape its type was written for, clamped."""
    return max(0.85, min(2.0, window.width() / float(design_w)))


def rescale_fonts(window: QWidget, design_w: int = DESIGN_W) -> bool:
    """Re-apply every descendant stylesheet with its ``font-size`` scaled; True when it happened.

    Scaling always starts from the AUTHORED stylesheet, cached on the window itself, so two
    resizes never compound the multiplier.
    """
    scale = ui_scale(window, design_w)
    if abs(scale - getattr(window, "_applied_ui_scale", 0.0)) < _SCALE_EPSILON:
        return False
    window._applied_ui_scale = scale

    cache = window.__dict__.setdefault("_qss_original", {})
    for w in [window] + window.findChildren(QWidget):
        key = id(w)
        if key not in cache:
            qss = w.styleSheet()
            if "font-size:" not in qss:
                cache[key] = None
                continue
            cache[key] = qss
        base = cache[key]
        if base is None:
            continue
        w.setStyleSheet(scale_qss_fonts(base, scale))
    return True
