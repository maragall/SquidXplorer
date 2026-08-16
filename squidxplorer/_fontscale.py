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


def default_root_width(avail_w: int, min_w: int, design_w: int) -> int:
    """How wide the root window opens on a work area *avail_w* logical px across.

    THE PLATE IS A NAVIGATOR, not a document: it picks the wells and the pixels live in the view
    window beside it. So the root takes about a fifth of the screen and leaves the rest, which is
    the shape people arrive with from other suites.

    Bounded at both ends, and both bounds bind on real machines:

      * ``min_w`` wins on a small screen. A fifth of a 1440-wide laptop is 288 px, and the band
        under the plate does not fit in that -- the navigator is already the column every pixel
        below the design width comes out of. So the split degrades to about 30/70 rather than
        20/80, which is correct: you cannot make a panel narrower than its controls.
      * ``design_w`` wins on a big screen. A fifth of a 4K work area at 100% is 768 px, which is
        wider than the shape this layout was drawn against and would be gutters either side of the
        plate rather than more plate. So the split becomes about 15/85, which is also correct.

    Pure arithmetic, deliberately: it is exercised across six screen shapes by tests that never
    build a window, because the offscreen platform reports one 800x600 screen and a literal there
    would pin nothing.
    """
    return max(int(min_w), min(int(design_w), int(avail_w) // 5))


def beside_rect(avail, anchor, min_w_frac: float = 1.0 / 3.0):
    """The rect for a window filling the work area *avail* to the RIGHT of *anchor*.

    Takes and returns ``QRect``. Aligned to the anchor's top and height rather than to the work
    area's, so the two windows read as one layout instead of two that happen to be adjacent, and
    so they stay in step if the user drags the root's height.

    Floored at *min_w_frac* of the work area: a root dragged nearly full width would otherwise
    leave a sliver, and a sliver with a napari canvas in it is worse than an overlap.
    """
    from qtpy.QtCore import QRect

    left = anchor.right() + 1
    width = max(avail.right() - anchor.right(), int(avail.width() * min_w_frac))
    left = min(left, max(avail.left(), avail.right() - width + 1))
    top = max(anchor.top(), avail.top())
    height = min(anchor.height(), avail.height())
    return QRect(left, top, width, height)


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
