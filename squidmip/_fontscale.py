"""Display-derived window facts every window needs: the type scale, and WHICH SCREEN.

Spencer, 2026-07-27: "the root window rescales its type on resize (`501f71e`), but the
`[N] <well>` view windows and the Log window do not. They inherit the high-DPI fix, so they are
the right size, but dragging one bigger leaves the type behind."

The reason the root got this and the children did not is structural, not a missing `resizeEvent`.
`PlateWindow._rescale_fonts` walked `self.findChildren(QWidget)`, and a `RegionViewer` is a
separate TOP-LEVEL window: it is not in the plate's Qt child tree, so it was never in the walk.
Decentralising the viewer into independent windows is the whole architecture here, so the fix
belongs where any window can reach it rather than deeper inside the root.

**Why rewrite stylesheets instead of setting a widget font.** A QSS ``font-size`` on a child beats
any font inherited from a parent, and this GUI sets one inline at ~79 call sites. `setFont` would
be silently ignored at every one of them.

**Why the original is cached.** Scaling is always computed from the AUTHORED stylesheet, never
from the last scaled one. Without the cache, two resizes compound the multiplier and the type runs
away.

**Why `window_screen` lives here too.** It is the same shape of fact as the type scale: something
read off the DISPLAY that both the root window and every view window need in order to open at a
sensible size, and that neither can ask the other for. This module imports only qtpy on purpose, so
both `_viewer` and `_region_viewer` can use it without either importing the other: `_viewer` already
imports `_region_viewer`, so keeping the shared helper in `_viewer` would have been a circular
import.
"""
from __future__ import annotations

import re
from typing import Optional

from qtpy.QtWidgets import QWidget

#: Width the type was authored against. Width-driven rather than height- or area-driven: these
#: windows grow mostly sideways, and tying type to height makes a tall thin window shout.
DESIGN_W = 1100

#: Below this, a resize is sub-pixel churn and restyling the whole tree is not worth it.
_SCALE_EPSILON = 0.02

_FONT_SIZE_RE = re.compile(r"(font-size\s*:\s*)(\d+(?:\.\d+)?)(px)", re.IGNORECASE)


def window_screen(widget: Optional[QWidget]):
    """The screen *widget* is actually on, falling back to the primary screen.

    On a laptop + external monitor setup ``primaryScreen()`` is whichever display the OS calls
    first, NOT the one the user dragged the app onto. Sizing or placing a window against it gives
    a window measured for the wrong display: a view window sized to a third of the laptop panel
    while it opens on the 4K external, or vice versa. ``QWidget.screen()`` reports the display the
    widget's window occupies, so the numbers and the pixels are about the same monitor.

    Defensive on both counts: ``screen()`` only exists from Qt 5.14, and it returns None for a
    widget with no window handle yet. Either way the primary screen is still a better answer than
    no answer, so the fallback is the OLD behaviour rather than a crash.
    """
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
        # Floors at 8px so shrinking toward the minimum window size cannot produce type
        # nobody can read. This floor predates the move into this module; dropping it to a
        # rounding guard (max(1, ...)) is a legibility regression, caught by test_root_resize.
        return f"{m.group(1)}{max(8, int(round(float(m.group(2)) * scale)))}{m.group(3)}"

    return _FONT_SIZE_RE.sub(_sub, qss)


def ui_scale(window: QWidget, design_w: int = DESIGN_W) -> float:
    """How much bigger *window* is than the shape its type was written for.

    Clamped so a very wide window does not produce absurd type and a minimum-size one stays
    legible.
    """
    return max(0.85, min(2.0, window.width() / float(design_w)))


def rescale_fonts(window: QWidget, design_w: int = DESIGN_W) -> bool:
    """Re-apply every descendant stylesheet with its ``font-size`` multiplied by the scale.

    Returns True when a rescale actually happened, so callers and tests can tell "scaled" from
    "skipped as churn" without reading private state.

    State is stashed on the window itself (`_applied_ui_scale`, `_qss_original`) rather than in a
    module-level dict keyed by ``id()``. A module dict would outlive the windows and could hand a
    recycled id() someone else's cached stylesheet, which is a live hazard here because these
    windows are opened and closed constantly.
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
                cache[key] = None            # nothing to scale here; remember and skip it
                continue
            cache[key] = qss
        base = cache[key]
        if base is None:
            continue
        w.setStyleSheet(scale_qss_fonts(base, scale))
    return True
