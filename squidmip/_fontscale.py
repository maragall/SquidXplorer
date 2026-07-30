"""Type that grows with its window, for EVERY window rather than just the root.

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

This module imports only qtpy on purpose, so both `_viewer` and `_region_viewer` can use it
without either importing the other: `_viewer` already imports `_region_viewer`, so keeping the
shared helper in `_viewer` would have been a circular import.
"""
from __future__ import annotations

import re

from qtpy.QtWidgets import QWidget

#: Width the type was authored against. Width-driven rather than height- or area-driven: these
#: windows grow mostly sideways, and tying type to height makes a tall thin window shout.
DESIGN_W = 1100

#: Below this, a resize is sub-pixel churn and restyling the whole tree is not worth it.
_SCALE_EPSILON = 0.02

_FONT_SIZE_RE = re.compile(r"(font-size\s*:\s*)(\d+(?:\.\d+)?)(px)", re.IGNORECASE)


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
