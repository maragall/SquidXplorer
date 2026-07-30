"""The plate says on its face which recipes made it.

Task 3 of the GUI backlog plan (2026-07-29): combine two runs in one plate view. The workflow is
ordinary -- dial parameters on A1 alone, run the other 95 with what you learned, look at the whole
plate -- and Julio's decision was PER-CELL IDENTITY: a mixed-recipe plate is LEGAL. Nothing here
prevents one, warns about one, or asks whether two cells agree.

WHY A LEGEND AND NOT A WARNING
------------------------------
An earlier draft of Task 3 proposed detecting a mixed plate and warning about it. Julio banned it:
that is disclosure bolted onto a painter, and the next mismatch (z depth, pixel size, dtype) would
need its own bolt. So this widget compares nothing. It renders a
:class:`squidmip._recipe.PlateCensus`, which is a GROUPING of what the cache holds, and every
channel set in it was read from a cell's own :class:`squidmip._result.Substance`.

WHY IT IS NOT OPTIONAL, AND WHY THE TEST ASSERTS THE WIDGET
-----------------------------------------------------------
**A plate showing more than one recipe says so ON ITS FACE, not in a tooltip.** Earned expensively
on 2026-07-28: a tooltip promised a "3D view (AGAVE)..." button the app did not have, and a PASSING
test held that phantom in place for weeks, because the test pinned the STRING and not the button.
So the visibility rule here is one line -- :attr:`~squidmip._recipe.PlateCensus.is_mixed` and
nothing else -- and ``tests/test_plate_census.py`` asserts it through ``isVisible()`` on the real
widget in a shown window, never by matching text.

THE MARK IS A BORDER, NOT A WASH
--------------------------------
The plate's washes are spoken for twice over: a per-view hue says which window owns a well, and a
blue wash says which wells are selected. A third colour meaning would collide with two that are
already load-bearing, so a diverging cell gets a dashed BORDER
(:data:`squidmip._qtstyle.DIVERGENT`). This widget draws a swatch in the same ink beside the
diverging chain's row, so the mark on the plate and the row in the legend are one statement rather
than two things the reader has to connect.

WHAT A ROW SAYS, AND WHAT IT NEVER SAYS
---------------------------------------
Three things, exactly as the plan specifies: a HUMAN label from the recipes
(:meth:`squidmip._recipe.RecipeChain.label`, ``mip + decon(sigma=2.0)``), the number of cells, and
the channel set those cells declare. Never the chain's hash. A hash cannot be un-hashed into
recipes, which is why :class:`squidmip._recipe.Entry` keeps the chain OBJECT beside the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from squidmip import _qtstyle

__all__ = ["LegendRow", "RecipeLegend"]


@dataclass(frozen=True)
class LegendRow:
    """One row's DATA, so a test can read what the legend says without matching a rendered string.

    A test that pins a string is how the phantom AGAVE button survived: the string was right and the
    thing behind it did not exist. These are the values the row was built from, exposed by
    :meth:`RecipeLegend.rows`, and the widgets are built from exactly them.
    """

    label: str                  # "mip + decon(sigma=2.0)" -- from the recipes, never a hash
    count: int                  # cells this chain produced
    channels: "tuple" = ()      # what those cells DECLARE (read from their Substance)
    marked: bool = False        # this chain's cells carry the border mark on the plate

    def text(self) -> str:
        return f"{self.label}   {self.count} cell(s)   {','.join(self.channels)}"


class RecipeLegend(QWidget):
    """The plate's legend: one row per recipe present, visible only when more than one is.

    Visibility is the whole guarantee, so it has exactly one input and one rule::

        legend.set_census(census)      # visible <=> census.is_mixed

    There is no ``show_legend`` flag, no user toggle and no menu item, on purpose. A plate showing
    two recipes has to disclose it, and a control that can hide the disclosure is the same defect as
    burying it in a tooltip.
    """

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self._rows: "tuple" = ()
        self.setStyleSheet("background:#0b0e14;border-top:1px solid #232b3a;")
        self._l = QVBoxLayout(self)
        self._l.setContentsMargins(12, 6, 12, 6)
        self._l.setSpacing(3)
        self._head = QLabel("")
        self._head.setStyleSheet("color:#8b98ad;font-size:11px;font-weight:700;border:none;")
        self._l.addWidget(self._head)
        self._body = QVBoxLayout()          # the rows, rebuilt per census
        self._body.setContentsMargins(0, 0, 0, 0)
        self._body.setSpacing(2)
        self._l.addLayout(self._body)
        self.hide()                          # a one-recipe plate has nothing to disclose

    # -- what it is showing ------------------------------------------------------------------
    def rows(self) -> "tuple":
        """The :class:`LegendRow` values currently rendered, in the census's order."""
        return self._rows

    def summary(self) -> str:
        """The header line, e.g. ``2 recipes on this plate``. Empty when nothing is shown."""
        return self._head.text()

    # -- the one input -----------------------------------------------------------------------
    def set_census(self, census: Any) -> None:
        """Render *census* and become visible if and only if it holds more than one chain.

        *census* is a :class:`squidmip._recipe.PlateCensus`, or ``None`` for "no plate", which is
        the same thing on screen as a single-recipe plate: nothing.
        """
        groups = tuple(getattr(census, "groups", ()) or ())
        mixed = bool(getattr(census, "is_mixed", False))
        plurality_key = getattr(getattr(census, "plurality", None), "key", None)

        self._rows = tuple(
            LegendRow(label=g.label(), count=g.count, channels=tuple(g.channels),
                      marked=(plurality_key is not None and g.key != plurality_key))
            for g in groups)
        self._rebuild()
        self._head.setText(f"{len(self._rows)} recipes on this plate" if mixed else "")
        self.setVisible(mixed)

    def clear(self) -> None:
        """Forget everything and hide. What opening another acquisition does."""
        self.set_census(None)

    # -- rendering ---------------------------------------------------------------------------
    def _rebuild(self) -> None:
        """Drop the old row widgets and build one per :class:`LegendRow`.

        Rebuilt rather than updated in place: a census can gain and lose chains, and a widget pool
        that has to be reconciled against a changing list is a second place for the count to be
        wrong. The list is one row per recipe on a plate, so it is short by construction.
        """
        while self._body.count():
            item = self._body.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        for row in self._rows:
            self._body.addWidget(self._row_widget(row))

    def _row_widget(self, row: LegendRow) -> QWidget:
        """``[swatch] mip + decon(sigma=2.0)   48 cell(s)   DAPI,GFP``.

        The swatch is the SAME ink as the border the plate draws around this chain's cells, and it
        is an outline rather than a filled chip for the same reason the plate's mark is: a filled
        colour would read as a third wash meaning.
        """
        host = QWidget()
        h = QHBoxLayout(host)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)

        swatch = QFrame()
        swatch.setFixedSize(12, 12)
        if row.marked:
            swatch.setStyleSheet(
                f"background:transparent;border:1px dashed {_qtstyle.DIVERGENT_CSS};")
            swatch.setToolTip("these cells carry the dashed border on the plate")
        else:
            swatch.setStyleSheet("background:transparent;border:none;")
        h.addWidget(swatch, 0, Qt.AlignVCenter)

        name = QLabel(row.label)
        name.setStyleSheet("color:#e6edf3;font-size:11px;font-weight:700;border:none;")
        h.addWidget(name, 0, Qt.AlignVCenter)

        count = QLabel(f"{row.count} cell(s)")
        count.setStyleSheet("color:#c9d1d9;font-size:11px;border:none;")
        h.addWidget(count, 0, Qt.AlignVCenter)

        channels = QLabel(",".join(row.channels))
        channels.setStyleSheet("color:#8b98ad;font-size:11px;border:none;")
        channels.setToolTip("the channels these cells declare")
        h.addWidget(channels, 1, Qt.AlignVCenter)
        return host
