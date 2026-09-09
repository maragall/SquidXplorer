"""Stepping a decon run's captured iterations with a BOTTOM SLIDER.

Julio, 2026-09-09: "you should do a slider for the decon iterations that pops on the
bottom. Like the fov's button makes that slider." Same family as the FOV walk
(:mod:`squidxplorer._fov_nav`): napari's own dims slider row, built lazily, added to the
window's bottom bars. A step repaints HELD MIP planes (`_decon`'s capture store) into the
view's decon layers, so it issues no read and no solve, and the frame gate reopens
immediately: stepping is instant by construction.

Like ``FovSlider`` this is a factory, not a base class: ``AxisPlayback`` imports napari,
and this module stays importable without it.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence


class IterationSlider:
    """napari's dims slider over a decon run's captured iterations, with the turbo box.

    Constructed lazily by ``RegionViewer._refresh_iteration_slider`` the first time a run
    lands captures; a window that never runs decon does not pay for a ``QtDims``, a
    ``QTimer`` and an ``AnimationThread`` (the FOV axis's own rule).
    """

    def __new__(cls, on_change: "Optional[Callable[[int], None]]" = None,
                on_turbo: "Optional[Callable[[bool], None]]" = None, parent=None):
        return _build_iteration_slider(on_change=on_change, on_turbo=on_turbo, parent=parent)


def _build_iteration_slider(*, on_change=None, on_turbo=None, parent=None):
    """The real class, built against napari's ``AxisPlayback`` at first use."""
    from squidxplorer._region_nav import AxisPlayback

    class _IterAxisPlayback(AxisPlayback):
        def __init__(self) -> None:
            self._ks: list[int] = []
            self._on_change = on_change
            super().__init__(axis_label="iteration", noun="iteration", parent=parent)

            from qtpy.QtWidgets import QCheckBox, QLabel

            self._label = QLabel("")
            self._label.setMinimumWidth(110)
            self._label.setStyleSheet("color:#c9d1d9;font-size:12px;border:none;")
            self._row.addWidget(self._label)
            # Turbo rides the bar itself: the FOV bar already carries an extra row widget
            # (its id label), so the pattern accommodates one compact control.
            self.turbo_check = QCheckBox("turbo")
            self.turbo_check.setStyleSheet("color:#c9d1d9;font-size:12px;")
            self.turbo_check.setToolTip(
                "View the stepped decon layers under the turbo colormap, so intensity "
                "differences between iterations read clearly; untick to restore.")
            if on_turbo is not None:
                self.turbo_check.toggled.connect(lambda on: on_turbo(bool(on)))
            self._row.addWidget(self.turbo_check)
            self.setToolTip(
                "Step the decon result through its captured RL iterations.\nEvery "
                "iteration's MIP is already held in memory, so stepping repaints "
                "instantly; nothing is re-solved.")

        # -- the order, which is ours; the index, which is napari's ----------------------
        def set_iterations(self, ks: "Sequence[int]") -> None:
            """Size the axis to the captured counts. Does NOT announce: a landing is not a
            navigation. The position follows the FINAL iteration, which is what the run's
            own delivery put on screen."""
            self._ks = [int(k) for k in ks]
            self.set_count(len(self._ks))
            if self._ks:
                self._follow(len(self._ks) - 1)
            self._refresh_label()

        @property
        def iteration(self) -> "Optional[int]":
            """The CURRENT iteration count. Derived from the order and napari's index."""
            i = self.index
            if 0 <= i < len(self._ks):
                return self._ks[i]
            return None

        # -- moving ----------------------------------------------------------------------
        def _on_step(self, index: int) -> None:
            self._refresh_label()
            k = self.iteration
            if k is not None and self._on_change is not None:
                self._on_change(int(k))
            # A repaint of held planes: there is nothing to wait for, so the playback gate
            # reopens at once (the FOV walk waits for a camera frame; this axis need not).
            self.frame_done()

        def _refresh_label(self) -> None:
            k = self.iteration
            if k is None:
                self._label.setText("")
            else:
                self._label.setText(f"iteration {k}/{self._ks[-1]}")

        def _refusal(self) -> "Optional[str]":
            n = len(self._ks)
            if n == 0:
                return "no captured iterations here - run a decon preview first."
            if n == 1:
                return "one captured iteration; there is nothing to play through."
            return None

    return _IterAxisPlayback()
