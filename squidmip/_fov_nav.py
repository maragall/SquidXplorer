"""Walking a region's FOVs with the CAMERA. Loads nothing, places nothing.

WHY THIS IS NOT A REVERSAL OF ``_region_nav``'s FIRST PARAGRAPH
----------------------------------------------------------------
``squidmip/_region_nav.py`` opens with "THE NAVIGATION UNIT IS THE REGION. A region is a MOSAIC of
FOVs, never a single FOV." That is still true, and this module does not weaken it, because the
claim is about what is LOADED, PLACED and OPERATED ON -- and the three reasons given directly
beneath it are all about exactly that. Regions are ragged, so they cannot be one array. Each region
sits at its own stage position, and a layer has one ``translate``. A second ``Dims`` is cheaper
than an image axis. Every one of those is silent about a control that moves a camera.

The thing that paragraph killed is on the record and is a different object. ndviewer_light's FOV
slider WAS the plate navigator (``docs/DESIGN.md``: "the slider is the plate navigator. Moving the
FOV slider moves the red box"; ``docs/SCOPE.md`` records the whole ndv-era "FOV sliders, one FOV
per well" model as superseded). It decided which WELL was fetched, it called that well an FOV, and
it held a second copy of a selection the red box also held. It is dead and it stays dead.

A :class:`FovSlider` is none of those three:

* It lives in ONE window over ONE region whose mosaic is ALREADY RESIDENT. A step issues no read,
  builds no layer and cannot be superseded, so ``_load_gen``, ``_shown_region`` and the whole load
  pipeline are untouched by it. That is also why it opens its own playback gate the instant the
  camera has moved -- there is nothing to wait for. See ``RegionViewer._on_fov_changed``.
* It holds no copy of anything anyone else holds. ``RegionCursor`` exists because "which region"
  is WRITTEN from four places (the plate's double-click, ``show_region``'s adopt, the slider, a
  re-scope) and read by three more. "Which FOV is current" is written by one widget.
* It does not navigate the acquisition. ``window.address()`` still answers with a region, and
  ``_run_scope`` still returns regions: **a run's scope does not change when the camera moves.**

WHO OWNS THE INDEX, AND WHY IT IS NOT A CURSOR
-----------------------------------------------
napari's ``Dims`` does, exactly as it does for the time axis. ``_time_point.py`` settled this
question for this same widget: with ``playback=True`` "napari's dims widget IS the slider, so the
position has exactly ONE owner (napari's ``Dims``) rather than a QSlider and a Dims hand-synced".

A ``FovCursor`` mirroring ``RegionCursor`` was considered and rejected. ``RegionCursor`` earns its
existence by having several WRITERS; three readers is not the defect shape it was built against.
Adding one here would create a second owner beside ``Dims`` and force ``_follow``/``_echo``
round-tripping between them -- manufacturing precisely the hand-sync ``_region_nav`` exists to
remove. So ``Dims`` owns the index, this widget owns the ORDER, and the current FOV id is DERIVED
(``self._fovs[self.index]``) and never stored. ``RegionViewer`` keeps no ``_current_fov``.

The order is re-derived from ``meta`` per region rather than copied: ``meta`` owns which FOVs a
region has, and a widget holding its own answer to that is a stale answer waiting to happen.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence


class FovSlider:
    """Step the camera across a region's FOVs. napari's dims slider, its play button, its fps.

    Constructed lazily by :meth:`squidmip._region_viewer.RegionViewer._build` and ONLY in a FOVs
    view. A window that will never walk fields does not pay for a ``QtDims``, a ``QTimer`` and an
    ``AnimationThread`` it cannot use -- and "this window has no FOV axis" stays a question with
    an answer (``None``) rather than a do-nothing object, which is ``_time_point.playback``'s rule
    for the same shape of thing.

    This is a factory, not a base class: :class:`squidmip._region_nav.AxisPlayback` imports napari,
    and this module must stay importable without it so the geometry it drives can be tested light.
    """

    def __new__(cls, on_change: "Optional[Callable[[int, int], None]]" = None, parent=None):
        return _build_fov_slider(on_change=on_change, parent=parent)


def _build_fov_slider(*, on_change=None, parent=None):
    """The real class, built against napari's ``AxisPlayback`` at first use."""
    from squidmip._region_nav import AxisPlayback

    class _FovAxisPlayback(AxisPlayback):
        def __init__(self) -> None:
            self._fovs: list[int] = []
            self._on_change = on_change
            super().__init__(axis_label="fov", noun="FOV", parent=parent)

            from qtpy.QtWidgets import QLabel

            # The FOV ID, which napari's dims cannot know: its spin box shows the INDEX, and the
            # user navigates by the id the log lines and coordinates.csv use. Same split, and same
            # reason, as RegionSlider's region label.
            self._label = QLabel("")
            self._label.setMinimumWidth(150)
            self._label.setStyleSheet("color:#c9d1d9;font-size:12px;border:none;")
            self._row.addWidget(self._label)
            self.setToolTip(
                "Step the camera through this region's FOVs, one at a time, framed at native "
                "zoom.\nNothing reloads -- the mosaic is already on screen -- so stepping is "
                "instant.\nPress play to walk them; right-click play for frames per second and "
                "loop mode."
            )

        # -- the order, which is ours; the index, which is napari's ----------------------
        def set_fovs(self, fovs: "Sequence[int]") -> None:
            """Size the axis to a region's FOVs. Does NOT announce: this is a re-scope.

            A re-scope that lands you on a different field is not a navigation the user made, and
            firing ``on_change`` here would move the camera as a side effect of arriving in a
            region. The caller frames the first field explicitly, so the two are separable.
            """
            self._fovs = [int(f) for f in fovs]
            self.set_count(len(self._fovs))
            if self.index >= len(self._fovs):
                self._follow(0)
            self._refresh_label()

        @property
        def fovs(self) -> "list[int]":
            return list(self._fovs)

        @property
        def fov(self) -> "Optional[int]":
            """The CURRENT FOV id. Derived from the order and napari's index, never stored."""
            i = self.index
            if 0 <= i < len(self._fovs):
                return self._fovs[i]
            return None

        # -- moving ----------------------------------------------------------------------
        def _on_step(self, index: int) -> None:
            self._refresh_label()
            fov = self.fov
            if fov is not None and self._on_change is not None:
                self._on_change(int(index), int(fov))

        def _refresh_label(self) -> None:
            fov = self.fov
            if fov is None:
                self._label.setText("")
            else:
                self._label.setText(f"FOV {fov}   ({self.index + 1} of {len(self._fovs)})")

        def _refusal(self) -> "Optional[str]":
            """Refuse in the words of what is actually being walked, and count FOVs.

            The count comes from the order this widget was given, not from the dims range: a
            control that answered from its own widget state is the second copy ``_region_nav``
            exists to remove.
            """
            n = len(self._fovs)
            if n == 0:
                return "no FOVs here — there is nothing to step through."
            if n == 1:
                return "this region has one FOV; there is nothing to play through."
            return None

    return _FovAxisPlayback()
