"""The two operator interfaces in PANE 1: the stitcher's controls and deconvolution's QC loop.

Julio: "Right now I'm blocked in testing the post-processing because Stitcher doesn't have
that maragall/Stitcher interface embedded in our top-left subpane. The deconvolution is not
showing the XZ/YZ strips on the turbo colormap on the exploration pane so that we can choose
the iterations."

WHERE THINGS LIVE, AND WHY
--------------------------
* Every CONTROL is in pane 1. Pane 3 gets no operator launcher. A UI audit found two
  operator registries (``_OPERATIONS`` and ``runnable_operators()``) launching the same
  operators from panes 1 and 3 with different labels and different ``save`` defaults, and a
  comment in ``_viewer.py`` records that they diverged in production. So "run this on the
  subset parked in pane 3" is a SCOPE VALUE on the pane-1 panel (:func:`scope_options`),
  not a second set of buttons. This module adds no third caller to either registry.
* The deconvolution RESULT - the 2-D image in turbo with the x-z and y-z strips concatenated
  to it - is a TAB IN PANE 3 (:class:`DeconQCResultView`). That is where a preview result is
  looked at, and it is big: it needs the room.

THE SEAM WITH PANE 3, STATED NARROWLY
-------------------------------------
This module never touches pane 3's tab bar. It calls exactly one method on its host::

    host.publish_qc_result(widget, title)   # -> pane 3 shows `widget` as a tab called `title`

``PlateWindow`` implements that with its EXISTING ``_open_op_tab(key, title, builder,
tabs=self._explore_tabs)`` - the same mechanism exploration tabs already use, so no new tab
API is introduced and the pane-3 owner has nothing to merge. If a host does not implement
it, the panel SAYS SO in the readout; the picture is never computed and then dropped.

WHAT WAS PORTED FROM maragall/stitcher, AND WHAT WAS NOT
--------------------------------------------------------
Ported (its ``Settings`` group, ``gui/app.py``):

* "Enable registration refinement" -> ``register=``, which is also the ``stitch`` vs
  ``coordinate`` operator choice already in ``_stitch.py``.
* Registration channel -> ``registration_channel=``.
* "Blend pixels" -> ``blend_px=``.
* "Outlier rel: N%" / "abs: N px" -> ``rel_thresh=`` / ``abs_thresh=``. These were module
  constants; IMA-decon-stitch-ui threads them through ``solve_offsets_px``.
* Run button + progress + a log of per-region results.

NOT ported, each for a reason rather than for lack of time:

* **Downsample factor.** ``_DOWNSAMPLE_FACTORS = (1, 1)`` is PINNED here with a stated
  reason ("registration MUST be full-res; any downsample coarsens the sub-pixel shift").
  Exposing a knob whose whole effect is to make the answer worse, on a post-acquisition
  tool where the run is not interactive, is not a control - it is a trap.
* **Registration z-level / timepoint.** In the stitcher those pick a plane out of a stack.
  Here the projector has already reduced z before registration runs, and geometry is solved
  once on t=0 by construction. There is no plane to pick.
* **Flatfield group.** squidmip has ``_flatfield.py`` as its own operator with its own
  profile chooser. A second flat-field UI here would be a second owner of one setting.
* **"Auto" blend width** (stitcher computes ~2x the seam overlap). squidmip has no overlap
  measurement of its own - ``tilefusion.find_adjacent_pairs`` owns that geometry internally
  and does not report it back. Deriving a second estimate here would be an unvalidated
  second representation of the same number. The default is the measured ``_BLEND_PX`` and
  the tooltip states the real overlap it was sized against.
* **Lens-distortion correction.** Not in the ``tilefusion`` call chain ``_stitch.py`` ports.
  A checkbox for a stage that does not run would be a lie.
* **Preview grid (N x M), drag-and-drop, "Open in Napari", "Export OME-TIFF", "Open
  Existing", "Max Projection".** The stitcher is a standalone app that has to load a
  dataset and then hand its result to a viewer. This IS the viewer: the plate is already
  open, and the fused mosaic lands in the embedded napari by itself.

FROM maragall/deconvolution (``petakit``'s ``gui/main.py``)
----------------------------------------------------------
Ported: the iterations spinner with a "recommended: N" hint beside it, the "Force CPU"
checkbox, the channel selector, and a status line that shows the reason on failure.

Not ported: the **method combo** (petakit offers ``omw``/``rl``; ``_decon.py`` PINS ``rl``
because ``omw`` returns an all-zero volume on this instrument's geometry - offering it would
be offering a black image), the **output directory + OME-TIFF save** (this is a viewer, not
an exporter), the **drag-and-drop acquisition box** (the plate window owns dataset loading),
and the **"Preview (5 FOVs)" ComparisonWindow** (a second top-level window with two more
viewers in it, in an app whose whole point is bundling three panes into one).

Added, because petakit's GUI is fire-and-forget and Julio's loop is not: **"+1 iteration"**,
and the halo/core number with :func:`squidmip._decon_qc.halo_verdict`'s sentence next to the
picture, so "are the light halos handled" has a number beside the eye.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from squidmip._stitch import _ABS_THRESH, _BLEND_PX, _REL_THRESH

#: The panel's starting position. Every value here is the pipeline's own default, so an
#: untouched panel launches byte-for-byte what ``stitch_region`` does unaided - a panel with
#: opinions of its own would be a second set of defaults to keep in step.
STITCH_DEFAULTS = {
    "register": True,
    "registration_channel": None,
    "channels": None,
    "blend_px": _BLEND_PX,
    "outlier_rel_pct": int(round(_REL_THRESH * 100)),
    "outlier_abs_px": int(round(_ABS_THRESH)),
}


# ---------------------------------------------------------------------------------------
# policy (no Qt) — the decisions, separated from the pixels
# ---------------------------------------------------------------------------------------

def scope_options(order, selected, explore_scopes):
    """``[(label, regions_or_None), ...]`` for the panel's scope selector.

    ``regions=None`` is ``run_operator``'s "the whole plate", so it is offered first and
    always. A plate selection and each subset parked in pane 3 follow, as VALUES rather
    than as separate launchers.

    Region lists are COPIED: the selector holds a scope for as long as the panel is open,
    and a later click on the plate must not retroactively change what a pending run covers.
    An empty subset is not offered at all - choosing it would only produce "empty selection
    - nothing to run" at the far end of the click.
    """
    order = list(order)
    options = [(f"Whole dataset — {len(order)} well" + ("s" if len(order) != 1 else ""), None)]
    selected = list(selected or [])
    if selected:
        options.append((f"Selected wells — {len(selected)}", selected))
    for label, regions in (explore_scopes or []):
        regions = list(regions or [])
        if regions:
            options.append((f"Pane 3 subset · {label} — {len(regions)}", regions))
    return options


def stitch_operator_kwargs(*, register, registration_channel, channels, blend_px,
                           outlier_rel_pct, outlier_abs_px,
                           n_channels: Optional[int] = None,
                           tile_px: Optional[int] = None) -> dict:
    """Turn the panel's widget values into ``stitch_region`` keyword arguments.

    Every key returned is a real parameter of :func:`squidmip._stitch.stitch_region` - a
    test asserts that against its signature, because a typo'd key raises ``TypeError``
    inside a worker thread where the only symptom is a status line that stops updating.

    The conversions that must happen exactly once, and here:

    * ``outlier_rel_pct`` is a PERCENTAGE in the UI (maragall/stitcher shows "50%") and a
      FRACTION in ``two_round_optimization``. Handing 50 straight through would reject
      nothing while the control looked like it worked.
    * With ``register=False`` there is no pose graph, so the blunder thresholds and the
      registration channel are DROPPED rather than passed and ignored.
    * "every channel" is spelled ``None``, not a full index list - that is the spelling
      ``stitch_region``'s docstring (and its memory note) is written against.
    """
    if channels is not None:
        channels = [int(c) for c in channels]
        if not channels:
            raise ValueError(
                "select at least one channel to fuse: a mosaic with no channels is not a "
                "smaller result, it is no result.")
        if n_channels is not None and len(channels) == int(n_channels):
            channels = None
    blend_px = int(blend_px)
    if tile_px is not None and blend_px >= int(tile_px):
        raise ValueError(
            f"blend width {blend_px} px is not smaller than the {int(tile_px)} px tile. The "
            "Hann feather has to fit INSIDE the real overlap; a ramp that never reaches full "
            "weight dims every seam, which looks like a stitching artefact rather than a "
            "setting.")

    kwargs = {"register": bool(register), "channels": channels, "blend_px": blend_px}
    if register:
        kwargs["registration_channel"] = registration_channel
        kwargs["rel_thresh"] = float(outlier_rel_pct) / 100.0
        kwargs["abs_thresh"] = float(outlier_abs_px)
    return kwargs


def plane_op_refusal(projector: str) -> Optional[str]:
    """The sentence explaining why *projector* cannot be stitched, or ``None`` if it can.

    ``stitch_region`` already refuses a plane-op (IMA-277: fusing one would keep z-plane 0
    and silently discard the rest of the stack). This asks the SAME registry the same
    question before the run starts, because discovering it at the end of a multi-minute
    fuse is a bad way to learn it. It is a pre-check, not a second guard - the operator's
    own refusal stays exactly where it is.
    """
    from squidmip._stitch import _resolve_projector

    try:
        op = _resolve_projector(projector)
    except Exception as exc:                       # unknown name -> name it, don't crash
        return (f"{projector!r} is not a projector this build knows: {exc}")
    if op.consumes:
        return None
    return (
        f"{projector!r} is a plane-op: it maps plane -> plane and keeps the z-stack at full "
        f"depth. Stitching does not fuse per z-plane yet, so it would keep z-plane 0 and "
        f"silently discard the rest of the stack. Reduce z first (mip), or pick a "
        f"z-reducing operator such as decon3d.")


# ---------------------------------------------------------------------------------------
# shared chrome
# ---------------------------------------------------------------------------------------

_BG = "#0d1117"
_SUB = "color:#8b98ad;font-size:11px;"
_HEAD = "color:#57606a;font-size:10px;font-weight:800;letter-spacing:1.5px;padding-top:6px;"


def _qss():
    """The window's OWN button/combo/checkbox styles, imported lazily.

    Not a second dark theme: `_viewer` already owns these three strings and every other
    control in pane 1 is drawn with them. Screenshotting the first build is what caught
    this -- unstyled QPushButtons render as flat text on this background and do not read as
    clickable at all. Lazy so this module stays importable without pulling in the 6k-line
    viewer, and so a Qt-free test of the policy functions above costs nothing.
    """
    from squidmip._viewer import _BTN_QSS, _CHECK_QSS, _COMBO_QSS

    return _BTN_QSS, _COMBO_QSS, _CHECK_QSS


def _apply_qss(root: QWidget) -> None:
    """Style every control in *root* the way the rest of pane 1 is styled."""
    btn, combo, check = _qss()
    for w in root.findChildren(QPushButton):
        w.setStyleSheet(btn)
        w.setCursor(Qt.PointingHandCursor)
    for w in root.findChildren(QComboBox):
        w.setStyleSheet(combo)
    for w in root.findChildren(QSpinBox):
        w.setStyleSheet(combo)
    for w in root.findChildren(QCheckBox):
        w.setStyleSheet(check)


def _wrapped(text: str, style: str) -> QLabel:
    """A word-wrapped QLabel that actually RESERVES the height its wrapping needs.

    A plain word-wrapped QLabel reports a single-line sizeHint to the layout, so the
    paragraph paints over whatever sits under it. In the first build the deconvolution
    blurb's third line printed on top of the "WHERE TO MEASURE" header -- visible in the
    screenshot, invisible in the source.
    """
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setStyleSheet(style)
    lab.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
    lab.setMinimumHeight(lab.fontMetrics().height() * 2)
    return lab


def _head(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setStyleSheet(_HEAD)
    return lab


def _row(*widgets) -> QHBoxLayout:
    lay = QHBoxLayout()
    lay.setSpacing(6)
    for w in widgets:
        lay.addWidget(w) if isinstance(w, QWidget) else lay.addLayout(w)
    lay.addStretch(1)
    return lay


def _channel_names(host) -> list:
    meta = getattr(host, "_meta", None) or {}
    return [c["name"] for c in meta.get("channels", [])]


class _Panel(QWidget):
    """Common shell: a title, a blurb, and a status line that is never silent."""

    def __init__(self, host, title: str, blurb: str):
        super().__init__()
        self.host = host
        self.setStyleSheet(f"background:{_BG};color:#e6edf3;")

        # A SCROLL AREA, because these are tall control stacks in a narrow pane. The
        # stitcher's alone is scope + z-reduction + registration + fusion + channels + run,
        # and pane 1 also has to hold the plate view. Without this the bottom controls are
        # simply unreachable at ordinary window heights -- and "the canvas squeezed to a
        # 140 px sliver" is the precedent for trusting a screenshot over a layout argument.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        body = QWidget()
        self.v = QVBoxLayout(body)
        self.v.setContentsMargins(16, 14, 16, 14)
        self.v.setSpacing(8)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea{border:none;background:transparent;}")
        scroll.setWidget(body)
        outer.addWidget(scroll)

        t = QLabel(title)
        t.setStyleSheet("font-size:16px;font-weight:800;")
        self.v.addWidget(t)
        self.v.addWidget(_wrapped(blurb, "color:#8b98ad;font-size:12px;"))
        self.status = _wrapped("", "color:#d29922;font-size:11px;")

    def say(self, text: str) -> None:
        """Put a SENTENCE in front of the user. Never log-and-continue.

        Routed to the window's readout as well as the panel's own line, because the panel
        may not be the visible tab when the thing worth saying happens.
        """
        self.status.setText(text)
        say = getattr(self.host, "say", None)
        if callable(say):
            say(text)


# ---------------------------------------------------------------------------------------
# 1. the stitcher
# ---------------------------------------------------------------------------------------

class StitcherPanel(_Panel):
    """maragall/stitcher's control surface over squidmip's ``stitch`` region operator."""

    def __init__(self, host):
        super().__init__(
            host, "Stitch (register + fuse)",
            "Register every FOV of a region against its neighbours and fuse one seamless "
            "mosaic. A region is a MOSAIC of FOVs, so one region produces one image.")
        names = _channel_names(host)

        # -- what to run it on ---------------------------------------------------------
        self.v.addWidget(_head("SCOPE"))
        self.scope_combo = QComboBox()
        self._scopes = scope_options(getattr(host, "_order", []),
                                     getattr(host, "_selected_regions", []),
                                     host.explore_scopes() if hasattr(host, "explore_scopes") else [])
        for label, _regions in self._scopes:
            self.scope_combo.addItem(label)
        self.scope_combo.setToolTip(
            "Every operator control lives here in pane 1. 'Run it on the subset parked in "
            "pane 3' is one of these values, not a second button over there.")
        self.v.addLayout(_row(self.scope_combo))

        # -- what to reduce z with before fusing ---------------------------------------
        self.v.addWidget(_head("Z REDUCTION"))
        self.projector_combo = QComboBox()
        from squidmip import available_projectors

        for name in sorted(available_projectors()):
            self.projector_combo.addItem(name)
        self.projector_combo.setCurrentText("mip")
        self.projector_combo.setToolTip(
            "Each FOV's z-stack is reduced to one plane before registration. Only a "
            "z-REDUCER can be stitched; a plane-op is refused with a reason.")
        self.projector_combo.currentTextChanged.connect(self._check_projector)
        self.v.addLayout(_row(QLabel("Reduce z with:"), self.projector_combo))

        # -- registration --------------------------------------------------------------
        self.v.addWidget(_head("REGISTRATION"))
        self.register_cb = QCheckBox("Enable registration refinement")
        self.register_cb.setChecked(STITCH_DEFAULTS["register"])
        self.register_cb.setToolTip(
            "On: phase-correlate overlapping pairs and solve a global pose graph.\n"
            "Off: pure coordinate placement from the stage — the honest control for judging "
            "whether registration actually helped.")
        self.register_cb.toggled.connect(self._on_register_toggled)
        self.v.addWidget(self.register_cb)

        self.reg_channel_combo = QComboBox()
        for name in names:
            self.reg_channel_combo.addItem(name)
        self.reg_channel_combo.setToolTip(
            "ONE channel drives the geometry and every channel is then fused with that one "
            "solution — channels of a FOV share a sensor and must not get disagreeing "
            "placements.")
        self.v.addLayout(_row(QLabel("Registration channel:"), self.reg_channel_combo))

        self.rel_spin = QSpinBox()
        self.rel_spin.setRange(1, 200)
        self.rel_spin.setValue(STITCH_DEFAULTS["outlier_rel_pct"])
        self.rel_spin.setSuffix("%")
        self.rel_spin.setToolTip(
            "Blunder rejection, relative term: drop a link whose residual exceeds this "
            "percentage of the median residual.")
        self.abs_spin = QSpinBox()
        self.abs_spin.setRange(1, 50)
        self.abs_spin.setValue(STITCH_DEFAULTS["outlier_abs_px"])
        self.abs_spin.setSuffix(" px")
        self.abs_spin.setToolTip(
            "Blunder rejection, absolute term: a link must ALSO be off by at least this "
            "many pixels to be dropped. Both conditions have to hold, so a very clean plate "
            "does not start rejecting links that were off by a fraction of a pixel.")
        self.v.addLayout(_row(QLabel("Outlier rel:"), self.rel_spin,
                              QLabel("abs:"), self.abs_spin))

        # -- fusion --------------------------------------------------------------------
        self.v.addWidget(_head("FUSION"))
        self.blend_spin = QSpinBox()
        self.blend_spin.setRange(1, 2000)
        self.blend_spin.setValue(STITCH_DEFAULTS["blend_px"])
        self.blend_spin.setSuffix(" px")
        self.blend_spin.setToolTip(
            "Hann feather ramp width. It must fit INSIDE the real overlap: on the 10x tissue "
            "set the measured overlap is ~208 px, which is what the 128 px default was sized "
            "against. A ramp wider than the overlap never reaches full weight and dims the "
            "seam.")
        self.v.addLayout(_row(QLabel("Blend width:"), self.blend_spin))

        self.channel_boxes = []
        if names:
            self.v.addWidget(_head("CHANNELS TO FUSE"))
            self.v.addWidget(_wrapped(
                "Every channel is fused with the ONE geometry solved above. This is the "
                "memory lever: a 27-FOV 10x region is ~0.2 GB at one channel and ~0.9 GB at "
                "four.", _SUB))
            box_row = QHBoxLayout()
            box_row.setSpacing(8)
            for name in names:
                cb = QCheckBox(name)
                cb.setChecked(True)
                self.channel_boxes.append(cb)
                box_row.addWidget(cb)
            box_row.addStretch(1)
            self.v.addLayout(box_row)

        # -- run -----------------------------------------------------------------------
        self.run_btn = QPushButton("Run stitcher iteration")
        self.run_btn.setCursor(Qt.PointingHandCursor)
        self.run_btn.setToolTip(
            "Register and fuse the selected scope and show the result. Nothing is written "
            "to disk and the acquisition is never modified.")
        self.run_btn.clicked.connect(self._run)
        self.v.addWidget(self.run_btn)

        self.save_cb = QCheckBox("Also write the fused mosaics to disk (OME-Zarr)")
        self.save_cb.setToolTip(
            "Off by default: tuning a registration/fusion run should cost compute, not disk. "
            "The settings above travel to the saved run too.")
        self.v.addWidget(self.save_cb)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setMaximumHeight(6)
        self.progress.setVisible(False)
        self.v.addWidget(self.progress)
        self.v.addWidget(self.status)
        self.v.addStretch(1)

        _apply_qss(self)
        self._on_register_toggled(self.register_cb.isChecked())
        self._check_projector(self.projector_combo.currentText())

    # -- behaviour ---------------------------------------------------------------------
    def _on_register_toggled(self, on: bool) -> None:
        """Grey out the knobs that provably do nothing with registration off."""
        for w in (self.reg_channel_combo, self.rel_spin, self.abs_spin):
            w.setEnabled(bool(on))

    def _check_projector(self, name: str) -> None:
        why = plane_op_refusal(name)
        self.run_btn.setEnabled(why is None)
        self.say("" if why is None else why)

    def kwargs(self) -> dict:
        """The panel's settings as ``stitch_region`` keyword arguments."""
        meta = getattr(self.host, "_meta", None) or {}
        frame = meta.get("frame_shape") or (None, None)
        selected = [i for i, cb in enumerate(self.channel_boxes) if cb.isChecked()]
        return stitch_operator_kwargs(
            register=self.register_cb.isChecked(),
            registration_channel=(self.reg_channel_combo.currentIndex()
                                  if self.register_cb.isChecked() else None),
            channels=selected if self.channel_boxes else None,
            blend_px=self.blend_spin.value(),
            outlier_rel_pct=self.rel_spin.value(),
            outlier_abs_px=self.abs_spin.value(),
            n_channels=len(self.channel_boxes) or None,
            tile_px=min(frame) if all(frame) else None,
        )

    def _run(self) -> None:
        why = plane_op_refusal(self.projector_combo.currentText())
        if why is not None:
            self.say(why)
            return
        try:
            kwargs = self.kwargs()
        except ValueError as exc:                 # a refused setting -> say it, run nothing
            self.say(str(exc))
            return
        kwargs["projector"] = self.projector_combo.currentText()
        _label, regions = self._scopes[max(self.scope_combo.currentIndex(), 0)]
        self.say("")
        self.host.run_operator("stitch", regions=regions,
                               save=self.save_cb.isChecked(), operator_kwargs=kwargs)


# ---------------------------------------------------------------------------------------
# 2. deconvolution — the iterative QC loop
# ---------------------------------------------------------------------------------------

class _DeconQCWorker(QThread):
    """Run RL at ONE iteration count on ONE FOV's z-stack and measure the halo.

    A thread because RL on a 256-px crop is seconds, not milliseconds, and a frozen window
    during a QC loop is the reason nobody runs the QC loop.
    """

    done = pyqtSignal(int, object, float)        # (iterations, composite, halo/core ratio)
    failed = pyqtSignal(str)

    def __init__(self, dataset, region, fov, channel, iterations, gpu, crop_half, view_half):
        super().__init__()
        self._args = (dataset, region, fov, channel, iterations, gpu, crop_half, view_half)

    def run(self):
        dataset, region, fov, channel, iterations, gpu, crop_half, view_half = self._args
        try:
            from squidmip._decon import OpticsParams, _run, make_psf
            from squidmip._decon_qc import (
                brightest_structure,
                crop_around,
                halo_core_ratio,
                load_stack,
                qc_composite,
                qc_window_um,
            )

            stack, region, channel, _meta = load_stack(dataset, region, fov, channel)
            optics = OpticsParams.from_acquisition(dataset, channel=channel)
            optics = OpticsParams(optics.na, optics.wavelength_um, optics.dxy_um,
                                  optics.dz_um, int(stack.shape[0]), optics.ni)
            core_um = 0.61 * optics.wavelength_um / optics.na
            window_um = qc_window_um(core_um, stack.shape[0], optics.dz_um)
            z_margin = int(np.ceil(window_um / optics.dz_um))
            centre_full = brightest_structure(stack, optics.dxy_um, optics.dz_um, core_um,
                                              z_margin=z_margin, xy_margin=crop_half)
            crop, centre = crop_around(stack, centre_full, crop_half)
            volume = _run(crop, make_psf(optics), int(iterations), gpu=gpu)
            ratio = halo_core_ratio(volume, centre, optics.dxy_um, optics.dz_um,
                                    core_um, window_um)
            self.done.emit(int(iterations),
                           qc_composite(volume, centre, view_half=view_half), float(ratio))
        except Exception as exc:                  # reported as a sentence, never swallowed
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class DeconQCResultView(QWidget):
    """PANE 3. The deconvolved 2-D image in turbo with the x-z and y-z strips attached.

    It RENDERS what :func:`squidmip._decon_qc.qc_composite` and
    :func:`squidmip._decon_qc.turbo_rgb` produced; it builds no picture of its own. A view
    that assembled three panels itself would be a second renderer to keep in step with the
    CLI montage, which is this project's dominant defect shape.
    """

    def __init__(self, subject: str):
        super().__init__()
        self.setStyleSheet(f"background:{_BG};color:#e6edf3;")
        self.history: list = []
        v = QVBoxLayout(self)
        v.setContentsMargins(14, 12, 14, 12)
        v.setSpacing(8)
        t = QLabel(f"Deconvolution QC · {subject}")
        t.setStyleSheet("font-size:15px;font-weight:800;")
        v.addWidget(t)
        legend = QLabel("x-y with the y-z strip to its right and the x-z strip below, all "
                        "TURBO on one shared scale. Turbo has a steep ramp through the low "
                        "intensities where a halo lives; on a grey ramp the halo is the part "
                        "of the image the eye is worst at.")
        legend.setWordWrap(True)
        legend.setStyleSheet(_SUB)
        v.addWidget(legend)

        self.caption_label = QLabel("")
        self.caption_label.setWordWrap(True)
        self.caption_label.setStyleSheet("font-size:12px;font-weight:700;")
        v.addWidget(self.caption_label)

        self.verdict_label = QLabel("")
        self.verdict_label.setWordWrap(True)
        self.verdict_label.setStyleSheet("color:#d29922;font-size:11px;")
        v.addWidget(self.verdict_label)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea{border:none;background:transparent;}")
        scroll.setWidget(self.image_label)
        v.addWidget(scroll, 1)

        self.trail_label = QLabel("")
        self.trail_label.setWordWrap(True)
        self.trail_label.setStyleSheet(_SUB)
        v.addWidget(self.trail_label)

    def show_iteration(self, iterations: int, composite, ratio: float,
                       kind: str, verdict: str) -> None:
        """Display one iteration's composite and remember it, so the loop can be compared."""
        from squidmip._decon_qc import turbo_rgb

        rgb = np.ascontiguousarray(turbo_rgb(composite))
        h, w, _ = rgb.shape
        image = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self.image_label.setPixmap(QPixmap.fromImage(image))
        self.image_label.setMinimumSize(w, h)
        self._rgb = rgb                       # keep the buffer alive alongside the pixmap
        self.history.append((int(iterations), float(ratio)))
        self.caption_label.setText(
            f"{iterations} iteration" + ("s" if iterations != 1 else "")
            + f"  ·  halo/core {ratio:.3f}")
        self.verdict_label.setText(verdict)
        self.trail_label.setText("  ".join(f"k={k}: {r:.3f}" for k, r in self.history))


class DeconQCPanel(_Panel):
    """PANE 1. Pick an iteration count, run it, judge the picture in pane 3, add one more."""

    def __init__(self, host):
        super().__init__(
            host, "Deconvolution (Richardson-Lucy)",
            "Richardson-Lucy is SEMI-CONVERGENT: the halo tightens for a few iterations and "
            "then a disc around the core starts growing back as the algorithm fits noise. "
            "There is no universally correct count, so run one, look at the turbo x-z / y-z "
            "view in pane 3, then add ONE more and look again.")
        from squidmip._decon import QC_START_ITERATIONS
        from squidmip._decon_qc import DEFAULT_CROP_HALF, DEFAULT_VIEW_HALF

        self._crop_half = DEFAULT_CROP_HALF
        self._view_half = DEFAULT_VIEW_HALF
        self._worker = None
        self._view = None
        self._view_subject = None

        self.v.addWidget(_head("WHERE TO MEASURE"))
        self.region_combo = QComboBox()
        for region in getattr(host, "_order", []):
            self.region_combo.addItem(region)
        self.fov_spin = QSpinBox()
        self.fov_spin.setRange(0, 9999)
        self.fov_spin.setToolTip(
            "The QC runs on ONE FOV. A recommendation is for THIS sample at THIS exposure — "
            "SNR and structure decide the answer, so it is never a global default.")
        self.v.addLayout(_row(QLabel("Region:"), self.region_combo,
                              QLabel("FOV:"), self.fov_spin))

        self.channel_combo = QComboBox()
        for name in _channel_names(host):
            self.channel_combo.addItem(name)
        self.channel_combo.setToolTip(
            "The emission wavelength of this channel sets the PSF. The kernel is a VECTORIAL "
            "PSF computed from the acquisition's own optics, not a Gaussian.")
        self.v.addLayout(_row(QLabel("Channel:"), self.channel_combo))

        self.v.addWidget(_head("ITERATIONS"))
        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(1, 100)
        self.iter_spin.setValue(QC_START_ITERATIONS)
        self.iter_spin.setToolTip("Richardson-Lucy iterations for the next run.")
        self.iter_hint = QLabel(f"shipped default: {_shipped_iterations()}")
        self.iter_hint.setStyleSheet(_SUB)
        self.plus_btn = QPushButton("+1 iteration")
        self.plus_btn.setToolTip(
            "Add exactly one and re-run. One at a time is the point: the turn is judged by "
            "eye between steps, and a jump of five hides where it happened.")
        self.plus_btn.clicked.connect(
            lambda: self.iter_spin.setValue(self.iter_spin.value() + 1))
        self.v.addLayout(_row(QLabel("Run with:"), self.iter_spin, self.iter_hint,
                              self.plus_btn))

        self.cpu_cb = QCheckBox("Force CPU (disable GPU)")
        self.cpu_cb.setToolTip(
            "Selects a BACKEND, not an algorithm — the RL update is identical either way.")
        self.v.addWidget(self.cpu_cb)

        self.run_btn = QPushButton("Deconvolve and show the turbo x-z / y-z view")
        self.run_btn.setCursor(Qt.PointingHandCursor)
        self.run_btn.clicked.connect(self.run)
        self.v.addWidget(self.run_btn)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setMaximumHeight(6)
        self.progress.setVisible(False)
        self.v.addWidget(self.progress)
        self.v.addWidget(self.status)

        note = _wrapped("The result opens as a tab in pane 3. Nothing is written next to "
                        "the acquisition; the datasets are opened read only.", _SUB)
        self.v.addWidget(note)
        self.v.addStretch(1)
        _apply_qss(self)

    # -- behaviour ---------------------------------------------------------------------
    def _subject(self) -> str:
        return (f"{self.region_combo.currentText()}/{self.fov_spin.value()}/"
                f"{self.channel_combo.currentText()}")

    def run(self) -> None:
        dataset = getattr(self.host, "_acq_path", None)
        if not dataset:
            self.say("no acquisition is open — deconvolution needs the dataset folder to read "
                     "its optics (NA, emission wavelength, pixel size, z-step) from.")
            return
        if self._worker is not None and self._worker.isRunning():
            self.say("a deconvolution is already running — let it finish before adding an "
                     "iteration.")
            return
        if not hasattr(self.host, "publish_qc_result"):
            self.say("this window cannot show a QC result: it does not implement "
                     "publish_qc_result(widget, title), which is how pane 3 is handed a "
                     "result tab. Refusing to deconvolve and then drop the picture.")
            return

        iterations = self.iter_spin.value()
        subject = self._subject()
        if self._view is None or self._view_subject != subject:
            self._view = DeconQCResultView(subject)
            self._view_subject = subject
            self.host.publish_qc_result(self._view, f"Decon QC · {subject}")

        self.progress.setVisible(True)
        self.run_btn.setEnabled(False)
        self.say(f"deconvolving {subject} at {iterations} iterations …")
        self._worker = _DeconQCWorker(
            dataset, self.region_combo.currentText(), self.fov_spin.value(),
            self.channel_combo.currentText(), iterations, not self.cpu_cb.isChecked(),
            self._crop_half, self._view_half)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_done(self, iterations, composite, ratio) -> None:
        from squidmip._decon_qc import halo_verdict

        self.progress.setVisible(False)
        self.run_btn.setEnabled(True)
        # The verdict needs THIS iteration included, and show_iteration is what appends it,
        # so build the history to judge on here rather than reading it back afterwards.
        history = list(self._view.history) + [(int(iterations), float(ratio))]
        kind, verdict = halo_verdict(history)
        self._view.show_iteration(iterations, composite, ratio, kind, verdict)
        self.say(verdict)

    def _on_failed(self, message: str) -> None:
        self.progress.setVisible(False)
        self.run_btn.setEnabled(True)
        self.say(f"deconvolution did not run: {message}")

    def stop(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(50)


def _shipped_iterations() -> int:
    from squidmip._decon import DEFAULT_ITERATIONS

    return DEFAULT_ITERATIONS
