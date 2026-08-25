"""The two HAND-WRITTEN operator interfaces: the stitcher's controls and decon's QC loop.

Every other operator's panel is built from its ``params`` declaration by
:mod:`squidxplorer._param_panel`. These two stay hand-written because they do things a
parameter form cannot: :class:`StitcherPanel` converts a percentage to a fraction, greys out
knobs that do nothing with registration off, and refuses a labels operator before the run
starts; :class:`DeconQCPanel` runs a semi-convergence SWEEP (one RL solve capturing every
iteration) and shows the iteration stepper INLINE, so the whole choose-the-iteration
loop lives wherever the panel is hosted — since 2026-08-24 that is the views window's
operator dock, never the plate window. ``_viewer._activate_operator`` prefers a hand-written
panel and falls back to the generic one.

This module never touches a tab bar or a dock; the host places the panel.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np
from qtpy.QtCore import Qt, QThread, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from squidxplorer import _qtstyle
from squidxplorer._stitch import _ABS_THRESH, _BLEND_PX, _REL_THRESH


def _stitch_declared() -> dict:
    """The ``stitch`` registration's own declared defaults — never mirrored here."""
    from squidxplorer._engine import operator_params

    return {p.name: p.default for p in operator_params("stitch")}


_DECLARED = _stitch_declared()

# The panel's starting position mirrors the stitch declaration. The divergences are the knobs
# the declaration deliberately does not carry (see _stitch._STITCH_PARAMS): blend_px
# (stitch_region's default is None, "measure the real overlap", and the panel starts at the
# fixed fallback with an "Auto" box beside it), channels (None = all of them; a subset is
# panel state), correct_distortion (None = on wherever registration ran) and the two outlier
# thresholds, stated by _stitch's own constants.
STITCH_DEFAULTS = {
    "register": _DECLARED["register"],
    "registration_channel": _DECLARED["registration_channel"],
    "channels": None,
    "blend_px": _BLEND_PX,
    "outlier_rel_pct": int(round(_REL_THRESH * 100)),
    "outlier_abs_px": int(round(_ABS_THRESH)),
    "auto_blend": False,
    "correct_distortion": True,  # ON by default (Julio, 2026-08-03)
    "registration_t": _DECLARED["registration_t"],
}


def stitch_operator_kwargs(*, register, registration_channel, channels, blend_px,
                           outlier_rel_pct, outlier_abs_px,
                           auto_blend: bool = False,
                           correct_distortion: bool = True,
                           registration_t: int = 0,
                           n_channels: Optional[int] = None,
                           tile_px: Optional[int] = None) -> dict:
    """Turn the panel's widget values into ``stitch_region`` keyword arguments."""
    if channels is not None:
        channels = [int(c) for c in channels]
        if not channels:
            raise ValueError(
                "select at least one channel to fuse: a mosaic with no channels is not a "
                "smaller result, it is no result.")
        if n_channels is not None and len(channels) == int(n_channels):
            channels = None
    # "Auto" is spelled blend_px=None down to stitch_region, which measures the real overlap.
    # The spin's value is ignored, not clamped.
    if auto_blend:
        blend_px = None
    else:
        blend_px = int(blend_px)
    if blend_px is not None and tile_px is not None and blend_px >= int(tile_px):
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
        kwargs["registration_t"] = int(registration_t)
        # Distortion correction fits the residual left after the global solve, so it needs
        # registration; drop it here rather than forward a False.
        kwargs["correct_distortion"] = bool(correct_distortion)
    return kwargs


#: The Z-handling combo's spelling of ``z_operator=None``: fuse every acquired plane unchanged,
#: no inner operator. NOT a registry name (the registered `keepz` identity was shelved
#: 2026-08-24); the panel maps this label to None before anything asks the registry.
KEEP_EVERY_PLANE = "keep every z plane"


def stitch_refusal(name: str) -> Optional[str]:
    """The refusal sentence for *name*, mirroring stitch_region's own guard against fusing labels."""
    from squidxplorer._stitch import LABELS, _resolve_operator

    try:
        op = _resolve_operator(name)
    except Exception as exc:                       # unknown name -> name it, don't crash
        return (f"{name!r} is not an operator this build knows: {exc}")
    if op.produces != LABELS:
        return None
    return (
        f"{name!r} produces label images (integer object ids), and fusion blends "
        f"overlapping tiles by a weighted average - the mean of two label ids is a third, "
        f"nonexistent object, and per-FOV ids collide across every seam. Segment per FOV "
        f"instead, or stitch an intensity operator such as mip or decon.")


_BG = "#0d1117"
_SUB = "color:#8b98ad;font-size:11px;"
_HEAD = "color:#57606a;font-size:10px;font-weight:800;letter-spacing:1.5px;padding-top:6px;"


def _apply_qss(root: QWidget) -> None:
    """Style every control in *root* with the app-wide dark theme (squidxplorer._qtstyle)."""
    for w in root.findChildren(QPushButton):
        w.setStyleSheet(_qtstyle.BTN_QSS)
        w.setCursor(Qt.PointingHandCursor)
    for w in root.findChildren(QComboBox):
        w.setStyleSheet(_qtstyle.COMBO_QSS)
    for w in root.findChildren(QSpinBox):
        w.setStyleSheet(_qtstyle.COMBO_QSS)
    for w in root.findChildren(QCheckBox):
        w.setStyleSheet(_qtstyle.CHECK_QSS)


def _wrapped(text: str, style: str) -> QLabel:
    """Word-wrapped QLabel that reserves the height its wrapping needs (a plain one doesn't)."""
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

        # Scroll area: the control stacks (esp. the stitcher's) are taller than the pane.
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
        """Put a sentence in front of the user; also routed to the host's own readout."""
        self.status.setText(text)
        say = getattr(self.host, "say", None)
        if callable(say):
            say(text)


class StitcherPanel(_Panel):
    """maragall/stitcher's control surface over squidxplorer's ``stitch`` region operator."""

    def __init__(self, host):
        super().__init__(
            host, "Stitch (register + fuse)",
            "Register every FOV of a region against its neighbours and fuse one seamless "
            "mosaic. A region is a MOSAIC of FOVs, so one region produces one image.")
        names = _channel_names(host)

        # Scope is not a control here: it belongs to the run, and the window's "run on"
        # selector owns it live (a cached scope combo here was stale by construction).

        # Z HANDLING, not Z REDUCTION: stitch_region fuses per plane, so reduction is one of
        # two things that can happen here, not the only one.
        self.v.addWidget(_head("Z HANDLING"))
        self._n_z = int(((getattr(host, "_meta", None) or {}).get("n_z")) or 1)
        self.z_operator_combo = QComboBox()
        from squidxplorer import available_plane_operators

        # Not a registry name: the combo's spelling of z_operator=None — every acquired plane,
        # fused unchanged. It replaced the registered `keepz` identity operator (2026-08-24).
        self.z_operator_combo.addItem(KEEP_EVERY_PLANE)
        for name in sorted(available_plane_operators()):
            self.z_operator_combo.addItem(name)
        # The declared default even on a z-stack: RegionViewer switches to keep-every-plane
        # only when the window is actually in 3D mode, so a 2D canvas never gets a volume it
        # can't show.
        self.z_operator_combo.setCurrentText(_DECLARED["z_operator"])
        self.z_operator_combo.setToolTip(
            "What each FOV's z-stack becomes before registration.\n\n"
            "A z-REDUCER (mip, reference) collapses it to one plane, so the well fuses to one "
            "image. A PLANE-OP (flatfield) keeps every plane, and the well fuses to a "
            "volume: the pose graph is solved ONCE and every plane is fused from those same "
            "origins, so the planes cannot drift apart.\n\n"
            f"'{KEEP_EVERY_PLANE}' fuses every acquired plane unchanged (z_operator=None).")
        self.z_operator_combo.currentTextChanged.connect(self._check_z_operator)
        self.v.addLayout(_row(QLabel("Z handling:"), self.z_operator_combo))

        # z-plane count comes from the chosen operator's `consumes` declaration crossed with
        # n_z, not from anything visible on this panel; kept in sync by _check_z_operator.
        self.z_note = QLabel("")
        self.z_note.setWordWrap(True)
        self.v.addWidget(self.z_note)

        self.v.addWidget(_head("REGISTRATION"))
        self.register_cb = QCheckBox("Enable registration refinement")
        self.register_cb.setChecked(STITCH_DEFAULTS["register"])
        self.register_cb.setToolTip(
            "On: phase-correlate overlapping pairs and solve a global pose graph.\n"
            "Off: pure coordinate placement from the stage - the honest control for judging "
            "whether registration actually helped.")
        self.register_cb.toggled.connect(self._on_register_toggled)
        self.v.addWidget(self.register_cb)

        self.reg_channel_combo = QComboBox()
        for name in names:
            self.reg_channel_combo.addItem(name)
        self.reg_channel_combo.setToolTip(
            "ONE channel drives the geometry and every channel is then fused with that one "
            "solution - channels of a FOV share a sensor and must not get disagreeing "
            "placements.")
        self.v.addLayout(_row(QLabel("Registration channel:"), self.reg_channel_combo))

        # Hidden on a single-timepoint acquisition, where 0 is the only legal value.
        n_t = int(((getattr(host, "_meta", None) or {}).get("n_t")) or 1)
        self.reg_t_spin = QSpinBox()
        self.reg_t_spin.setRange(0, max(n_t - 1, 0))
        self.reg_t_spin.setToolTip(
            "Which timepoint the pose graph is solved on. Every timepoint is then fused with "
            "that ONE solution, so a drifting stage does not give t=0 and t=9 different "
            "placements.")
        self.reg_t_row = _row(QLabel("Registration timepoint:"), self.reg_t_spin)
        if n_t > 1:
            self.v.addLayout(self.reg_t_row)

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

        # Unlike maragall/stitcher, where this checkbox is dead (created but never read), here
        # it actually decides; tests pin both directions.
        self.distortion_cb = QCheckBox("Correct lens distortion (per-seam elastic)")
        self.distortion_cb.setChecked(STITCH_DEFAULTS["correct_distortion"])
        self.distortion_cb.setToolTip(
            "Fit a per-tile elastic warp from the REGISTERED seams and apply it during fusion "
            "(tilefusion.distortion). It corrects what a rigid solve cannot: field curvature "
            "and lens distortion bending each tile, which shows up as seams that are sharp in "
            "the middle and doubled at the ends.\n\n"
            "Needs registration -- it corrects the residual left after the global solve.\n\n"
            "ON by default. Untick it to fuse on the rigid solve alone, which is faster and is "
            "the right control when you want to see what the elastic fit is actually buying.")
        self.v.addWidget(self.distortion_cb)

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
        self.blend_auto_cb = QCheckBox("Auto (measure the real overlap)")
        self.blend_auto_cb.setChecked(STITCH_DEFAULTS["auto_blend"])
        self.blend_auto_cb.setToolTip(
            "Measure this acquisition's actual seam overlap and size the ramp to it (median "
            "overlap x 2), instead of the fixed default -- which is sized to ONE acquisition "
            "(~208 px on the 10x tissue set) and is wrong on a denser grid.")
        self.blend_auto_cb.toggled.connect(lambda on: self.blend_spin.setEnabled(not on))
        self.v.addLayout(_row(QLabel("Blend width:"), self.blend_spin, self.blend_auto_cb))

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

        # NO run button and NO save checkbox here (one flow, Julio 2026-08-25): the view's
        # operators row launches every run — Preview (window scope) and Run on plate (the one
        # save). This panel is the PARAMETERS; `kwargs()` is how they reach a run.
        self.v.addWidget(self.status)
        self.v.addStretch(1)

        _apply_qss(self)
        self._on_register_toggled(self.register_cb.isChecked())
        self._check_z_operator(self.z_operator_combo.currentText())

    def _on_register_toggled(self, on: bool) -> None:
        """Grey out the knobs that provably do nothing with registration off."""
        for w in (self.reg_channel_combo, self.rel_spin, self.abs_spin,
                  self.reg_t_spin, self.distortion_cb):
            w.setEnabled(bool(on))

    def _check_z_operator(self, name: str) -> None:
        why = None if name == KEEP_EVERY_PLANE else stitch_refusal(name)
        self.say("" if why is None else why)
        self.z_note.setText(self._z_line(name))

    def _z_line(self, name: str) -> str:
        """How many z-levels this run will stitch, read off the operator's `consumes` declaration."""
        from squidxplorer._engine import Z_REDUCER, operator_consumes

        if name == KEEP_EVERY_PLANE:
            if self._n_z <= 1:
                return "z: this acquisition has 1 plane, so there is one plane to fuse."
            return (f"z: all {self._n_z} planes, unchanged (z_operator=None). The pose graph is "
                    f"solved ONCE and every plane is fused from those same origins, so the "
                    f"planes cannot drift apart. Renderable in 3D.")
        try:
            reduces = bool(operator_consumes(name) & Z_REDUCER)
        except Exception as exc:                      # noqa: BLE001 - unknown name, reported
            return f"cannot say how many z-levels {name!r} would stitch: {exc}"
        if reduces:
            return (f"z: {self._n_z} acquired plane(s) → 1 fused plane. {name!r} collapses z, so "
                    f"the result is flat and cannot be opened in 3D.")
        if self._n_z <= 1:
            return "z: this acquisition has 1 plane, so there is one plane to fuse."
        return (f"z: all {self._n_z} planes. maragall/stitcher runs on each z-level "
                f"independently - the pose graph is solved ONCE and every plane is fused from "
                f"those same origins, so the planes cannot drift apart. Renderable in 3D.")

    def kwargs(self) -> dict:
        """The panel's settings as ``stitch_region`` keyword arguments."""
        meta = getattr(self.host, "_meta", None) or {}
        frame = meta.get("frame_shape") or (None, None)
        selected = [i for i, cb in enumerate(self.channel_boxes) if cb.isChecked()]
        return stitch_operator_kwargs(
            register=self.register_cb.isChecked(),
            registration_channel=(self.reg_channel_combo.currentIndex()
                                  if self.register_cb.isChecked() else None),
            # An empty selection means ALL, never none: channels=[] would fuse zero channels
            # and report a green run with no output.
            channels=(selected or None) if self.channel_boxes else None,
            blend_px=self.blend_spin.value(),
            auto_blend=self.blend_auto_cb.isChecked(),
            correct_distortion=self.distortion_cb.isChecked(),
            registration_t=self.reg_t_spin.value(),
            outlier_rel_pct=self.rel_spin.value(),
            outlier_abs_px=self.abs_spin.value(),
            n_channels=len(self.channel_boxes) or None,
            tile_px=min(frame) if all(frame) else None,
        )


def z_operator_choice(text: str):
    """The z-handling combo's text as ``stitch_region``'s ``z_operator`` value: the
    keep-every-plane label spells ``None`` (every acquired plane, fused unchanged)."""
    return None if str(text) == KEEP_EVERY_PLANE else str(text)


class QCFrame(NamedTuple):
    """One captured QC iteration, as the panel receives it. The volume is the preview: it
    lands in a view as a real data layer under the channel's own colormap (the turbo
    composite panes were removed, Julio 2026-08-25: "The turbo colormap preview makes no
    sense. remove it")."""

    volume: object             # (Z, Y, X) the deconvolved crop
    centre: tuple              # (z, y, x) in `volume`: the brightest structure RL was judged at
    view_half: object          # the lateral half-width a section view would cut to, or None
    delta: object = None       # mean |delta| against the previous iteration's volume, or None
    fov_origin: object = None  # (y0, x0) of the crop inside its FOV frame, in FOV pixels


class _DeconQCWorker(QThread):
    """ONE RL solve on ONE FOV's z-stack, capturing EVERY iteration up to the count
    (petakit's ``snapshot_iters``) and measuring the halo per capture (threaded).

    One solve, not one solve per count: stepping k back and forth afterwards is a repaint of
    the captured crops. The captures are of the QC CROP (crop_half around the brightest
    structure), so a sweep's memory is iterations × one small volume, never the full stack."""

    done = Signal(int, object, float)        # per captured iteration: (k, QCFrame, halo/core)
    sweep_done = Signal(int)                 # the sweep finished; how many iterations landed
    failed = Signal(str)

    def __init__(self, dataset, region, fov, channel, iterations, gpu, crop_half, view_half):
        super().__init__()
        self._args = (dataset, region, fov, channel, iterations, gpu, crop_half, view_half)

    def run(self):
        dataset, region, fov, channel, iterations, gpu, crop_half, view_half = self._args
        if self.isInterruptionRequested():
            return          # a close/shutdown beat us to the start
        try:
            from squidxplorer._decon import OpticsParams, _run, make_psf, optics_for_channel
            from squidxplorer._decon_qc import (
                brightest_structure,
                crop_around,
                halo_core_ratio,
                load_stack,
                qc_window_um,
            )

            stack, region, channel, _meta = load_stack(dataset, region, fov, channel)
            # Through optics_for_channel, never from_acquisition directly: the session's
            # immersion / NA choices must reach the QC's PSF exactly as they reach a run's.
            optics = optics_for_channel(dataset, channel)
            optics = OpticsParams(optics.na, optics.wavelength_um, optics.dxy_um,
                                  optics.dz_um, int(stack.shape[0]), optics.ni)
            core_um = 0.61 * optics.wavelength_um / optics.na
            window_um = qc_window_um(core_um, stack.shape[0], optics.dz_um)
            z_margin = int(np.ceil(window_um / optics.dz_um))
            centre_full = brightest_structure(stack, optics.dxy_um, optics.dz_um, core_um,
                                              z_margin=z_margin, xy_margin=crop_half)
            crop, centre = crop_around(stack, centre_full, crop_half)
            snaps = list(range(1, int(iterations) + 1))
            volumes = _run(crop, make_psf(optics), int(iterations), gpu=gpu,
                           snapshot_iters=snaps)
            centre_t = tuple(int(v) for v in centre)
            # Where the crop sits inside its FOV frame, recoverable from the two centres —
            # what places the sweep's data preview at its true stage position in a view.
            origin = (int(centre_full[1]) - int(centre[1]),
                      int(centre_full[2]) - int(centre[2]))
            previous = np.ascontiguousarray(crop, dtype=np.float32)
            for k in snaps:
                if self.isInterruptionRequested():
                    return
                volume = volumes[k]
                ratio = halo_core_ratio(volume, centre, optics.dxy_um, optics.dz_um,
                                        core_um, window_um)
                delta = float(np.mean(np.abs(volume - previous)))
                previous = volume
                self.done.emit(int(k),
                               QCFrame(volume, centre_t, view_half, delta, origin),
                               float(ratio))
            self.sweep_done.emit(len(snaps))
        except Exception as exc:                  # reported as a sentence, never swallowed
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class DeconQCResultView(QWidget):
    """The iteration stepper over the sweep's captures. The PICTURE is the in-view data
    preview: each displayed capture lands in a view as a real layer under the channel's own
    colormap (the turbo composite panes were removed, Julio 2026-08-25: "The turbo colormap
    preview makes no sense. remove it"; the x-z / y-z strips went with them — they could not
    stand without the turbo rendering and its legend).

    Every shown iteration is KEPT (Julio, 2026-08-24: "as I click iteration by iteration") so
    stepping is a repaint of a cached capture, never a re-solve; the "use k iterations"
    button emits :attr:`useIterations` with the DISPLAYED count — the whole preview exists
    so the user can decide how many iterations the real run gets."""

    useIterations = Signal(int)            # the displayed k, adopted as the run's iterations
    iterationDisplayed = Signal(int)       # a capture was (re)painted: the data preview follows

    def __init__(self, subject: str):
        super().__init__()
        self.setStyleSheet(f"background:{_BG};color:#e6edf3;")
        self.history: list = []
        #: k -> the record show_iteration stored; the stepper repaints from these.
        self._records: dict = {}
        self._shown_k = None
        v = QVBoxLayout(self)
        v.setContentsMargins(14, 12, 14, 12)
        v.setSpacing(8)
        t = QLabel(f"Deconvolution QC · {subject}")
        t.setStyleSheet("font-size:15px;font-weight:800;")
        v.addWidget(t)

        self.caption_label = QLabel("")
        self.caption_label.setWordWrap(True)
        self.caption_label.setStyleSheet("font-size:12px;font-weight:700;")
        v.addWidget(self.caption_label)

        self.verdict_label = QLabel("")
        self.verdict_label.setWordWrap(True)
        self.verdict_label.setStyleSheet("color:#d29922;font-size:11px;")
        v.addWidget(self.verdict_label)

        self.trail_label = QLabel("")
        self.trail_label.setWordWrap(True)
        self.trail_label.setStyleSheet(_SUB)
        v.addWidget(self.trail_label)

        # -- the iteration stepper: prev/next + a slider over the CAPTURED iterations, and the
        # -- adoption button. Hidden until a second iteration exists to step between. ----------
        self.prev_btn = QPushButton("◂")
        self.prev_btn.setToolTip("Show the previous captured iteration.")
        self.next_btn = QPushButton("▸")
        self.next_btn.setToolTip("Show the next captured iteration.")
        for b in (self.prev_btn, self.next_btn):
            b.setCursor(Qt.PointingHandCursor)
            b.setFixedWidth(28)
        self.iter_slider = QSlider(Qt.Horizontal)
        self.iter_slider.setRange(1, 1)
        self.iter_slider.setToolTip(
            "Every iteration of the sweep is kept; stepping repaints a capture, nothing is "
            "re-deconvolved.")
        self.prev_btn.clicked.connect(lambda *_: self._step(-1))
        self.next_btn.clicked.connect(lambda *_: self._step(+1))
        self.iter_slider.valueChanged.connect(self._on_slider)
        self._stepper_row = QWidget()
        sr = QHBoxLayout(self._stepper_row)
        sr.setContentsMargins(0, 0, 0, 0)
        sr.setSpacing(6)
        sr.addWidget(self.prev_btn)
        sr.addWidget(self.iter_slider, 1)
        sr.addWidget(self.next_btn)
        self._stepper_row.setVisible(False)
        v.addWidget(self._stepper_row)

        self.use_btn = QPushButton("Use these iterations")
        self.use_btn.setCursor(Qt.PointingHandCursor)
        self.use_btn.setToolTip(
            "Adopt the DISPLAYED iteration count as the decon run's iterations parameter - "
            "the whole preview exists to make this choice.")
        self.use_btn.setVisible(False)
        self.use_btn.clicked.connect(self._on_use)
        v.addWidget(self.use_btn)

    def show_iteration(self, iterations: int, ratio: float,
                       kind: str, verdict: str, volume=None, centre=None,
                       view_half=None, delta=None, fov_origin=None) -> None:
        """Record one iteration's capture and display it, so the loop can be compared and
        the stepper can revisit it. The volume is what the in-view data preview renders."""
        k = int(iterations)
        self._records[k] = {
            "ratio": float(ratio), "verdict": verdict,
            "volume": None if volume is None else np.asarray(volume),
            "centre": None if centre is None else tuple(int(v) for v in centre),
            "view_half": view_half, "delta": delta,
            "fov_origin": None if fov_origin is None else tuple(int(v) for v in fov_origin),
        }
        self.history.append((k, float(ratio)))
        self.trail_label.setText("  ".join(f"k={kk}: {r:.3f}" for kk, r in self.history))
        ks = sorted(self._records)
        self.iter_slider.blockSignals(True)      # a range change must not re-enter _display
        self.iter_slider.setRange(ks[0], ks[-1])
        self.iter_slider.blockSignals(False)
        self._stepper_row.setVisible(len(ks) > 1)
        self.use_btn.setVisible(True)
        self.verdict_label.setText(verdict)
        self._display(k)

    # -- stepping -----------------------------------------------------------------------------
    def _display(self, k: int) -> None:
        """Show the captured iteration *k*: caption, adoption button, and the preview push."""
        rec = self._records.get(int(k))
        if rec is None:
            return
        self._shown_k = int(k)
        top = max(self._records)
        self.caption_label.setText(
            f"ITERATION {k} of {top}  ·  halo/core {rec['ratio']:.3f}"
            + (f"  ·  mean |Δ| vs k-1: {rec['delta']:.4g}" if rec["delta"] is not None else ""))
        self.use_btn.setText(f"Use {k} iteration" + ("s" if k != 1 else "")
                             + " for the decon run")
        self.iter_slider.blockSignals(True)
        self.iter_slider.setValue(int(k))
        self.iter_slider.blockSignals(False)
        self.iterationDisplayed.emit(int(k))

    def capture(self, k: int) -> Optional[dict]:
        """The stored record for iteration *k* (volume, fov_origin, ...), or None."""
        return self._records.get(int(k))

    def _step(self, direction: int) -> None:
        if self._shown_k is None:
            return
        ks = sorted(self._records)
        try:
            i = ks.index(self._shown_k)
        except ValueError:
            return
        j = min(max(i + int(direction), 0), len(ks) - 1)
        self._display(ks[j])

    def _on_slider(self, value: int) -> None:
        if int(value) in self._records:
            self._display(int(value))

    def _on_use(self, *_) -> None:
        if self._shown_k is not None:
            self.useIterations.emit(int(self._shown_k))


class DeconQCPanel(_Panel):
    """Sweep the iterations once, step through the captures, and ADOPT the count that looks right.

    The preview exists so the user can DECIDE how many iterations the real run gets (Julio,
    2026-08-24): one solve captures every iteration, the inline stepper revisits them (the
    in-view data preview is the picture), and 'use k iterations' writes k into the run's own
    parameter."""

    def __init__(self, host):
        super().__init__(
            host, "Deconvolution (Richardson-Lucy)",
            "Semi-convergent: sweep once, step the captures, adopt the count that looks right.")
        from squidxplorer._decon import (
            DEFAULT_ITERATIONS, IMMERSION_MEDIA, QC_START_ITERATIONS,
            session_ni, set_session_ni,
        )
        from squidxplorer._decon_qc import DEFAULT_CROP_HALF, DEFAULT_VIEW_HALF

        self._crop_half = DEFAULT_CROP_HALF
        self._view_half = DEFAULT_VIEW_HALF
        self._worker = None
        self._view = None
        self._view_subject = None
        self._sweep_at = None            # (region, fov, channel) the captures belong to

        # ── THE HEADLINE: iterations + NI + NA. Everything else stays at its default and
        # hides (Julio, 2026-08-25: "There are so many parameters that should be default and
        # our life science user should not want to see them"). ─────────────────────────────
        self.run_iter_spin = QSpinBox()
        self.run_iter_spin.setRange(1, 100)
        self.run_iter_spin.setValue(DEFAULT_ITERATIONS)
        self.run_iter_spin.setToolTip(
            "The iteration count every decon run uses while this panel is open; the sweep's "
            "'use k iterations' button writes here.")
        self.v.addLayout(_row(QLabel("iterations:"), self.run_iter_spin))

        # The immersion index is CHOSEN, never inferred off the NA (Julio, 2026-08-24); the
        # choice lands in _decon's session state, one source of truth for QC and runs alike.
        self.ni_combo = QComboBox()
        for value, medium in IMMERSION_MEDIA:
            self.ni_combo.addItem(f"{value:.3f} ({medium})", value)
        self.ni_combo.addItem("custom", None)     # data None = read the spin beside it
        self.ni_custom = QDoubleSpinBox()
        self.ni_custom.setRange(1.0, 2.0)
        self.ni_custom.setDecimals(3)
        self.ni_custom.setSingleStep(0.01)
        self.ni_custom.setToolTip("A custom immersion index for this session.")
        self.ni_custom.setVisible(False)
        current_ni = session_ni()
        if current_ni is not None:            # the session already chose; a rebuild keeps it
            matched = False
            for i in range(self.ni_combo.count()):
                data = self.ni_combo.itemData(i)
                if data is not None and abs(float(data) - current_ni) < 5e-3:
                    self.ni_combo.setCurrentIndex(i)
                    matched = True
                    break
            if not matched:                   # a custom value survives the rebuild too
                self.ni_combo.setCurrentIndex(self.ni_combo.count() - 1)
                self.ni_custom.setValue(float(current_ni))
                self.ni_custom.setVisible(True)
        self.ni_combo.setToolTip(
            "The immersion medium's refractive index (shapes the axial PSF). Holds for this "
            "session, for the sweep and every decon run alike.")
        self.ni_combo.currentIndexChanged.connect(self._on_ni_changed)
        self.ni_custom.valueChanged.connect(self._on_ni_changed)
        set_session_ni(self._current_ni())
        self.v.addLayout(_row(QLabel("NI:"), self.ni_combo, self.ni_custom))

        self.na_spin = QDoubleSpinBox()
        self.na_spin.setRange(0.0, 1.7)
        self.na_spin.setDecimals(2)
        self.na_spin.setSingleStep(0.05)
        self.na_spin.setSpecialValueText("recorded")
        self.na_spin.setToolTip(
            "The objective NA the PSF is computed with; 'recorded' uses the acquisition's "
            "own value, printed beside.")
        self.na_spin.valueChanged.connect(self._on_na_changed)
        # "recorded NA should print the value to it's side" (Julio, 2026-08-25).
        self.na_recorded = QLabel("")
        self.na_recorded.setStyleSheet(_SUB)
        self.v.addLayout(_row(QLabel("NA:"), self.na_spin, self.na_recorded))

        self.run_btn = QPushButton("Run the iteration sweep")
        self.run_btn.setCursor(Qt.PointingHandCursor)
        self.run_btn.setToolTip(
            "One RL solve to the sweep count, capturing every iteration; step them below "
            "and adopt the one that looks right.")
        self.run_btn.clicked.connect(self.run)
        self.v.addWidget(self.run_btn)

        # ── the two disclosures: 'more' (where to measure + sweep knobs) and 'PSF detail'.
        self.more_btn = QPushButton("more…")
        self.more_btn.setCheckable(True)
        self.more_btn.setToolTip("Where the sweep measures, and its knobs.")
        self.psf_btn = QPushButton("PSF detail")
        self.psf_btn.setCheckable(True)
        self.psf_btn.setToolTip("The recorded optics and the values the solve will use.")
        self.v.addLayout(_row(self.more_btn, self.psf_btn))

        self._more = QWidget()
        mv = QVBoxLayout(self._more)
        mv.setContentsMargins(0, 0, 0, 0)
        mv.setSpacing(6)
        self.region_combo = QComboBox()
        for region in getattr(host, "_order", []):
            self.region_combo.addItem(region)
        self.fov_spin = QSpinBox()
        self.fov_spin.setRange(0, 9999)
        self.fov_spin.setToolTip("The sweep runs on this ONE FOV.")
        mv.addLayout(_row(QLabel("Region:"), self.region_combo,
                          QLabel("FOV:"), self.fov_spin))
        self.channel_combo = QComboBox()
        for name in _channel_names(host):
            self.channel_combo.addItem(name)
        self.channel_combo.setToolTip(
            "The PREVIEW channel: the sweep solves this one only; a run covers every channel.")
        self.channel_combo.currentTextChanged.connect(
            lambda *_: self._refresh_optics_note())
        mv.addLayout(_row(QLabel("Channel:"), self.channel_combo))
        # The preview-vs-run contract: Julio (2026-08-24) read a one-channel preview as
        # "decon was running on only one channel".
        self.preview_note = _wrapped("", _SUB)
        mv.addWidget(self.preview_note)
        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(1, 100)
        self.iter_spin.setValue(QC_START_ITERATIONS)
        self.iter_spin.setToolTip("The sweep's top count; every iteration on the way is kept.")
        self.iter_hint = QLabel(f"shipped default: {_shipped_iterations()}")
        self.iter_hint.setStyleSheet(_SUB)
        self.plus_btn = QPushButton("+1 iteration")
        self.plus_btn.setToolTip("Extend the sweep by exactly one and re-run.")
        self.plus_btn.clicked.connect(
            lambda: self.iter_spin.setValue(self.iter_spin.value() + 1))
        mv.addLayout(_row(QLabel("Sweep to:"), self.iter_spin, self.iter_hint, self.plus_btn))
        self.cpu_cb = QCheckBox("Force CPU (disable GPU)")
        self.cpu_cb.setToolTip("A backend choice, not an algorithm - the RL update is identical.")
        mv.addWidget(self.cpu_cb)
        self._more.setVisible(False)
        self.more_btn.toggled.connect(self._more.setVisible)
        self.v.addWidget(self._more)

        # The recorded optics and the FINAL values the PSF is built from (Julio, 2026-08-24:
        # "the decon UI should show the final magnification / psf parameters"), on demand.
        self._psf = QWidget()
        pv2 = QVBoxLayout(self._psf)
        pv2.setContentsMargins(0, 0, 0, 0)
        pv2.setSpacing(4)
        self.optics_note = _wrapped("", _SUB)
        pv2.addWidget(self.optics_note)
        self.effective_note = _wrapped("", "color:#e6edf3;font-size:12px;font-weight:700;")
        pv2.addWidget(self.effective_note)
        self._psf.setVisible(False)
        self.psf_btn.toggled.connect(self._psf.setVisible)
        self.v.addWidget(self._psf)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setMaximumHeight(6)
        self.progress.setVisible(False)
        self.v.addWidget(self.progress)
        self.v.addWidget(self.status)

        # The result view lives INLINE, right under its controls — the panel is one surface
        # wherever it is hosted (a dock page or a tab), never a picture thrown to another
        # window (2026-08-24; publish_qc_result stays a window capability, unused here).
        self._view_slot = QVBoxLayout()
        self._view_slot.setContentsMargins(0, 0, 0, 0)
        self.v.addLayout(self._view_slot)

        self.v.addStretch(1)
        _apply_qss(self)
        self._refresh_optics_note()

    def _current_ni(self) -> float:
        """The NI the session gets: the combo's medium, or the custom spin's typed value."""
        data = self.ni_combo.currentData()
        return float(self.ni_custom.value()) if data is None else float(data)

    def _subject(self) -> str:
        return (f"{self.region_combo.currentText()}/{self.fov_spin.value()}/"
                f"{self.channel_combo.currentText()}")

    # -- optics row ---------------------------------------------------------------------------
    def _recorded_optics(self):
        """``(OpticsParams, "")`` for the current channel, or ``(None, why)`` — never a guess."""
        from squidxplorer._decon import OpticsParams

        dataset = getattr(self.host, "_acq_path", None)
        if not dataset:
            return None, "no acquisition is open"
        try:
            return OpticsParams.from_acquisition(dataset, channel=self.channel_combo.currentText()), ""
        except Exception as exc:               # noqa: BLE001 - shown as a sentence, not hidden
            return None, f"{exc}"

    def _sensor_pixel_um(self):
        """``sensor_pixel_size_um`` from 'acquisition parameters.json', or None — never guessed."""
        import json as _json
        from pathlib import Path as _Path

        dataset = getattr(self.host, "_acq_path", None)
        if not dataset:
            return None
        try:
            params = _json.loads((_Path(str(dataset)) / "acquisition parameters.json").read_text())
            value = float(params.get("sensor_pixel_size_um"))
            return value if value > 0 else None
        except Exception:                      # noqa: BLE001 - absent file/key: no magnification
            return None

    def _effective_optics(self):
        """The recorded optics WITH the session's NI/NA applied — what the solve will use.

        ``(OpticsParams, "")``, or ``(None, why)`` when unreadable or physically refused."""
        from squidxplorer._decon import apply_session_optics

        optics, why = self._recorded_optics()
        if optics is None:
            return None, why
        try:
            return apply_session_optics(optics), ""
        except ValueError as exc:              # NA impossible in this medium: the named refusal
            return None, str(exc)

    def _refresh_optics_note(self) -> None:
        """The auto-derived optics AND the final PSF parameters, on screen BEFORE a run."""
        from squidxplorer._decon import medium_for_ni, session_na

        channel = self.channel_combo.currentText()
        self.preview_note.setText(
            f"preview: {channel} only. Run deconvolves every channel, each with its own PSF "
            "(wavelength per channel)." if channel else "")
        optics, why = self._recorded_optics()
        if optics is None:
            self.optics_note.setText(f"recorded optics unreadable: {why}")
            self.na_recorded.setText("recorded NA: ?")
        else:
            self.optics_note.setText(
                f"recorded: NA {optics.na:.2f} · emission {optics.wavelength_um:.3f} µm · "
                f"pixel {optics.dxy_um:.3f} µm · dz {optics.dz_um:.2f} µm. A wrong rig "
                "profile shows up here.")
            # The value beside the control (Julio, 2026-08-25: "recorded NA should print
            # the value to it's side").
            self.na_recorded.setText(f"recorded NA: {optics.na:.2f}")
        effective, e_why = self._effective_optics()
        if effective is None:
            self.effective_note.setText("" if optics is None
                                        else f"the solve will refuse: {e_why}")
        else:
            ni = effective.ni if effective.ni is not None else self._current_ni()
            # mag = sensor / dxy is GEOMETRY: NA and ni shape the PSF, never this number.
            sensor = self._sensor_pixel_um()
            mag = (f" · magnification {sensor / effective.dxy_um:.1f}x "
                   f"(sensor {sensor:g} µm / pixel)" if sensor else "")
            self.effective_note.setText(
                f"the solve will use: NA {effective.na:.2f} · {medium_for_ni(ni)} "
                f"(ni {float(ni):.3f}) · emission {effective.wavelength_um:.3f} µm "
                f"({channel}) · pixel {effective.dxy_um:.3f} µm · dz {effective.dz_um:.2f} µm "
                f"· nz {int(effective.nz)}{mag}")
        na = session_na() or (optics.na if optics is not None else None)
        ni = self._current_ni()
        if na is not None and na > ni + 1e-9:
            self.say(f"NA {na:.2f} is impossible in {medium_for_ni(ni)} (ni {ni:.3f}): pick "
                     "the objective's actual immersion. The solve will refuse until they agree.")

    def _on_ni_changed(self, *_) -> None:
        from squidxplorer._decon import set_session_ni

        self.ni_custom.setVisible(self.ni_combo.currentData() is None)
        set_session_ni(self._current_ni())
        self.say("")
        self._refresh_optics_note()

    def _on_na_changed(self, value: float) -> None:
        # 0.00 shows as 'recorded' and clears the override; anything else installs it.
        from squidxplorer._decon import set_session_na

        set_session_na(float(value) if value > 0 else None)
        self.say("")
        self._refresh_optics_note()

    # -- the sweep ------------------------------------------------------------------------------
    def run(self) -> None:
        dataset = getattr(self.host, "_acq_path", None)
        if not dataset:
            self.say("no acquisition is open - deconvolution needs the dataset folder to read "
                     "its optics (NA, emission wavelength, pixel size, z-step) from.")
            return
        if self._worker is not None and self._worker.isRunning():
            self.say("a deconvolution is already running - let it finish before adding an "
                     "iteration.")
            return

        iterations = self.iter_spin.value()
        subject = self._subject()
        # What the sweep's captures BELONG to, for the in-view data preview's placement.
        self._sweep_at = (self.region_combo.currentText(), int(self.fov_spin.value()),
                          self.channel_combo.currentText())
        if self._view is None or self._view_subject != subject:
            if self._view is not None:
                self._view_slot.removeWidget(self._view)
                self._view.deleteLater()
            self._view = DeconQCResultView(subject)
            self._view.useIterations.connect(self._adopt_iterations)
            self._view.iterationDisplayed.connect(self._push_view_preview)
            self._view_subject = subject
            self._view_slot.addWidget(self._view)

        self.progress.setVisible(True)
        self.run_btn.setEnabled(False)
        self.say(f"deconvolving {subject}, capturing every iteration up to {iterations} …")
        self._worker = _DeconQCWorker(
            dataset, self.region_combo.currentText(), self.fov_spin.value(),
            self.channel_combo.currentText(), iterations, not self.cpu_cb.isChecked(),
            self._crop_half, self._view_half)
        self._worker.done.connect(self._on_done)
        self._worker.sweep_done.connect(self._on_sweep_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_done(self, iterations, frame, ratio) -> None:
        from squidxplorer._decon_qc import halo_verdict

        # The verdict needs this iteration included; show_iteration is what appends it.
        history = list(self._view.history) + [(int(iterations), float(ratio))]
        kind, verdict = halo_verdict(history)
        self._view.show_iteration(iterations, ratio, kind, verdict,
                                  volume=frame.volume, centre=frame.centre,
                                  view_half=frame.view_half,
                                  delta=getattr(frame, "delta", None),
                                  fov_origin=getattr(frame, "fov_origin", None))
        self.say(verdict)

    #: The layer key the sweep's in-view data preview lands under. Not a registered operator
    #: on purpose: `operator_reduces_depth` answers False for an unknown key, so the preview
    #: volume keeps its z and can be opened in 3D like any depth-keeping layer.
    PREVIEW_OP = "decon QC"

    def _preview_view(self):
        """A live view showing the sweep's region, the active one preferred; else None."""
        manager = getattr(self.host, "_viewer_manager", None)
        if manager is None or self._sweep_at is None:
            return None
        region = self._sweep_at[0]
        try:
            candidates = [manager.active_view()] + list(manager.windows())
        except Exception:                      # noqa: BLE001 - a manager double without windows()
            return None
        for win in candidates:
            if win is not None and region in list(getattr(win, "_regions", []) or []):
                return win
        return None

    def _push_view_preview(self, k: int) -> None:
        """Land the DISPLAYED iteration's deconvolved crop in a view, at its true stage
        position, as a real layer — the sweep is judged on actual data at real size (Julio,
        2026-08-24: "see the actual data preview as we sweep"). Stepping the slider swaps the
        layer's pixels; nothing is re-solved."""
        view = self._view
        rec = view.capture(k) if view is not None else None
        if rec is None or rec.get("volume") is None or rec.get("fov_origin") is None \
                or self._sweep_at is None:
            return
        target = self._preview_view()
        if target is None:
            return                             # no view over that region: the panel still shows it
        region, fov, channel = self._sweep_at
        meta = getattr(target, "_meta", None) or {}
        px = meta.get("pixel_size_um")
        mosaic = getattr(getattr(target, "_pane", None), "mosaic", None)
        if mosaic is None or not px:
            return
        try:
            from squidxplorer._mosaic_source import mosaic_fov_bboxes_um
            from squidxplorer._napari_pane import _colormap_for

            box = mosaic_fov_bboxes_um(meta, region).get(int(fov))
            if box is None:
                return
            fx0, fy0 = float(box[0]), float(box[1])
            volume = rec["volume"]
            y0, x0 = (int(v) for v in rec["fov_origin"])
            h, w = int(volume.shape[-2]), int(volume.shape[-1])
            bbox = (fx0 + x0 * float(px), fy0 + y0 * float(px),
                    fx0 + (x0 + w) * float(px), fy0 + (y0 + h) * float(px))
            # The channel's OWN colormap: the preview is a normal data layer, judged like data
            # (Julio, 2026-08-25: "The turbo colormap preview makes no sense. remove it").
            mosaic.add_result("intensity", self.PREVIEW_OP, channel, volume,
                              bbox_um=bbox, z_scale_um=meta.get("dz_um") or None,
                              colormap=_colormap_for(channel, meta.get("channels")))
        except Exception as exc:               # noqa: BLE001 - the preview is a nicety, SAID
            self.say(f"the sweep's data preview could not reach the view: "
                     f"{type(exc).__name__}: {exc}")

    def _on_sweep_done(self, n: int) -> None:
        self.progress.setVisible(False)
        self.run_btn.setEnabled(True)

    def _adopt_iterations(self, k: int) -> None:
        """The QC's chosen count becomes THE run's iterations (read via ``kwargs``)."""
        self.run_iter_spin.setValue(int(k))
        self.say(f"decon runs will use {int(k)} iteration" + ("s" if int(k) != 1 else "")
                 + " - adopted from the QC sweep; a view's Run button reads it off this panel.")

    def kwargs(self) -> dict:
        """The decon run's parameters, read by ``operator_kwargs_for('decon')`` at launch."""
        return {"iterations": int(self.run_iter_spin.value())}

    def set_param(self, name: str, value) -> Optional[str]:
        """Write ONE run parameter into this panel — the run's single source of truth — so a
        view's inline iterations control edits the SAME number the QC's 'use k' button writes.
        Returns a refusal sentence for a name this panel does not hold."""
        if str(name) == "iterations":
            self.run_iter_spin.setValue(int(value))
            return None
        return f"the decon panel holds no parameter {name!r}; it declares ['iterations']."

    def _on_failed(self, message: str) -> None:
        self.progress.setVisible(False)
        self.run_btn.setEnabled(True)
        self.say(f"deconvolution did not run: {message}")

    def shutdown(self) -> None:
        """Join the QC worker before this panel is destroyed. Without this, closing the tab
        mid-run drops the last reference to a running QThread and aborts the interpreter."""
        w = self._worker
        if w is None:
            return
        # Drop the result callbacks first: a done/failed emit landing during teardown would
        # call show_iteration on a deleted widget.
        for sig, slot in ((w.done, self._on_done), (w.sweep_done, self._on_sweep_done),
                          (w.failed, self._on_failed)):
            try:
                sig.disconnect(slot)
            except (TypeError, RuntimeError):
                pass          # not connected, or already gone
        if w.isRunning():
            w.requestInterruption()   # honoured before the expensive RL call begins
            w.wait()                  # block until run() returns — bounded on a crop
        self._worker = None

    def stop(self) -> None:
        self.shutdown()


def _shipped_iterations() -> int:
    from squidxplorer._decon import DEFAULT_ITERATIONS

    return DEFAULT_ITERATIONS
