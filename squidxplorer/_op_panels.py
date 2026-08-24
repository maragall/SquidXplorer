"""The two HAND-WRITTEN operator interfaces: the stitcher's controls and decon's QC loop.

Every other operator's panel is built from its ``params`` declaration by
:mod:`squidxplorer._param_panel`. These two stay hand-written because they do things a
parameter form cannot: :class:`StitcherPanel` converts a percentage to a fraction, greys out
knobs that do nothing with registration off, and refuses a labels operator before the run
starts; :class:`DeconQCPanel` runs an iterative semi-convergence loop and publishes a picture
as a tab of its own. ``_viewer._activate_operator`` prefers a hand-written panel and falls
back to the generic one.

This module never touches a tab bar; it calls ``host.publish_qc_result(widget, title)``.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np
from qtpy.QtCore import Qt, QThread, Signal
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import (
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
        self._check_z_operator(self.z_operator_combo.currentText())

    def _on_register_toggled(self, on: bool) -> None:
        """Grey out the knobs that provably do nothing with registration off."""
        for w in (self.reg_channel_combo, self.rel_spin, self.abs_spin,
                  self.reg_t_spin, self.distortion_cb):
            w.setEnabled(bool(on))

    def _check_z_operator(self, name: str) -> None:
        why = None if name == KEEP_EVERY_PLANE else stitch_refusal(name)
        self.run_btn.setEnabled(why is None)
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
                f"independently — the pose graph is solved ONCE and every plane is fused from "
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

    def _run(self) -> None:
        chosen = self.z_operator_combo.currentText()
        why = None if chosen == KEEP_EVERY_PLANE else stitch_refusal(chosen)
        if why is not None:
            self.say(why)
            return
        try:
            kwargs = self.kwargs()
        except ValueError as exc:                 # a refused setting -> say it, run nothing
            self.say(str(exc))
            return
        # The label is the combo's spelling of z_operator=None: every acquired plane, unchanged.
        kwargs["z_operator"] = None if chosen == KEEP_EVERY_PLANE else chosen
        self.say("")
        # regions=None means UNSCOPED, resolved against the run selector's live selection.
        self.host.run_operator("stitch", regions=None,
                               save=self.save_cb.isChecked(), operator_kwargs=kwargs)


class QCFrame(NamedTuple):
    """One finished QC run, as the panel receives it. The volume travels with the composite
    because the picture is clickable: re-sectioning is qc_composite again on this same array."""

    composite: object          # (H, W) float, what qc_composite returned at `centre`
    volume: object             # (Z, Y, X) the deconvolved crop the sections were cut from
    centre: tuple              # (z, y, x) in `volume`: the brightest structure RL was judged at
    view_half: object          # the lateral half-width the composite was cut to, or None


class _DeconQCWorker(QThread):
    """Run RL at ONE iteration count on ONE FOV's z-stack and measure the halo (threaded)."""

    done = Signal(int, object, float)        # (iterations, QCFrame, halo/core ratio)
    failed = Signal(str)

    def __init__(self, dataset, region, fov, channel, iterations, gpu, crop_half, view_half):
        super().__init__()
        self._args = (dataset, region, fov, channel, iterations, gpu, crop_half, view_half)

    def run(self):
        dataset, region, fov, channel, iterations, gpu, crop_half, view_half = self._args
        if self.isInterruptionRequested():
            return          # a close/shutdown beat us to the start
        try:
            from squidxplorer._decon import OpticsParams, _run, make_psf
            from squidxplorer._decon_qc import (
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
                           QCFrame(qc_composite(volume, centre, view_half=view_half),
                                   volume, tuple(int(v) for v in centre), view_half),
                           float(ratio))
        except Exception as exc:                  # reported as a sentence, never swallowed
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class _ClickableImage(QLabel):
    """A QLabel showing an unscaled pixmap; reports clicks in pixmap pixels, compensating
    for the centring offset a scroll area introduces."""

    clicked = Signal(int, int)             # (row, col) in the pixmap's own pixels

    def mousePressEvent(self, event):      # noqa: N802 (Qt's spelling)
        pm = self.pixmap()
        if pm is not None and not pm.isNull():
            pos = event.pos()
            dx = max((self.width() - pm.width()) // 2, 0)
            dy = max((self.height() - pm.height()) // 2, 0)
            col, row = pos.x() - dx, pos.y() - dy
            if 0 <= col < pm.width() and 0 <= row < pm.height():
                self.clicked.emit(int(row), int(col))
        super().mousePressEvent(event)


class DeconQCResultView(QWidget):
    """The deconvolved 2-D image in turbo with the x-z and y-z strips attached. Renders what
    :func:`squidxplorer._decon_qc.qc_composite` / ``turbo_rgb`` produced. Clicking the image
    re-sections the same in-memory volume through the clicked point; no RL run happens."""

    def __init__(self, subject: str):
        super().__init__()
        self.setStyleSheet(f"background:{_BG};color:#e6edf3;")
        self.history: list = []
        # The volume behind the current picture, kept so a click can re-section it.
        self._volume = None
        self._centre = None
        self._view_half = None
        self._gap = 2
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

        self.crosshair_label = QLabel("")
        self.crosshair_label.setWordWrap(True)
        self.crosshair_label.setStyleSheet(_SUB)
        v.addWidget(self.crosshair_label)

        self.image_label = _ClickableImage()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setCursor(Qt.CrossCursor)
        self.image_label.setToolTip(
            "Click anywhere in the x-y plane, the x-z strip or the y-z strip to move the "
            "crosshairs there. The sections are re-cut from the same deconvolved volume; "
            "nothing is re-run.")
        self.image_label.clicked.connect(self._on_image_clicked)
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
                       kind: str, verdict: str, volume=None, centre=None,
                       view_half=None, gap: int = 2) -> None:
        """Display one iteration's composite and remember it, so the loop can be compared.
        Passing volume/centre/view_half makes the picture clickable; omitting them leaves it
        static (the older three-argument call shape)."""
        self._volume = None if volume is None else np.asarray(volume)
        self._centre = None if centre is None else tuple(int(v) for v in centre)
        self._view_half = view_half
        self._gap = int(gap)
        self._paint(composite)
        self.history.append((int(iterations), float(ratio)))
        self.caption_label.setText(
            f"{iterations} iteration" + ("s" if iterations != 1 else "")
            + f"  ·  halo/core {ratio:.3f}")
        self.verdict_label.setText(verdict)
        self.trail_label.setText("  ".join(f"k={k}: {r:.3f}" for k, r in self.history))
        self._sync_crosshair_label()

    def _paint(self, composite) -> None:
        """The only place a pixmap is set, so the first paint and every crosshair move agree."""
        from squidxplorer._decon_qc import turbo_rgb

        rgb = np.ascontiguousarray(turbo_rgb(composite))
        h, w, _ = rgb.shape
        image = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self.image_label.setPixmap(QPixmap.fromImage(image))
        self.image_label.setMinimumSize(w, h)
        self._rgb = rgb                       # keep the buffer alive alongside the pixmap

    def _on_image_clicked(self, row: int, col: int) -> None:
        """Move the crosshairs to the clicked voxel and re-cut the three sections there (a
        re-slice of the in-memory volume, no RL run)."""
        if self._volume is None or self._centre is None:
            return
        from squidxplorer._decon_qc import composite_centre_at, qc_composite

        centre = composite_centre_at(self._volume.shape, self._centre, row, col,
                                     view_half=self._view_half, gap=self._gap)
        if centre is None or centre == self._centre:
            return
        self._centre = centre
        self._paint(qc_composite(self._volume, centre, view_half=self._view_half,
                                 gap=self._gap))
        self._sync_crosshair_label(moved=True)

    def _sync_crosshair_label(self, moved: bool = False) -> None:
        if self._centre is None:
            self.crosshair_label.setText(
                "" if self._volume is None else "crosshairs: unknown")
            return
        z, y, x = self._centre
        where = f"crosshairs at z={z}, y={y}, x={x}"
        self.crosshair_label.setText(
            where + ("  ·  moved by hand; the halo/core number above was measured at the "
                     "structure the run picked, not here." if moved else
                     "  ·  the brightest structure the run found. Click the picture to "
                     "section somewhere else."))


class DeconQCPanel(_Panel):
    """Pick an iteration count, run it, judge the picture in the tab beside this, add one more."""

    def __init__(self, host):
        super().__init__(
            host, "Deconvolution (Richardson-Lucy)",
            "Richardson-Lucy is SEMI-CONVERGENT: the halo tightens for a few iterations and "
            "then a disc around the core starts growing back as the algorithm fits noise. "
            "There is no universally correct count, so run one, look at the turbo x-z / y-z "
            "view in the tab it opens, then add ONE more and look again.")
        from squidxplorer._decon import QC_START_ITERATIONS
        from squidxplorer._decon_qc import DEFAULT_CROP_HALF, DEFAULT_VIEW_HALF

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

        note = _wrapped("The result opens as a tab beside this one. Nothing is written next to "
                        "the acquisition; the datasets are opened read only.", _SUB)
        self.v.addWidget(note)
        self.v.addStretch(1)
        _apply_qss(self)

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
                     "publish_qc_result(widget, title), which is how it is handed a result "
                     "tab. Refusing to deconvolve and then drop the picture.")
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

    def _on_done(self, iterations, frame, ratio) -> None:
        from squidxplorer._decon_qc import halo_verdict

        self.progress.setVisible(False)
        self.run_btn.setEnabled(True)
        # The verdict needs this iteration included; show_iteration is what appends it.
        history = list(self._view.history) + [(int(iterations), float(ratio))]
        kind, verdict = halo_verdict(history)
        self._view.show_iteration(iterations, frame.composite, ratio, kind, verdict,
                                  volume=frame.volume, centre=frame.centre,
                                  view_half=frame.view_half)
        self.say(verdict)

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
        for sig, slot in ((w.done, self._on_done), (w.failed, self._on_failed)):
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
