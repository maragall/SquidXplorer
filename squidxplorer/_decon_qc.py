"""The decon iteration QC: tri-MIP layers in a DECK TAB, iteration as a dims axis.

Julio, 2026-09-09, verbatim: "For looking at iterations we look at the mip only. So a mip
is part of the computation. ... But the MIP is more than enough for QC." And the
correction that shaped this module: "It should be a tab in the napari GUI. You leverage
napari layers, decon sliders, the mip is just a 2d view of the ROI. You're not leveraging
the GUI capabilities and you added some sepparate window code."

So the QC surface RIDES THE APP: the inspect button runs one Preview-scoped solve
capturing every iteration's three max-projections (XY, XZ, YZ), then opens an ROI child
view through the ordinary ``open_child`` deck machinery and adds the captures as REAL
napari layers with the iteration count as dims AXIS 0 - napari's own bottom slider steps
k for every layer at once (its Dims owns the index; a second wrapped slider would be the
two-owner hand-sync ``_fov_nav``'s notes exist to forbid, so the native slider won). Per
channel the three projections are three layers ADOPTED under ONE identity, so the layer
tree's channel checkbox, the contrast controls and the colormap dropdown (turbo lives
there) drive all three through the identity mirror; contrast is seeded once from the
final iteration's window and napari owns it after that. The bands are placed by layer
``translate`` and z-scaled by dz through their own ``scale`` - no resampling: the YZ band
sits beside the XY MIP (transposed so y aligns), the XZ band below it. One small bottom
bar carries "use iteration k", which writes the decon panel's own spin.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from qtpy.QtCore import QThread, Signal

from squidxplorer._logpane import get_logger

log = get_logger("decon.qc")

#: The identity every QC layer is adopted under (three layers per channel, one identity).
QC_OP = "decon QC"

#: The contrast seed: percentiles of the FINAL iteration's XY MIP, per channel, set once
#: as add-time contrast_limits; napari's own controls take over from there.
CONTRAST_PCT = (1.0, 99.9)

#: Gap between the MIP and its bands, in acquisition pixels.
GAP_PX = 8


class QCWorker(QThread):
    """ONE scoped decon solve with capture armed; the engine's own project_well does the
    window + halo + per-channel binding and lands the tri-MIPs in the store."""

    done = Signal()
    failed = Signal(str)

    def __init__(self, reader, region, fov, window, op, time_point, parent=None):
        super().__init__(parent)
        self._args = (reader, str(region), int(fov), window, op, int(time_point))

    def run(self):  # pragma: no cover - thread body; the sync path is tested directly
        try:
            run_qc_solve(*self._args)
            self.done.emit()
        except Exception as exc:                 # noqa: BLE001 - reported as a sentence
            self.failed.emit(f"{type(exc).__name__}: {exc}")


def run_qc_solve(reader, region, fov, window, op, time_point) -> None:
    """The QC solve, synchronous: arm capture around ONE project_well of the scoped field."""
    from squidxplorer._decon import arm_capture, clear_captures
    from squidxplorer.projection import project_well

    clear_captures()
    arm_capture(True)
    try:
        project_well(reader, region, fov, op, time_point=time_point, window=window)
    finally:
        arm_capture(False)


def qc_scope(view) -> "tuple[str, int, Optional[tuple]]":
    """The QC solve's field, scoped EXACTLY like a Preview: the ROI's first windowed
    field when a box is drawn, else the in-view region's CENTRE field, whole."""
    regions, windows = view._run_scope()
    if windows:
        (region, fov), window = sorted(windows.items())[0]
        return str(region), int(fov), tuple(window)
    region = view.current_region()
    if not region:
        raise ValueError("no region is in view.")
    fovs = list(((view._meta or {}).get("fovs_per_region") or {}).get(region) or [])
    if not fovs:
        raise ValueError(f"region {region!r} has no fields.")
    return str(region), int(fovs[len(fovs) // 2]), None


def qc_bbox_um(meta: dict, region: str, fov: int, window) -> "tuple[float, float, float, float]":
    """The solved pixels' stage box: the field's own box, cut to the window when one ran."""
    from squidxplorer._mosaic_source import mosaic_fov_bboxes_um

    x0, y0, x1, y1 = mosaic_fov_bboxes_um(meta, region)[int(fov)]
    if window is None:
        return (float(x0), float(y0), float(x1), float(y1))
    r0, r1, c0, c1 = (int(v) for v in window)
    px = float(meta["pixel_size_um"])
    return (float(x0) + c0 * px, float(y0) + r0 * px,
            float(x0) + c1 * px, float(y0) + r1 * px)


def _capture_ks(caps: dict) -> list:
    return sorted(set.intersection(*(set(b) for b in caps.values()))) if caps else []


def open_qc_tab(view, caps: dict, region: str, fov: int, window,
                on_use: "Optional[Callable[[int], None]]" = None):
    """Open the QC deck tab: an ROI child over the solved box, the tri-MIPs as layers.

    Every layer is ADOPTED under ``(QC_OP, channel)`` so the identity rules hold; the
    iteration count is dims axis 0 (each layer is ``(N, 1, H, W)``, so the raw crop's own
    z stays a separate axis); ``show_op`` lights the QC layers and dims raw, and the tree
    turns raw back on if wanted. Returns the child view, or ``None`` said to the view.
    """
    manager = getattr(view, "_manager", None)
    meta = view._meta or {}
    ks = _capture_ks(caps)
    if manager is None or not ks:
        view._say("iteration QC: nothing to open (no manager or no captures).")
        return None
    bbox = qc_bbox_um(meta, region, fov, window)
    child = manager.open_child([region], roi_bbox=bbox, parent_id=view.window_id)
    pane = getattr(child, "_pane", None) if child is not None else None
    mosaic = getattr(pane, "mosaic", None) if pane is not None else None
    if mosaic is None:
        view._say("iteration QC: the QC tab could not open a viewer pane.")
        return None

    from squidxplorer._napari_pane import _colormap_for

    model = mosaic.model
    px = float(meta["pixel_size_um"])
    dz = float(meta.get("dz_um") or px)
    x0, y0, x1, y1 = bbox
    gap = GAP_PX * px
    for channel in sorted(caps):
        by_k = caps[channel]
        # Axis 0 is the ITERATION; the size-1 axis 1 keeps it apart from the raw crop's z.
        xy = np.stack([by_k[k]["xy"] for k in ks])[:, None]
        xz = np.stack([by_k[k]["xz"] for k in ks])[:, None]
        yz = np.stack([np.ascontiguousarray(by_k[k]["yz"].T) for k in ks])[:, None]
        lo, hi = np.percentile(by_k[ks[-1]]["xy"], CONTRAST_PCT)
        colormap = _colormap_for(channel, meta.get("channels"))
        specs = (
            ("xy", xy, (1.0, 1.0, px, px), (0.0, 0.0, y0, x0)),
            ("xz", xz, (1.0, 1.0, dz, px), (0.0, 0.0, y1 + gap, x0)),   # the band below
            ("yz", yz, (1.0, 1.0, px, dz), (0.0, 0.0, y0, x1 + gap)),   # the band beside
        )
        for tag, data, scale, translate in specs:
            layer = model.add_image(
                data, name=f"{QC_OP} {tag} [{channel}]",
                scale=scale, translate=translate,
                colormap=colormap, blending="additive",
                contrast_limits=(float(lo), float(hi)),
            )
            mosaic.adopt(QC_OP, channel, layer)
    try:
        mosaic.show_op(QC_OP)                    # the QC layers lead; the tree offers raw back
    except Exception:                            # noqa: BLE001 - a look, never a failed open
        pass
    labels = list(model.dims.axis_labels)
    if labels:
        labels[0] = "iteration"
        model.dims.axis_labels = tuple(labels)
    _attach_use_bar(child, model, ks, on_use)
    model.dims.set_current_step(0, len(ks) - 1)  # open on the final iteration
    fit = getattr(mosaic, "reset_view", None)
    if callable(fit):
        fit()
    return child


def _attach_use_bar(child, model, ks: list, on_use) -> None:
    """One small bottom bar in the QC tab: the shown k, and 'use iteration k' writing the
    decon panel's spin through *on_use*. The k is READ off napari's own Dims (its slider
    owns the index; this bar holds no copy)."""
    from qtpy.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

    from squidxplorer import _qtstyle

    bar = QWidget()
    row = QHBoxLayout(bar)
    row.setContentsMargins(8, 2, 8, 2)
    row.setSpacing(8)
    label = QLabel("")
    label.setStyleSheet("color:#c9d1d9;font-size:12px;")
    button = QPushButton("")
    button.setStyleSheet(_qtstyle.BTN_QSS)
    button.setToolTip(
        "Adopt the shown iteration count as the decon run's own parameter; Preview then "
        "runs it on the whole stack.")

    def _k() -> int:
        i = min(max(int(model.dims.current_step[0]), 0), len(ks) - 1)
        return int(ks[i])

    def _refresh(*_a) -> None:
        k = _k()
        label.setText(f"iteration {k} of {ks[-1]}")
        button.setText(f"use iteration {k}")

    model.dims.events.current_step.connect(_refresh)
    if on_use is not None:
        button.clicked.connect(lambda *_: on_use(_k()))
    row.addWidget(label, 1)
    row.addWidget(button)
    layout = child.centralWidget().layout() if child.centralWidget() is not None else None
    if layout is not None:
        layout.addWidget(bar)
    child._qc_use_bar = bar                      # the tests' handle; dies with the tab
    child._qc_use_button = button
    _refresh()


def start_qc(panel, view) -> None:
    """Run the QC solve off-thread and open the QC tab when its captures land.

    *panel* is the DeconPanel (owns the worker; its ``shutdown`` joins it); *view* is the
    asking RegionViewer. The store is TAKEN whole into the tab's layers, so it never
    outlives the run, and the tab's ordinary dispose frees the pixels.
    """
    from squidxplorer._decon import take_captures
    from squidxplorer._engine import bind_operator

    worker = getattr(panel, "_qc_worker", None)
    if worker is not None and worker.isRunning():
        panel.say("iteration QC: a solve is already running.")
        return
    reader, meta = getattr(view, "_reader", None), getattr(view, "_meta", None)
    if reader is None or not meta:
        panel.say("iteration QC: open an acquisition first.")
        return
    try:
        region, fov, window = qc_scope(view)
        kwargs = {n: v for n, v in (panel.kwargs() or {}).items() if n == "iterations"}
        op = bind_operator("decon", kwargs)
    except Exception as exc:                     # noqa: BLE001 - a refusal, said not raised
        panel.say(f"iteration QC: {exc}")
        return
    what = (f"{region} field {fov}"
            + (f", {window[3] - window[2]}x{window[1] - window[0]} px window" if window
               else ", whole field"))
    log.info("iteration QC: solving %s, every channel, capturing each iteration", what)

    def _adopt(k: int) -> None:
        refusal = panel.set_param("iterations", int(k))
        panel.say(refusal or f"decon iterations set to {k}.")

    def _land():
        panel.inspect_btn.setEnabled(True)
        caps = take_captures()
        if not caps:
            panel.say("iteration QC: the solve captured nothing "
                      "(every channel copied through?).")
            return
        try:
            open_qc_tab(view, caps, region, fov, window, on_use=_adopt)
        except Exception as exc:                 # noqa: BLE001 - named, never a hang
            panel.say(f"iteration QC: the tab could not open: {exc}")

    def _fail(msg: str) -> None:
        panel.inspect_btn.setEnabled(True)
        panel.say(f"iteration QC: {msg}")

    qc = QCWorker(reader, region, fov, window, op,
                  int(getattr(view, "time_point", 0) or 0), parent=panel)
    qc.done.connect(_land)
    qc.failed.connect(_fail)
    panel._qc_worker = qc
    panel.inspect_btn.setEnabled(False)
    qc.start()
