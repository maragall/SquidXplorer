"""The decon iteration QC: a chained operation whose result carries an iteration axis.

Julio, 2026-09-09, the chain: "Like you're just chaining an operation. You do mip and you
open a new tab. and then you run decon and then you can see how it looks on turbomap."
So there is NO bespoke QC surface: the inspect button runs the decon solve through THE
SAME dispatch path a Preview rides (:func:`squidxplorer._dispatch.run_operator_once`,
capture armed around it - one kwargs contract, one scoping, one stop/verdict machinery),
and the captures are delivered as ORDINARY results into a LAYERS-ONLY child view: per
channel the XY MIP stack as ``(N, Y, X)`` whose depth axis is the iteration count
(``Substance.depth_label = "iteration"``, the one taught fact: unit depth scale, napari's
own bottom slider steps k), and the XZ / YZ bands as two more ordinary results placed
below and beside through their own bbox (the bbox height carries z at dz, no resampling).
Turbo is napari's colormap dropdown; teardown is the child's ordinary dispose.

MEASURED (2026-09-09, the defect build): a ``(N, 1, H, W)`` iteration layer beside a
z-ful raw pyramid shares world axis 1, and napari slices the size-1 axis at the OTHER
layer's z point - permanently out of range, so the QC layers never repainted and every
slider read dead ("iteration slider doesn't work ... sliders don't work"). The
layers-only child (no raw, uniform ``(iteration, y, x)`` dims) removes that class whole.
The bands, measured on G7 488 (256 px window, 15 z, 3 iterations): the full-other-axis
max under its OWN window reads cleanly - only 1.3% of band pixels sit above the XY MIP's
1/99.9 window, so the saturation Julio saw WAS the XY-seeded window, not the projection -
while a central slab (1/3, or 32 px) discards most structures. The projection therefore
stays the full max; each band layer seeds contrast from its own pixels (ordinary
delivery does that) and leaves the per-channel contrast link
(:meth:`~squidxplorer._napari_view.MosaicLayers.isolate_contrast`); bands render with
linear interpolation (15 rows stretched ~9x by dz read blocky under nearest) and keep
the app's additive blending, the multi-channel standard everywhere else.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from qtpy.QtCore import QThread, Signal

from squidxplorer._logpane import get_logger

log = get_logger("decon.qc")

#: The three delivered identities. Separate ops ON PURPOSE: one identity would mirror the
#: XY contrast onto the bands, which is the measured saturation defect.
QC_XY = "decon QC"
QC_XZ = "decon QC xz"
QC_YZ = "decon QC yz"

#: Gap between the MIP and its bands, in acquisition pixels.
GAP_PX = 8


class QCWorker(QThread):
    """ONE scoped decon run through the ordinary dispatch, capture armed around it."""

    done = Signal()
    failed = Signal(str)

    def __init__(self, reader, region, fov, window, parameters, parent=None):
        super().__init__(parent)
        self._args = (reader, str(region), int(fov), window, dict(parameters or {}))

    def run(self):  # pragma: no cover - thread body; the sync path is tested directly
        try:
            run_qc_solve(*self._args)
            self.done.emit()
        except Exception as exc:                 # noqa: BLE001 - reported as a sentence
            self.failed.emit(f"{type(exc).__name__}: {exc}")


def run_qc_solve(reader, region, fov, window, parameters) -> None:
    """THE fold (Julio: "preview iteratinos vs preview it looks like logic is being
    duplicated"): the QC solve IS a Preview-shaped run through ``run_operator_once`` -
    same kwargs contract, same scoping, same verdict - with capture armed around it.
    Every timepoint runs, exactly as a Preview's full solve does."""
    from squidxplorer._decon import arm_capture, clear_captures
    from squidxplorer._dispatch import run_operator_once

    clear_captures()
    arm_capture(True)
    try:
        run_operator_once(
            reader, operator="decon", save=False, owed=1,
            regions={str(region): [int(fov)]},
            windows=({(str(region), int(fov)): tuple(int(v) for v in window)}
                     if window else None),
            workers=1, parameters=dict(parameters or {}), tiff=False)
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


def deliver_qc(view, caps: dict, region: str, fov: int, window):
    """Deliver the captures as ordinary results into a layers-only child tab.

    Three results per the module docstring: the XY MIP stack at the solved box, the XZ
    band below it, the YZ band beside it (transposed so y aligns), every one through
    ``deliver_result`` with ``depth_label="iteration"``. Returns the child, or ``None``
    said to the asking view.
    """
    from squidxplorer.projection import cast_like
    from squidxplorer._result import Extent, Result, Substance

    manager = getattr(view, "_manager", None)
    meta = view._meta or {}
    ks = _capture_ks(caps)
    if manager is None or not ks:
        view._say("iteration QC: nothing to open (no manager or no captures).")
        return None
    child = manager.open_child([region], parent_id=view.window_id, layers_only=True,
                               title=f"decon QC {region}")
    pane = getattr(child, "_pane", None) if child is not None else None
    mosaic = getattr(pane, "mosaic", None) if pane is not None else None
    if mosaic is None:
        view._say("iteration QC: the QC tab could not open a viewer pane.")
        return None

    channels = sorted(caps)
    dtype = str(meta["dtype"])
    px = float(meta["pixel_size_um"])
    dz = float(meta.get("dz_um") or px)
    x0, y0, x1, y1 = qc_bbox_um(meta, region, fov, window)
    gap = GAP_PX * px
    nz = len(next(iter(caps.values()))[ks[0]]["xz"])
    z_um = nz * dz

    # THE DISPLAY NORMALIZATION (Julio, 2026-09-09: "intesity grows with the
    # iterations"): RL concentrates flux, so per-iteration maxima grow with k (measured
    # on G7 488: XY p99.5 31250 -> 35759 and max 55852 -> 78365 over k=1..3) while the
    # layer holds one contrast window - stepping read as global brightening. Each k's
    # THREE projections are scaled by ONE scalar per (k, channel): the stack's smallest
    # XY 99.5th percentile over that k's own, so apparent brightness holds constant
    # across the step, the raw input (k=0) anchors the scheme, XY/XZ/YZ stay mutually
    # consistent, and nothing is scaled UP into the cast ceiling. DISPLAY ONLY: these
    # captures are a QC instrument, not a quantitative result - the capture store keeps
    # the raw planes untouched.
    factors: "dict[str, dict[int, float]]" = {}
    for c in channels:
        p = {k: float(np.percentile(caps[c][k]["xy"], 99.5)) for k in ks}
        target = min(p.values())
        factors[c] = {k: target / max(p[k], 1e-12) for k in ks}
    log.info("iteration display normalized per step to its own 99.5th percentile "
             "(display only; the captures stay raw)")

    def _stack(name: str, transpose: bool = False):
        out = []
        for c in channels:
            planes = [(caps[c][k][name].T if transpose else caps[c][k][name])
                      * factors[c][k] for k in ks]
            out.append(cast_like(np.stack(planes), np.dtype(dtype)))
        return out

    def _result(data, bbox) -> Result:
        return Result(
            extent=Extent(region_id=str(region), fovs=(int(fov),), bbox_um=tuple(bbox)),
            substance=Substance(channels=tuple(channels), z_depth=len(ks), dtype=dtype,
                                pixel_size_um=px, kind="intensity",
                                depth_label="iteration"),
            data=data)

    child.deliver_result(QC_XY, _result(_stack("xy"), (x0, y0, x1, y1)), visible=True)
    child.deliver_result(QC_XZ, _result(_stack("xz"), (x0, y1 + gap, x1, y1 + gap + z_um)),
                         visible=True)
    child.deliver_result(QC_YZ, _result(_stack("yz", transpose=True),
                                        (x1 + gap, y0, x1 + gap + z_um, y1)),
                         visible=True)

    # Ordinary delivery lights ONE op per channel (the exclusive-op rule reads each
    # arrival as a gesture); a QC tab shows all three panels of a channel together.
    with mosaic.programmatic():
        for op in (QC_XY, QC_XZ, QC_YZ):
            for channel in channels:
                for ly in mosaic.layers_for(op, channel):
                    ly.visible = True
    # The bands: their own contrast (out of the channel link) and a continuous stretch.
    for op in (QC_XZ, QC_YZ):
        mosaic.isolate_contrast(op)
        for channel in channels:
            for ly in mosaic.layers_for(op, channel):
                try:
                    ly.interpolation2d = "linear"
                except Exception:                # noqa: BLE001 - a render hint, never fatal
                    pass
    model = mosaic.model
    labels = list(model.dims.axis_labels)
    if labels:
        labels[0] = "iteration"
        model.dims.axis_labels = tuple(labels)
    model.dims.set_current_step(0, len(ks) - 1)  # open on the final iteration
    fit = getattr(mosaic, "reset_view", None)
    if callable(fit):
        fit()
    return child


def start_qc(panel, view) -> None:
    """Run the QC solve off-thread and deliver the chained result when it lands.

    *panel* is the DeconPanel (owns the worker; its ``shutdown`` joins it); *view* is
    the asking RegionViewer. The store is TAKEN whole into the tab's layers, so it never
    outlives the run, and the tab's ordinary dispose frees the pixels.
    """
    from squidxplorer._decon import take_captures

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
    except Exception as exc:                     # noqa: BLE001 - a refusal, said not raised
        panel.say(f"iteration QC: {exc}")
        return
    # THE kwargs read Preview uses (operator_kwargs_for, the ONE reader of the live
    # panel): never this button's own instance, which can be stale while the plate
    # files a newer panel (Julio, 2026-09-09: "I set 4 iterations and it only did two").
    reader_kwargs = getattr(getattr(panel, "host", None), "operator_kwargs_for", None)
    try:
        parameters = dict(reader_kwargs("decon") or {}) if callable(reader_kwargs) else {}
    except ValueError as exc:                    # a refused setting: said, never defaults
        panel.say(f"iteration QC: {exc}")
        return
    if not parameters:                           # no plate, no filed panel: the button's own
        parameters = dict(panel.kwargs() or {})
    from squidxplorer._decon import DEFAULT_ITERATIONS

    what = (f"{region} field {fov}"
            + (f", {window[3] - window[2]}x{window[1] - window[0]} px window" if window
               else ", whole field"))
    log.info("iteration QC: solving %s at %s iteration(s) plus the raw anchor, every "
             "channel, capturing each iteration", what,
             parameters.get("iterations", DEFAULT_ITERATIONS))

    def _land():
        panel.inspect_btn.setEnabled(True)
        caps = take_captures()
        if not caps:
            panel.say("iteration QC: the solve captured nothing "
                      "(every channel copied through?).")
            return
        try:
            deliver_qc(view, caps, region, fov, window)
        except Exception as exc:                 # noqa: BLE001 - named, never a hang
            panel.say(f"iteration QC: the tab could not open: {exc}")

    def _fail(msg: str) -> None:
        panel.inspect_btn.setEnabled(True)
        panel.say(f"iteration QC: {msg}")

    qc = QCWorker(reader, region, fov, window, parameters, parent=panel)
    qc.done.connect(_land)
    qc.failed.connect(_fail)
    panel._qc_worker = qc
    panel.inspect_btn.setEnabled(False)
    qc.start()
