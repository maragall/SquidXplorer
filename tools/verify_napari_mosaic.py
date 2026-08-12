"""Open a real acquisition, open a region window on it, and verify napari painted a real mosaic.

Deliberately not offscreen: the offscreen Qt plugin has no OpenGL, so napari's canvas cannot
exist there (it segfaults rather than raising).

    python tools/verify_napari_mosaic.py ~/Downloads/sim_2x2_36fov_96wp
"""

from __future__ import annotations

import json
import os
import sys
import time

os.environ.pop("QT_QPA_PLATFORM", None)          # need a real GL context
os.environ.setdefault("SQUIDXPLORER_VIEWER", "napari")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# import squidxplorer before qtpy: it pins QT_API=pyqt6, which must happen before qtpy resolves
# a binding.
import squidxplorer  # noqa: F401
from qtpy.QtWidgets import QApplication

from squidxplorer._viewer import PlateWindow

if len(sys.argv) < 2:
    print("usage: python tools/verify_napari_mosaic.py <acquisition dir> [budget_s]")
    raise SystemExit(2)
path = sys.argv[1]
if not os.path.isdir(path):
    print("VERIFY " + json.dumps({"skipped": f"dataset absent on this machine: {path}"}))
    raise SystemExit(0)
budget = float(sys.argv[2]) if len(sys.argv) > 2 else 240.0

app = QApplication.instance() or QApplication([])
win = PlateWindow()
win.resize(1600, 900)
win.show()
app.processEvents()

out: dict = {"dataset": os.path.basename(path.rstrip("/"))}


# The checks this script runs, keyed as they appear in `out`; a key absent or None is unmeasured
# rather than failed (e.g. `layers` may legitimately be empty on a dataset with no mosaic).
VERDICT = {
    "pane_is_napari": "the region window fell back: napari did not build a canvas",
    "canvas_painted_pixels": "the canvas framebuffer is blank or uniform: no pixels were painted",
    "is_a_mosaic_not_one_fov": "the layer covers no more stage than a single FOV: not a mosaic",
    "placed_in_stage_um": "the layer is at scale 1.0: it was not placed in stage micrometres",
    "z_slider_present": "the viewer has no z axis, so the stack cannot be navigated",
    "plate_follows_napari_contrast": "dragging napari's contrast did not repaint the plate "
                                     "(IMA-261)",
}


def _verdict():
    """(exit code, [failure sentences]) read off *out*, never off the intent."""
    checks = dict(VERDICT)
    if int(out.get("n_z_in_meta") or 1) <= 1:
        checks.pop("z_slider_present", None)     # single-plane acquisition owes no slider
    bad = [why for key, why in checks.items() if out.get(key) is False]
    return (1 if bad else 0), bad


def report(code=None):
    if code is None:
        code, bad = _verdict()
        out["failed_checks"] = bad
        out["verdict"] = "PASS" if code == 0 else "FAIL"
    print("VERIFY " + json.dumps(out, indent=1))
    for why in out.get("failed_checks") or []:
        print("VERIFY FAIL: " + why)
    sys.stdout.flush()
    # os._exit: unwinding napari + Qt at interpreter shutdown segfaults on this machine
    os._exit(code)


t0 = time.perf_counter()
win.ingest(path)
app.processEvents()
if win._reader is None:
    out["failure"] = f"ingest failed: {win._readout.text()!r}"
    report(1)
out["regions"] = list((win._meta or {}).get("regions", []))

# ViewerManager.open is the same call the plate's double-click makes.
region = out["regions"][0]
rv = win._viewer_manager.open([region])
if rv is None:
    out["failure"] = f"ViewerManager.open([{region!r}]) returned None"
    report(1)
app.processEvents()
out["region_window"] = region

pane = getattr(rv, "_pane", None)
out["pane_is_napari"] = bool(pane is not None and getattr(pane, "ok", False))
if not out["pane_is_napari"]:
    out["failure"] = getattr(pane, "failure", None) if pane else "the region window has no pane"
    report(1)

# The fuse runs in a worker; pump the loop until layers land or the budget runs out.
while time.perf_counter() - t0 < budget:
    app.processEvents()
    worker = getattr(rv, "_worker", None)
    if pane.mosaic.ops() and not (worker is not None and worker.isRunning()):
        break
    time.sleep(0.05)

app.processEvents()
out["ingest_and_mosaic_s"] = round(time.perf_counter() - t0, 1)
out["ops"] = pane.mosaic.ops()
out["channels"] = pane.mosaic.channels("raw") if "raw" in pane.mosaic.ops() else []

frame = tuple(int(v) for v in (win._meta or {}).get("frame_shape", (0, 0)))
out["frame_shape"] = list(frame)

layers = []
for ly in pane.mosaic.ours():
    from squidxplorer._napari_view import key_of

    k = key_of(ly)
    layers.append({
        "op": k.op, "channel": k.channel,
        "shape": [int(v) for v in np.asarray(ly.data).shape[-2:]],
        "scale_um_per_px": [round(float(s), 4) for s in ly.scale],
        "translate_um": [round(float(t), 2) for t in ly.translate],
        "contrast": [round(float(c), 1) for c in ly.contrast_limits],
        "visible": bool(ly.visible),
    })
out["layers"] = layers

out["dims_ndim"] = int(pane.mosaic.model.dims.ndim)
out["dims_not_displayed"] = [int(a) for a in pane.mosaic.model.dims.not_displayed]
out["z_slider_present"] = out["dims_ndim"] > 2
out["n_z_in_meta"] = int((win._meta or {}).get("n_z") or 1)
out["blending"] = sorted({str(l.blending) for l in pane.mosaic.ours()})
out["colormaps"] = [str(l.colormap.name) for l in pane.mosaic.ours()]
out["all_channels_visible"] = all(l.visible for l in pane.mosaic.ours())
# Count real widgets, not row-tuple slots: the row's second element is now the read-only
# window readout.
from qtpy.QtWidgets import QPushButton, QSlider, QWidget
_plate = win._overview
out["plate_contrast_sliders"] = len(_plate.findChildren(QSlider)) if _plate is not None else 0
out["plate_auto_buttons"] = len(
    [b for b in _plate.findChildren(QPushButton) if b.text() == "auto"]) \
    if _plate is not None else 0
_names = {type(c).__name__ for c in pane.findChildren(QWidget)}
out["napari_layer_controls_mounted"] = "QtLayerControlsContainer" in _names
out["napari_dims_slider_mounted"] = any("QtDim" in n for n in _names)
out["napari_viewer_buttons_present"] = any(
    "ViewerButtons" in type(c).__name__ for c in pane.findChildren(QWidget))

if layers:
    # In micrometres, not pixels: the mosaic is a decimated multiscale pyramid, so its level-0
    # array can be smaller in pixels than one native FOV while covering far more stage.
    ext = np.asarray(pane.mosaic.ours()[0].extent.world, dtype=float)
    span_um = (ext[1] - ext[0])[-2:]
    fov_um = [f * float((win._meta or {}).get("pixel_size_um") or 0.0) for f in frame]
    out["mosaic_span_um"] = [round(float(v), 1) for v in span_um]
    out["one_fov_um"] = [round(float(v), 1) for v in fov_um]
    out["is_a_mosaic_not_one_fov"] = bool(
        all(fov_um) and (span_um[0] > fov_um[0] or span_um[1] > fov_um[1]))
    out["placed_in_stage_um"] = bool(any(s != 1.0 for s in layers[0]["scale_um_per_px"]))

try:
    img = pane.canvas.screenshot()
    arr = np.asarray(img)[..., :3]
    out["screenshot_shape"] = list(arr.shape)
    out["screenshot_distinct_values"] = int(len(np.unique(arr)))
    out["canvas_painted_pixels"] = bool(len(np.unique(arr)) > 8)
except Exception as exc:
    out["screenshot_error"] = f"{type(exc).__name__}: {exc}"
    out["canvas_painted_pixels"] = False        # unmeasurable is not the same as fine

# IMA-261: napari's contrast in this window should repaint the plate. Drive
# `layer.contrast_limits` (napari's own public event) rather than calling our sink directly.
ours = pane.mosaic.ours()
if not ours:
    out["plate_follows_napari_contrast"] = None
    out["contrast_follow_note"] = "no mosaic layer to drive"
else:
    layer = ours[0]
    ov = win._overview
    before_win = tuple(ov.channel_windows()[0]) if ov.channel_windows() else None
    before_px = ov.grab().toImage()
    layer.contrast_limits = (321.0, 8765.0)
    app.processEvents()
    for _ in range(40):                      # the sink is a queued slot; let it land
        app.processEvents()
        time.sleep(0.01)
    after_win = tuple(ov.channel_windows()[0]) if ov.channel_windows() else None
    after_px = ov.grab().toImage()
    out["napari_contrast_set_to"] = [321.0, 8765.0]
    out["plate_window_before"] = list(before_win) if before_win else None
    out["plate_window_after"] = list(after_win) if after_win else None
    out["plate_window_took_it"] = after_win == (321.0, 8765.0)
    out["plate_pixels_changed"] = before_px != after_px
    out["plate_follows_napari_contrast"] = bool(
        out["plate_window_took_it"] and out["plate_pixels_changed"])

report()
