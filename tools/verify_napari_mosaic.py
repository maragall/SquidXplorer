"""Open a real acquisition, open a REGION WINDOW on it, and prove napari painted a real mosaic.

Deliberately NOT offscreen: the offscreen Qt plugin has no OpenGL, so napari's canvas cannot
exist there at all (it segfaults rather than raising). This is the step the headless gates
structurally cannot cover, so it is run by hand against the real datasets.

    python tools/verify_napari_mosaic.py ~/Downloads/sim_2x2_36fov_96wp

RETARGETED 2026-08-06, from ``PlateWindow._mosaic_pane`` to a ``RegionViewer``. There is no pane 2
any more: 2b8fbc5 (2026-07-23, "Decentralize GUI") removed the locked central napari pane, and
``PlateWindow._mosaic_pane`` has been unconditionally None ever since. This script read it on its
third line, found None, printed ``{"pane_is_napari": false, "failure": "no mosaic pane"}`` and
exited 0 -- a script whose entire body was unreachable, reporting success. Viewing now happens in
independent windows spawned through ``ViewerManager``, so that is what is driven here.

Checks, in order of how much they would embarrass us if skipped:
  1. the region window's pane is the napari canvas, not a fallback
  2. a mosaic layer exists per channel, carrying our metadata identity
  3. the layer is a MOSAIC, not one FOV - its extent exceeds the frame shape
  4. the canvas actually PAINTED pixels: screenshot is not blank, not uniform
  5. the layer is placed in stage micrometres
  6. changing napari's contrast in that window REPAINTS THE PLATE (IMA-261)

Check 6 is here and not in ``tools/walkthrough.py`` because it cannot be anywhere else: the
control is napari's LUT row, and napari needs the GL context this file is the only harness to
have. The walkthrough used to drive ndv's slider for it; ndv is gone, and three checks there were
SKIPping or -- worse -- passing while measuring nothing.
"""

from __future__ import annotations

import json
import os
import sys
import time

os.environ.pop("QT_QPA_PLATFORM", None)          # we need a real GL context
os.environ.setdefault("SQUIDMIP_VIEWER", "napari")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# `import squidmip` FIRST, THEN qtpy. `squidmip/__init__` pins QT_API=pyqt6 and the pin has to be
# set before qtpy resolves a binding. This said `from PyQt5.QtWidgets import QApplication` (and
# imported it BEFORE squidmip) until 2026-08-06: the widgets under test were Qt6 while the
# application was Qt5, both frameworks loaded into one process, and the script aborted on
# "QWidget: Must construct a QApplication before a QWidget". Same defect commit 6b51793 fixed in
# tools/walkthrough.py; this file was not carried over. Dead since the Qt6 migration (10b8348,
# f7f9b28, ce5605c).
import squidmip  # noqa: F401
from qtpy.QtWidgets import QApplication

from squidmip._viewer import PlateWindow

if len(sys.argv) < 2:
    print("usage: python tools/verify_napari_mosaic.py <acquisition dir> [budget_s]")
    raise SystemExit(2)
path = sys.argv[1]
if not os.path.isdir(path):
    # Named, never a traceback: this is a developer-machine script over developer-machine data.
    print("VERIFY " + json.dumps({"skipped": f"dataset absent on this machine: {path}"}))
    raise SystemExit(0)
budget = float(sys.argv[2]) if len(sys.argv) > 2 else 240.0

app = QApplication.instance() or QApplication([])
win = PlateWindow()
win.resize(1600, 900)
win.show()
app.processEvents()

out: dict = {"dataset": os.path.basename(path.rstrip("/"))}


def report(code=0):
    print("VERIFY " + json.dumps(out, indent=1))
    sys.stdout.flush()
    # os._exit: unwinding napari + Qt at interpreter shutdown segfaults on this machine, and a
    # verdict decided by a teardown crash is not a verdict.
    os._exit(code)


t0 = time.perf_counter()
win.ingest(path)
app.processEvents()
if win._reader is None:
    out["failure"] = f"ingest failed: {win._readout.text()!r}"
    report(1)
out["regions"] = list((win._meta or {}).get("regions", []))

# THE WINDOW UNDER TEST. `ViewerManager.open` is the same call the plate's own double-click makes,
# so this drives the product's entry point rather than constructing a RegionViewer by hand.
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
    from squidmip._napari_view import key_of

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

# Report 2/3 evidence: z navigable, and all channels composited rather than occluded.
out["dims_ndim"] = int(pane.mosaic.model.dims.ndim)
out["dims_not_displayed"] = [int(a) for a in pane.mosaic.model.dims.not_displayed]
out["z_slider_present"] = out["dims_ndim"] > 2
out["n_z_in_meta"] = int((win._meta or {}).get("n_z") or 1)
out["blending"] = sorted({str(l.blending) for l in pane.mosaic.ours()})
out["colormaps"] = [str(l.colormap.name) for l in pane.mosaic.ours()]
out["all_channels_visible"] = all(l.visible for l in pane.mosaic.ours())
# Count REAL widgets, not row-tuple slots: the row's second element is now the read-only
# window readout, so a truthiness check on it reports a control that no longer exists.
from qtpy.QtWidgets import QPushButton, QSlider, QWidget
# Scoped to the PLATE WIDGET, not to `win._channel_bar`: that attribute has not existed since
# 8b0cbfc (2026-07-22) deleted the plate's channel bar outright, so this reported 0/0 without
# looking at anything. Same stale read as tools/gates.py's contrast_surfaces.
_plate = win._overview
out["plate_contrast_sliders"] = len(_plate.findChildren(QSlider)) if _plate is not None else 0
out["plate_auto_buttons"] = len(
    [b for b in _plate.findChildren(QPushButton) if b.text() == "auto"]) \
    if _plate is not None else 0
# Look for the real widgets in the tree rather than a pane attribute, so this keeps
# reporting the truth regardless of how the control column is stored.
_names = {type(c).__name__ for c in pane.findChildren(QWidget)}
out["napari_layer_controls_mounted"] = "QtLayerControlsContainer" in _names
out["napari_dims_slider_mounted"] = any("QtDim" in n for n in _names)
# 3D is reachable via napari's OWN ndisplay button now that the real Window is embedded.
out["napari_viewer_buttons_present"] = any(
    "ViewerButtons" in type(c).__name__ for c in pane.findChildren(QWidget))

if layers:
    # IN MICROMETRES, not pixels. The mosaic is a DECIMATED multiscale pyramid -- `_mosaic_source`
    # produces preview placement, not native resolution -- so its level-0 array is routinely
    # SMALLER in pixels than one native FOV while covering far more stage. Comparing pixel counts
    # therefore reported `is_a_mosaic_not_one_fov: false` on a genuine 27-FOV mosaic: measured
    # 2026-08-06 on the 10x tissue set, layer array (10, 128, 107) against a (2084, 2084) frame,
    # while `extent.world` said 8618 x 7208 um against one FOV's 1567 um. The pixel comparison was
    # answering a question about resolution; the claim is about EXTENT.
    ext = np.asarray(pane.mosaic.ours()[0].extent.world, dtype=float)
    span_um = (ext[1] - ext[0])[-2:]
    fov_um = [f * float((win._meta or {}).get("pixel_size_um") or 0.0) for f in frame]
    out["mosaic_span_um"] = [round(float(v), 1) for v in span_um]
    out["one_fov_um"] = [round(float(v), 1) for v in fov_um]
    out["is_a_mosaic_not_one_fov"] = bool(
        all(fov_um) and (span_um[0] > fov_um[0] or span_um[1] > fov_um[1]))
    out["placed_in_stage_um"] = bool(any(s != 1.0 for s in layers[0]["scale_um_per_px"]))

# Did it actually paint? A layer list proves the model; only the framebuffer proves pixels.
try:
    img = pane.canvas.screenshot()
    arr = np.asarray(img)[..., :3]
    out["screenshot_shape"] = list(arr.shape)
    out["screenshot_distinct_values"] = int(len(np.unique(arr)))
    out["canvas_painted_pixels"] = bool(len(np.unique(arr)) > 8)
except Exception as exc:
    out["screenshot_error"] = f"{type(exc).__name__}: {exc}"

# ---- 6. IMA-261: napari's contrast in THIS window repaints the PLATE ------------------------
#
# The claim the walkthrough can no longer make. `PlateWindow._follow_window_contrast` subscribes
# to `MosaicLayers.on_user_contrast` off `ViewerManager.windowOpened`; the sink resolves the
# channel index and calls `PlateOverview.set_channel_window`. Driving `layer.contrast_limits` is
# driving napari's own public event -- the same one the LUT row emits when a user drags it --
# rather than calling our sink directly, which is the distinction this whole family of harnesses
# exists to keep.
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

report(0)
