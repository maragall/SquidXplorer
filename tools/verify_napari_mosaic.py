"""Open a real acquisition in the real window and prove pane 2 painted a real mosaic.

Deliberately NOT offscreen: the offscreen Qt plugin has no OpenGL, so napari's canvas cannot
exist there at all (it segfaults rather than raising). This is the step the headless gates
structurally cannot cover, so it is run by hand against the real datasets.

Checks, in order of how much they would embarrass us if skipped:
  1. pane 2 is the napari canvas, not the ndviewer fallback
  2. a mosaic layer exists per channel, carrying our metadata identity
  3. the layer is a MOSAIC, not one FOV - its extent exceeds the frame shape
  4. the canvas actually PAINTED pixels: screenshot is not blank, not uniform
  5. the layer is placed in stage micrometres
"""

from __future__ import annotations

import json
import os
import sys
import time

os.environ.pop("QT_QPA_PLATFORM", None)          # we need a real GL context
os.environ.setdefault("SQUIDMIP_VIEWER", "napari")

import numpy as np
from PyQt5.QtWidgets import QApplication

from squidmip._viewer import PlateWindow

path = sys.argv[1]
budget = float(sys.argv[2]) if len(sys.argv) > 2 else 240.0

app = QApplication.instance() or QApplication([])
win = PlateWindow()
win.resize(1600, 900)
win.show()
app.processEvents()

out: dict = {"dataset": os.path.basename(path.rstrip("/"))}

pane = getattr(win, "_mosaic_pane", None)
out["pane_is_napari"] = bool(pane is not None and getattr(pane, "ok", False))
if not out["pane_is_napari"]:
    out["failure"] = getattr(pane, "failure", None) if pane else "no mosaic pane"
    print("VERIFY " + json.dumps(out))
    os._exit(0)

t0 = time.perf_counter()
win.ingest(path)
app.processEvents()

# The fuse runs in a worker; pump the loop until layers land or the budget runs out.
while time.perf_counter() - t0 < budget:
    app.processEvents()
    if pane.mosaic.ops() and not (
        win._mosaic_worker is not None and win._mosaic_worker.isRunning()
    ):
        break
    time.sleep(0.05)

app.processEvents()
out["ingest_and_mosaic_s"] = round(time.perf_counter() - t0, 1)
out["regions"] = list((win._meta or {}).get("regions", []))
out["mosaic_region"] = getattr(win, "_mosaic_region", None)
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
from PyQt5.QtWidgets import QPushButton, QSlider, QWidget
_bar = win._channel_bar
out["plate_contrast_sliders"] = len(_bar.findChildren(QSlider)) if _bar is not None else 0
out["plate_auto_buttons"] = len(
    [b for b in _bar.findChildren(QPushButton) if b.text() == "auto"]) if _bar is not None else 0
# Look for the real widgets in the tree rather than a pane attribute, so this keeps
# reporting the truth regardless of how the control column is stored.
_names = {type(c).__name__ for c in pane.findChildren(QWidget)}
out["napari_layer_controls_mounted"] = "QtLayerControlsContainer" in _names
out["napari_dims_slider_mounted"] = any("QtDim" in n for n in _names)
# 3D is reachable via napari's OWN ndisplay button now that the real Window is embedded.
out["napari_viewer_buttons_present"] = any(
    "ViewerButtons" in type(c).__name__ for c in pane.findChildren(QWidget))

if layers:
    h, w = layers[0]["shape"]
    out["is_a_mosaic_not_one_fov"] = bool(frame and (h > frame[0] or w > frame[1]))
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

print("VERIFY " + json.dumps(out, indent=1))
sys.stdout.flush()
os._exit(0)
