"""In-window bricked 3D: many napari Image layers, one per brick, in the pane already on screen."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional, Sequence

import numpy as np
from qtpy.QtCore import QThread, Signal

from squidxplorer import _bricks
from squidxplorer._logpane import get_logger

log = get_logger("brick3d")

#: Pending-queue cap; an unbounded queue turns a zoom into a backlog.
_QUEUE_MAX = 64

#: How much the cull pads the view, as a fraction of the view's own width.
_CULL_MARGIN_FRACTION = 0.25


class _BrickLoader(QThread):
    """Reads bricks off the UI thread, newest-first, and can be superseded mid-brick."""

    ready = Signal(object, object, object, int, int)   # brick, channel, array, step, epoch
    problem = Signal(str)
    idle = Signal(int)                                 # epoch that just finished draining

    def __init__(self, reader: Any, meta: dict, region: str, parent: Any = None,
                 read: Optional[Callable] = None) -> None:
        super().__init__(parent)
        self._reader, self._meta, self._region = reader, meta, region
        #: Where the voxels come from, injected: callable as ``read(brick, channel, step, should_stop)``.
        self._read = read
        self._pending: list = []
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        #: Bumped by every refresh; a brick read for an older epoch is dropped, not delivered.
        self._epoch = 0

    def request(self, jobs: Sequence[tuple], epoch: int) -> None:
        """Queue ``(brick, channel, step)`` jobs for *epoch*, discarding everything older."""
        with self._lock:
            self._epoch = int(epoch)
            self._pending = list(jobs)[-_QUEUE_MAX:]
        self._wake.set()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def _superseded(self, epoch: int) -> Callable[[], bool]:
        return lambda: self._stop.is_set() or self._epoch != epoch

    def run(self) -> None:                              # noqa: C901 - one loop, read top to bottom
        from squidxplorer._napari3d import read_brick

        while not self._stop.is_set():
            with self._lock:
                job = self._pending.pop() if self._pending else None
                epoch = self._epoch
            if job is None:
                self.idle.emit(epoch)
                self._wake.wait(0.05)
                self._wake.clear()
                continue
            brick, channel, step = job
            stop = self._superseded(epoch)
            try:
                if self._read is not None:
                    arr = self._read(brick, channel, step, stop)
                else:
                    arr = read_brick(
                        self._reader, self._meta, self._region,
                        (brick.r0, brick.r1, brick.c0, brick.c1), channel,
                        step=step, should_stop=stop,
                    )
            except Exception as exc:                    # noqa: BLE001 - reported, never swallowed
                self.problem.emit(f"brick {brick.iy},{brick.ix}/{channel}: "
                                  f"{type(exc).__name__}: {exc}")
                continue
            # None means cancelled or empty; a cancelled read must never be delivered as the answer.
            if arr is None or self._stop.is_set() or self._epoch != epoch:
                continue
            self.ready.emit(brick, channel, arr, int(step), int(epoch))


class BrickedVolume:
    """The bricks of one ROI, live in an existing napari viewer, following that viewer's camera."""

    def __init__(self, mosaic: Any, reader: Any, meta: dict, region: str,
                 window_px: Sequence[int], *, channels: Sequence[str], scale: Sequence[float],
                 origin_um: Sequence[float], limit: int, budget_bytes: int,
                 contrast_by: Optional[dict] = None, colormap_by: Optional[dict] = None,
                 op: str = "raw",
                 say: Optional[Callable[[str], None]] = None, parent: Any = None,
                 read: Optional[Callable] = None) -> None:
        #: The layer model, not a bare viewer: every brick is added, adopted and dropped through it.
        self._mosaic = mosaic
        self._viewer = mosaic.model
        self._meta = meta
        #: Which operator's volume this is, held so every brick can declare it.
        self._op = str(op)
        self._channels = list(channels)
        self._scale = tuple(float(v) for v in scale)          # (dz, py, px) micrometres
        self._origin_um = tuple(float(v) for v in origin_um)  # (z, y, x) of the ROI's corner
        self._limit = int(limit)
        self._budget = int(budget_bytes)
        self._contrast_by = dict(contrast_by or {})
        self._colormap_by = dict(colormap_by or {})
        self._say = say or (lambda _t: None)
        self._nz = len(list(meta.get("z_levels") or [0]))
        self._itemsize = 2                                    # uint16; corrected on first brick

        r0, r1, c0, c1 = (int(v) for v in window_px)
        self._window = (r0, r1, c0, c1)
        # Single-layer fast path as a brick plan: an ROI that fits one texture plans as ONE brick.
        edge = (self._limit if _bricks.fits_single_texture(r1 - r0, c1 - c0, self._nz, self._limit)
                else _bricks.DEFAULT_BRICK_EDGE)
        self._bricks = _bricks.plan(r1 - r0, c1 - c0, limit=self._limit, edge=edge)
        # Bricks are planned in ROI-local coordinates but read in mosaic coordinates.
        self._offset = (r0, c0)

        self._layers: dict = {}          # (channel, brick.key) -> napari layer
        self._steps: dict = {}           # (channel, brick.key) -> the step that layer holds
        self._epoch = 0
        self._step = 1
        self._hidden: list = []          # pane layers we hid, to restore on close
        #: (layer, identity) for every pane layer whose `(op, channel)` we took while 3D is up.
        self._surrendered: list = []
        self._closed = False
        self._t_open: Optional[float] = None
        self._t_first: Optional[float] = None
        self._t_settled: Optional[float] = None

        self._loader = _BrickLoader(reader, meta, region, parent=parent, read=read)
        self._loader.ready.connect(self._on_brick)
        self._loader.problem.connect(self._say)
        self._loader.idle.connect(self._on_idle)

    # -- lifecycle ----------------------------------------------------------------------
    @property
    def brick_count(self) -> int:
        return len(self._bricks)

    @property
    def resident_bytes(self) -> int:
        total = 0
        for ly in self._layers.values():
            data = getattr(ly, "data", None)
            total += int(getattr(data, "nbytes", 0) or 0)
        return total

    def open(self) -> None:
        """Take the scene over: the 2D layers surrender their identity until close() gives it back."""
        self._t_open = time.perf_counter()
        from squidxplorer._napari_view import META_KEY

        for ly in list(self._viewer.layers):
            try:
                meta = getattr(ly, "metadata", None)
                if isinstance(meta, dict) and META_KEY in meta:
                    self._surrendered.append((ly, meta.pop(META_KEY)))
                if ly.visible:
                    self._hidden.append(ly)
                    ly.visible = False
            except Exception:                           # noqa: BLE001 - a layer we cannot hide is
                pass                                    # cosmetic clutter, not a failure to render
        try:
            self._viewer.dims.ndisplay = 3
        except Exception as exc:                        # noqa: BLE001 - named, never silent
            self._say(f"3D: could not switch this pane to 3D ({exc}).")
        # Frame the ROI before the first cull: a fresh camera at the world origin culls every
        # brick out, and reset_view cannot help while there are no layers yet.
        self._frame_camera()
        self._loader.start()
        self.refresh(force=True)

    def _frame_camera(self) -> None:
        """Point the camera at this ROI and zoom so the whole box fits the canvas."""
        r0, r1, c0, c1 = self._window
        _oz, oy, ox = self._origin_um
        h_um = (r1 - r0) * self._scale[1]
        w_um = (c1 - c0) * self._scale[2]
        try:
            cw, ch = (float(v) for v in self._viewer.window._qt_viewer.canvas.size)
        except Exception:                               # noqa: BLE001 - no canvas: leave the camera
            return
        if h_um <= 0 or w_um <= 0 or cw <= 0 or ch <= 0:
            return
        try:
            self._viewer.camera.center = (self._nz * self._scale[0] / 2.0,
                                          oy + h_um / 2.0, ox + w_um / 2.0)
            self._viewer.camera.zoom = min(cw / w_um, ch / h_um) * 0.9
        except Exception as exc:                        # noqa: BLE001 - named; an unframed camera
            self._say(f"3D: could not frame the ROI ({exc}).")

    def close(self) -> None:
        """Remove every brick layer and put the 2D view back exactly as it was."""
        if self._closed:
            return
        self._closed = True
        self._loader.stop()
        self._loader.wait(2000)
        for key in list(self._layers):
            self._drop(key)
        try:
            self._viewer.dims.ndisplay = 2
        except Exception:                               # noqa: BLE001
            pass
        # Identity BEFORE visibility: a visible-but-foreign layer would paint with no tree row.
        from squidxplorer._napari_view import META_KEY

        for ly, identity in self._surrendered:
            try:
                meta = getattr(ly, "metadata", None)
                if isinstance(meta, dict):
                    meta[META_KEY] = identity
            except Exception:                           # noqa: BLE001 - the layer may be gone
                pass
        self._surrendered = []
        for ly in self._hidden:
            try:
                ly.visible = True
            except Exception:                           # noqa: BLE001 - the layer may be gone
                pass
        self._hidden = []

    # -- the camera decides what is resident ---------------------------------------------
    def view_um(self) -> Optional[tuple]:
        """The camera's (y0, x0, y1, x1) world window, padded by the diagonal for 3D rotation."""
        try:
            zoom = float(self._viewer.camera.zoom)
            cz, cy, cx = (float(v) for v in self._viewer.camera.center)
            w, h = (float(v) for v in self._viewer.window._qt_viewer.canvas.size)
        except Exception:                               # noqa: BLE001 - no canvas yet: keep all
            return None
        if zoom <= 0 or w <= 0 or h <= 0:
            return None
        half = ((w * w + h * h) ** 0.5) / 2.0 / zoom
        return (cy - half, cx - half, cy + half, cx + half)

    def um_per_screen_px(self) -> float:
        try:
            zoom = float(self._viewer.camera.zoom)
        except Exception:                               # noqa: BLE001
            return 1.0
        return (1.0 / zoom) if zoom > 0 else 1.0

    def refresh(self, force: bool = False) -> None:
        """Re-decide stride and visible set from the camera, then queue what is missing."""
        if self._closed:
            return
        py = self._scale[1]
        step = _bricks.uniform_step(self.um_per_screen_px(), py)
        # A single-texture ROI that fits the budget is always native: the GPU resamples for free.
        if len(self._bricks) == 1:
            native = self._bricks[0].nbytes(self._nz, self._itemsize, 1) * max(1, len(self._channels))
            if native <= self._budget:
                step = 1
        view = self.view_um()
        margin = 0.0
        if view is not None:
            margin = (view[3] - view[1]) * _CULL_MARGIN_FRACTION
        visible = _bricks.cull(self._bricks, origin_um=self._origin_um[1:], py=py, px=self._scale[2],
                               view_um=view, margin_um=margin)
        budget = _bricks.plan_budget(
            visible, nz=self._nz, itemsize=self._itemsize, step=step,
            bytes_limit=self._budget, n_channels=max(1, len(self._channels)))
        if budget.step != step:
            log.info("3D bricks: %d visible brick(s) do not fit the %.0f MB budget at stride %d; "
                     "using stride %d (%.0f MB). Zoom in for native.",
                     len(visible), self._budget / 1e6, step, budget.step,
                     budget.bytes_resident / 1e6)
        if budget.dropped:
            self._say(f"3D: {budget.dropped} brick(s) left out — the visible volume does not fit "
                      f"the {self._budget / 1e6:.0f} MB budget even at stride {budget.step}.")
        self._step = budget.step
        keep = {b.key for b in budget.bricks}

        # Evict first, so the GPU has room; removing the layer is what frees the texture.
        for (ch, bkey) in list(self._layers):
            if bkey not in keep:
                self._drop((ch, bkey))

        jobs: list = []
        for b in budget.bricks:
            for ch in self._channels:
                key = (ch, b.key)
                if not force and self._steps.get(key) == budget.step:
                    continue                            # already on screen at the right stride
                jobs.append((self._offset_brick(b), ch, budget.step))
        if not jobs:
            return
        # Reversed: the loader pops from the END, and `cull` ordered centre-first.
        self._epoch += 1
        self._loader.request(list(reversed(jobs)), self._epoch)

    def _offset_brick(self, b: "_bricks.Brick") -> "_bricks.Brick":
        """A brick in MOSAIC pixels (what the reader wants) from one in ROI-local pixels."""
        dr, dc = self._offset
        return _bricks.Brick(iy=b.iy, ix=b.ix, r0=b.r0 + dr, r1=b.r1 + dr,
                             c0=b.c0 + dc, c1=b.c1 + dc)

    # -- a brick arrived ------------------------------------------------------------------
    def _on_brick(self, brick, channel: str, arr, step: int, epoch: int) -> None:
        """Runs on the GUI thread (queued signal). Adds or updates ONE brick's layer."""
        if self._closed or epoch != self._epoch:
            return                                      # a later camera won the race
        self._itemsize = int(getattr(arr, "itemsize", 2) or 2)
        dr, dc = self._offset
        local = _bricks.Brick(iy=brick.iy, ix=brick.ix, r0=brick.r0 - dr, r1=brick.r1 - dr,
                              c0=brick.c0 - dc, c1=brick.c1 - dc)
        key = (channel, local.key)
        s = max(1, int(step))
        # scale carries the stride; translate does not — a brick's world corner is fixed.
        # An injected operator source may hand back a different pixel size, so the strided shape
        # is checked and the scale derived from the brick's world extent when it differs.
        expect = local.sampled_shape(self._nz, s)
        got = tuple(int(v) for v in np.shape(arr))
        if len(got) == 3 and got[1:] != expect[1:]:
            h_um = local.height * self._scale[1]
            w_um = local.width * self._scale[2]
            scale = (self._scale[0], h_um / max(1, got[1]), w_um / max(1, got[2]))
        else:
            scale = (self._scale[0], self._scale[1] * s, self._scale[2] * s)
        translate = local.translate_um(self._origin_um, self._scale[1], self._scale[2])
        # Re-frame once, after napari's reset_view fits the FIRST brick instead of the ROI.
        # Every later brick must leave the user's own camera alone.
        first = not self._layers
        existing = self._layers.get(key)
        if existing is not None:
            try:
                existing.data = arr
                existing.scale = scale
                self._steps[key] = s
                self._note_first_pixels()
                return
            except Exception:                           # noqa: BLE001 - fall through to a re-add
                self._drop(key)
        self._add_layer(key, channel, arr, scale, translate)
        if first and self._layers:
            self._frame_camera()
        self._note_first_pixels()

    def _add_layer(self, key, channel: str, arr, scale, translate) -> None:
        """Build ONE brick and hand it to the layer model to adopt.

        How a brick is drawn is 3D knowledge and lives here; what the layer IS
        (identity, contrast, colormap, visibility, group) lives in MosaicLayers.adopt.
        """
        from squidxplorer._napari3d import pin_max_compositing

        kwargs = {
            "name": f"{channel} ▪ {key[1][0]},{key[1][1]}",
            "scale": scale,
            "translate": translate,
            "rendering": "mip",
            # `additive` is what napari is told; the GL max equation is pinned on the visual after.
            # If pinning fails, additive is the graceful degradation: seams, not a blank canvas.
            "blending": "additive",
            "interpolation3d": "linear",
        }
        cmap = self._colormap_by.get(channel)
        if cmap is not None:
            kwargs["colormap"] = cmap
        # The seed only: once adopted, the channel owns its window. This covers the first brick
        # of a channel; the bricks are one volume and must be windowed as one.
        clim = self._contrast_by.get(channel)
        if clim is None:
            clim = self._mosaic.contrast(channel)
        if clim is None:
            from squidxplorer._napari3d import _auto_clim

            clim = _auto_clim(arr)
            if clim is not None:
                self._contrast_by[channel] = clim
        if clim is not None:
            kwargs["contrast_limits"] = tuple(clim)
        try:
            layer = self._viewer.add_image(arr, **kwargs)
        except Exception as exc:                        # noqa: BLE001 - named, never a silent hole
            self._say(f"3D: brick {key[1][0]},{key[1][1]} could not be added: {exc}")
            return
        pin_max_compositing(self._viewer, layer)
        # BEFORE `adopt`, deliberately. napari sized this layer's contrast_limits_range from ONE
        # brick, so a brick cut from a dim corner carries a range narrower than the channel's
        # window. `adopt` then copies its siblings' contrast_limits onto it, and a write outside
        # the range is clamped -- the dim brick would render on a window nobody chose. Widening
        # to the dataset's depth first means there is nothing left for adopt to be clamped by.
        from squidxplorer._napari3d import _seed_range

        _seed_range(layer, getattr(arr, "dtype", None), clim)
        self._mosaic.adopt(self._op, channel, layer)
        self._layers[key] = layer
        self._steps[key] = max(1, int(round(scale[1] / self._scale[1])))

    def _drop(self, key) -> None:
        layer = self._layers.pop(key, None)
        self._steps.pop(key, None)
        if layer is None:
            return
        try:
            # Swap for a stub before removing: vispy holds a C-level reference to the uploaded
            # volume, so removing the layer alone can leave the buffer resident.
            data = getattr(layer, "data", None)
            if isinstance(data, np.ndarray):
                layer.data = np.zeros((1, 1, 1), dtype=data.dtype)
            # Through the model, not `viewer.layers.remove`, so the identity survives while
            # another brick holds it.
            self._mosaic.drop_layer(layer)
        except Exception:                               # noqa: BLE001 - already gone
            pass

    # -- timing --------------------------------------------------------------------------
    def _note_first_pixels(self) -> None:
        if self._t_first is None and self._t_open is not None:
            self._t_first = time.perf_counter() - self._t_open
            log.info("3D bricks: first pixels in %.0f ms", self._t_first * 1000)

    def _on_idle(self, epoch: int) -> None:
        if self._t_settled is None and self._t_open is not None and epoch == self._epoch \
                and self._layers:
            self._t_settled = time.perf_counter() - self._t_open
            log.info("3D bricks: fully resolved in %.0f ms — %d layer(s), stride %d, %.0f MB "
                     "resident", self._t_settled * 1000, len(self._layers), self._step,
                     self.resident_bytes / 1e6)

