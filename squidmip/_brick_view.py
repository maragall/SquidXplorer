"""IN-WINDOW bricked 3D: many napari Image layers, one per brick, in the pane already on screen.

WHAT THIS REPLACES. Clicking 3D used to construct a fresh ``napari.Viewer`` popout, and an ROI
bigger than one GL texture (2048 px, measured) was REFUSED outright -- so the user could draw an
ROI and be told to draw a smaller one. Julio: "the user has no in-window computation and can select
ROIs that can't be seen (I thought we had decided to do bricking)". Both halves are that sentence:
render into THIS window's canvas, and brick anything too big instead of refusing it.

WHY IT IS NOT A NEW RENDERER, which was the explicit constraint ("as long as we don't have to
reinvent a viewer and switch to vispy and all of that BS"). Everything below is napari's own public
surface -- ``add_image``, ``translate``, ``scale``, ``dims.ndisplay`` -- plus ONE ``set_gl_state``
call per layer to get the max blend equation (``_napari3d.pin_max_compositing``, which explains
why). There is no second canvas, no camera to keep in sync, no second contrast model.

THE THREE PROPERTIES, and where each is enforced:

* FULL RESOLUTION -- ``_bricks.uniform_step`` picks one stride from the CAMERA, never from the
  texture limit, so whenever the screen can resolve native detail the stride is 1. Zooming in
  converges to 1:1; it never settles at coarse. A stride the budget forced is LOGGED and said in
  the window, never silent.
* MEMORY-SAFE -- only bricks the camera can see are ever read (``_bricks.cull``), the resident set
  is capped by ``_budget.cache_budget`` (``_bricks.plan_budget``), and a brick that leaves the view
  has its layer removed, which is what actually frees the GPU buffer. Resident bytes at
  screen-adequate stride are ~ (canvas pixels x nz), independent of how big the ROI is.
* FAST -- every read happens on a QThread, never on the Qt UI thread (the 493 ms freeze in
  ``_contrast.py:157`` is what that rule was written from), the queue drains NEWEST-FIRST, and a
  superseded refresh drops the whole pending queue rather than rendering the journey. Bricks appear
  one at a time, so first pixels do not wait for the last brick.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional, Sequence

import numpy as np
from qtpy.QtCore import QThread, Signal

from squidmip import _bricks
from squidmip._logpane import get_logger

log = get_logger("brick3d")

#: Pending-queue cap. A refresh that supersedes another drops the queue outright, so this only
#: bounds a single refresh of a very large region; it exists for the same reason
#: ``_plate_overview._TILE_QUEUE_MAX`` does -- an unbounded queue turns a zoom into a backlog.
_QUEUE_MAX = 64

#: How much the cull pads the view, as a fraction of the view's own width. A small pan must not
#: stall on a brick that was one pixel outside the frustum.
_CULL_MARGIN_FRACTION = 0.25


class _BrickLoader(QThread):
    """Reads bricks OFF the UI thread, newest-first, and can be superseded mid-brick.

    Modelled on ``_plate_overview._TileFetcher`` rather than on ``_MosaicWorker``: this services a
    STREAM of requests over the life of a 3D view (every zoom and pan issues more), so it is one
    long-lived thread with a queue, not a thread per job. LIFO because the most recent request is
    the one the user is looking at -- "FIFO would render the journey; LIFO renders the destination".
    """

    ready = Signal(object, object, object, int, int)   # brick, channel, array, step, epoch
    problem = Signal(str)
    idle = Signal(int)                                 # epoch that just finished draining

    def __init__(self, reader: Any, meta: dict, region: str, parent: Any = None,
                 read: Optional[Callable] = None) -> None:
        super().__init__(parent)
        self._reader, self._meta, self._region = reader, meta, region
        #: WHERE THE VOXELS COME FROM, injected rather than assumed. The default reads the RAW
        #: z-stack straight from the acquisition, but 3D must be able to show whatever the window
        #: is showing -- a decon or bgsub result is a volume too, and hardcoding raw is exactly why
        #: an operator result could be computed in 2D and then never seen as one. The caller picks
        #: the source from the layer DECLARATION (see ``_region_viewer._volume_source``); this only
        #: needs it to be callable as ``read(brick, channel, step, should_stop)``.
        self._read = read
        self._pending: list = []
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        #: Bumped by every refresh. A brick read for an older epoch is DROPPED rather than
        #: delivered: it was computed for a camera that has since moved, and painting it would put
        #: a stale stride on screen. This is the cancellation that keeps a zoom responsive.
        self._epoch = 0

    def request(self, jobs: Sequence[tuple], epoch: int) -> None:
        """Queue ``(brick, channel, step)`` jobs for *epoch*, discarding everything older."""
        with self._lock:
            self._epoch = int(epoch)
            self._pending = list(jobs)[-_QUEUE_MAX:]
        self._wake.set()

    def drop_all(self) -> None:
        with self._lock:
            self._pending = []
        self._wake.set()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def _superseded(self, epoch: int) -> Callable[[], bool]:
        return lambda: self._stop.is_set() or self._epoch != epoch

    def run(self) -> None:                              # noqa: C901 - one loop, read top to bottom
        from squidmip._napari3d import read_brick

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
            # None means cancelled or empty; either way there is nothing honest to paint, and a
            # cancelled read must never be delivered as if it were the answer.
            if arr is None or self._stop.is_set() or self._epoch != epoch:
                continue
            self.ready.emit(brick, channel, arr, int(step), int(epoch))


class BrickedVolume:
    """The bricks of ONE ROI, live in an existing napari viewer, following that viewer's camera.

    Owns its layers and nothing else: it does not touch the pane's own mosaic layers beyond hiding
    them while 3D is up, and it puts every one of them back on ``close``.
    """

    def __init__(self, viewer: Any, reader: Any, meta: dict, region: str,
                 window_px: Sequence[int], *, channels: Sequence[str], scale: Sequence[float],
                 origin_um: Sequence[float], limit: int, budget_bytes: int,
                 contrast_by: Optional[dict] = None, colormap_by: Optional[dict] = None,
                 op: str = "raw",
                 say: Optional[Callable[[str], None]] = None, parent: Any = None,
                 read: Optional[Callable] = None) -> None:
        self._viewer = viewer
        self._meta = meta
        #: WHICH operator's volume this is, held so every brick can DECLARE it.
        #:
        #: Julio, driving the real build 2026-08-05: "in 3d rendering, when all layers are off
        #: there is still a rendered image, unlike 2d that it's a black canvas ... there is still
        #: a layer that looks beautiful but that I can't control so then other controlled layers
        #: are overlayed".
        #:
        #: Root cause: every 2-D mosaic layer declares `{META_KEY: {"op", "channel"}}` and the
        #: layer tree recovers identity from it through `key_of`. A layer WITHOUT it is a FOREIGN
        #: layer, which the tree "deliberately tolerates and ignores". Bricks carried no metadata
        #: at all, so the entire volume was foreign: no group, no checkbox, nothing to switch off
        #: -- while the 2-D layers `open()` had force-hidden kept their checkboxes and could be
        #: switched back ON TOP of it. Stamping the bricks puts the volume inside the same
        #: visibility model as everything else, and all bricks of one channel collapse into ONE
        #: group row. That is the right control surface rather than a compromise: the bricks ARE
        #: one volume, so a per-brick toggle would only be a way to punch holes in it by hand --
        #: the same reason `_link_contrast` below refuses to give each brick its own contrast.
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
        # THE SINGLE-LAYER FAST PATH, expressed as a brick plan rather than as a second code path.
        # An ROI that already fits one texture gets an edge equal to the texture limit, so `plan`
        # returns exactly ONE brick and this renders as ONE add_image with a (dz, py, px) scale --
        # which is gallery-view's recipe unchanged, and the one it is worth keeping. A volume that
        # does not need bricking is not bricked, and there is no branch to keep in step.
        edge = (self._limit if _bricks.fits_single_texture(r1 - r0, c1 - c0, self._nz, self._limit)
                else _bricks.DEFAULT_BRICK_EDGE)
        self._bricks = _bricks.plan(r1 - r0, c1 - c0, limit=self._limit, edge=edge)
        # Bricks are planned in ROI-local coordinates but READ in mosaic coordinates, so each
        # brick carries the ROI's own offset. Keeping the offset here rather than baking it into
        # the brick keeps `_bricks` free of any notion of where a mosaic starts.
        self._offset = (r0, c0)

        self._layers: dict = {}          # (channel, brick.key) -> napari layer
        self._steps: dict = {}           # (channel, brick.key) -> the step that layer holds
        self._epoch = 0
        self._step = 1
        self._hidden: list = []          # pane layers we hid, to restore on close
        #: (layer, identity) for every pane layer whose `(op, channel)` we took while 3D is up, so
        #: the tree cannot lay a flat, coarser mosaic across the volume. Restored in `close`. See
        #: the reasoning in `open`.
        self._surrendered: list = []
        self._closed = False
        self._propagating = False
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
        """Take the scene over: the 2D mosaic stops being a layer the tree can reach, and stays
        that way until :meth:`close` gives it back.

        HIDING IS NOT ENOUGH, measured. Julio, 2026-08-05: *"When I turn on raw it overlays some
        probably downsampled copy of raw over an already full res version of raw that can't be
        controlled by the napari layer."* Both halves of that sentence are this method's doing:

        * the "full res version that can't be controlled" was the volume, before the bricks
          declared an identity (see ``self._op``);
        * the "downsampled copy" is the 2D mosaic layer -- a multiscale pyramid whose level 0 is
          capped to ``_MAX_FUSED_PX``, so it genuinely is coarser than the bricks. This method set
          it ``visible = False`` but left it in the layer TREE with a live checkbox, so one click
          laid a flat, coarser plane across the volume.

        Stamping the bricks alone would have made that WORSE for raw, not better: brick and mosaic
        would then share one ``(op, channel)`` key, so the single group checkbox would light both.
        The identity has to be EXCLUSIVE while 3D is up. So the 2D layers surrender theirs here and
        get it back in ``close``: ``key_of`` reads ``layer.metadata[META_KEY]``, and a layer without
        it is a FOREIGN layer the tree ignores. The tree then shows exactly what the scene contains
        -- one group per channel, driving the bricks -- which is also what makes the channel
        controls work in 3D at all.

        Every layer of ours is stripped, not only the visible ones: an already-hidden mosaic layer
        is just as clickable in the tree as a shown one.
        """
        self._t_open = time.perf_counter()
        from squidmip._napari_view import META_KEY

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
        # FRAME THE ROI BEFORE THE FIRST CULL, and this is load-bearing rather than cosmetic. The
        # cull asks "what can the camera see", and world coordinates here are STAGE micrometres --
        # this acquisition's region starts at x=96814 um. A fresh 3D camera sits at the world
        # origin, tens of thousands of micrometres away, so every brick culls out and the view
        # opens EMPTY with no error to explain it. napari's own reset_view cannot help: it fits the
        # layers, and at this point there are none. Measured: without this, 0 of 120 bricks load.
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
        # Identity BEFORE visibility: a layer made visible while still foreign would paint for one
        # repaint with no row in the tree to switch it off -- briefly the very defect `open` exists
        # to prevent.
        from squidmip._napari_view import META_KEY

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
        """The camera's (y0, x0, y1, x1) world window, padded for rotation.

        napari's ``camera.zoom`` is screen pixels per world unit and world units here ARE stage
        micrometres (every layer carries a micrometre ``scale``/``translate``), so the visible
        world extent is the canvas size over the zoom. The DIAGONAL is used rather than the width
        because in 3D the volume rotates inside the frustum: a box that fits the canvas edge-on can
        swing outside it, and culling a brick that is about to be visible costs a visible stall.
        Conservative on purpose -- over-keeping wastes a read, under-keeping shows a hole.
        """
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
        """Re-decide stride and visible set from the camera, then queue what is missing.

        Called on every camera settle. Idempotent and cheap when nothing changed: a pan inside the
        same brick set at the same stride queues nothing and touches no layer.
        """
        if self._closed:
            return
        py = self._scale[1]
        step = _bricks.uniform_step(self.um_per_screen_px(), py)
        # THE CAPPED ROI IS ALWAYS NATIVE. When the whole volume is ONE texture and it fits the
        # budget there is nothing to gain by sampling it coarsely -- the texture is uploaded once
        # and the GPU resamples it for free at every zoom, so a stride would only mean starting
        # coarse and reloading on the first zoom. Since the drawn ROI is now capped to one texture
        # by construction (``_bricks.clamp_bbox_um``), this is the path the user actually takes:
        # stride 1, full native resolution, no level-of-detail in play at all. The camera-driven
        # stride below remains for a volume that genuinely spans several textures.
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

        # EVICT FIRST, so the GPU has room before the new bricks land. A layer removed here is the
        # only thing that actually frees the texture -- dropping a python reference does not.
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
        # Reversed: the loader pops from the END, and `cull` ordered centre-first, so popping the
        # tail would serve the corners first. The centre is what the user is looking at.
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
        # scale carries the stride; translate does NOT -- a brick's world corner is where it is
        # regardless of how finely it is sampled, which is what lets a coarse brick be replaced by
        # a fine one without anything moving on screen.
        #
        # For the raw reader ``py * step`` is EXACT: sample i is voxel r0 + i*step whatever the
        # ragged last brick does. An injected operator source may hand back a DIFFERENT pixel size
        # (a parent window's fused level 0 is capped to _MAX_FUSED_PX), and assuming native there
        # would stretch the volume across its own footprint. So the expected strided shape is
        # checked, and the scale is derived from the brick's WORLD extent only when it differs --
        # the world corner and the world size are the two things known to be true either way.
        expect = local.sampled_shape(self._nz, s)
        got = tuple(int(v) for v in np.shape(arr))
        if len(got) == 3 and got[1:] != expect[1:]:
            h_um = local.height * self._scale[1]
            w_um = local.width * self._scale[2]
            scale = (self._scale[0], h_um / max(1, got[1]), w_um / max(1, got[2]))
        else:
            scale = (self._scale[0], self._scale[1] * s, self._scale[2] * s)
        translate = local.translate_um(self._origin_um, self._scale[1], self._scale[2])
        # RE-FRAME ONCE, on the first brick. napari calls reset_view() when a layer lands in a
        # viewer that had none, and it fits THAT LAYER -- which for a bricked ROI is one brick, not
        # the ROI. Measured: the bricked view ended up centred and zoomed on a 512 px brick while
        # the single-texture view of the same voxels sat on the full 2048 px box, so the two were
        # framed differently and 91% of pixels "differed" for no reason but the camera. Framing
        # before the first layer exists cannot survive that reset, so it is re-applied after it.
        # Once only: every later brick must leave the user's own camera alone.
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
        from squidmip._napari3d import pin_max_compositing
        from squidmip._napari_view import MosaicKey

        kwargs = {
            "name": f"{channel} ▪ {key[1][0]},{key[1][1]}",
            # IDENTITY, not decoration. `key_of` reads exactly this to decide whether a layer
            # belongs to the app's layer tree; without it a brick is a foreign layer with no
            # checkbox, which is how a 3-D volume came to be unswitchable-off. Every brick of a
            # channel carries the SAME (op, channel), so the tree groups them into one row and one
            # toggle drives the whole volume. See `self._op`.
            "metadata": MosaicKey(self._op, channel).as_metadata(),
            "scale": scale,
            "translate": translate,
            "rendering": "mip",
            # `additive` is what napari is TOLD; the GL max equation is pinned on the visual right
            # after (see pin_max_compositing for why the enum cannot carry it). If pinning fails on
            # some future napari, additive is the graceful degradation: seams, not a blank canvas.
            "blending": "additive",
            "interpolation3d": "linear",
        }
        cmap = self._colormap_by.get(channel)
        if cmap is not None:
            kwargs["colormap"] = cmap
        clim = self._contrast_by.get(channel)
        if clim is None:
            from squidmip._napari3d import _auto_clim

            clim = _auto_clim(arr)
            if clim is not None:
                # ONE window per channel for the whole volume. Deriving it per brick would give
                # every brick its own autoscale and the joins would step in brightness -- the
                # bricks are one volume and must be windowed as one.
                self._contrast_by[channel] = clim
        if clim is not None:
            kwargs["contrast_limits"] = tuple(clim)
        try:
            layer = self._viewer.add_image(arr, **kwargs)
        except Exception as exc:                        # noqa: BLE001 - named, never a silent hole
            self._say(f"3D: brick {key[1][0]},{key[1][1]} could not be added: {exc}")
            return
        pin_max_compositing(self._viewer, layer)
        self._link_contrast(layer, channel)
        self._layers[key] = layer
        self._steps[key] = max(1, int(round(scale[1] / self._scale[1])))

    def _link_contrast(self, layer: Any, channel: str) -> None:
        """One contrast drag moves every brick of that channel.

        The bricks are ONE volume, so a per-brick contrast slider is not a feature, it is a way to
        make the joins visible by hand. napari's own ``link_layers`` is not used because membership
        changes constantly here (bricks are added and evicted as the camera moves) and a link set
        that outlives its layers is a leak; propagating on the event handles that for free.
        """
        def _propagate(event=None, src=layer, ch=channel) -> None:
            if self._propagating or self._closed:
                return
            self._propagating = True
            try:
                value = tuple(src.contrast_limits)
                self._contrast_by[ch] = value
                for (c, _bk), other in self._layers.items():
                    if c == ch and other is not src:
                        other.contrast_limits = value
            except Exception:                           # noqa: BLE001 - contrast is cosmetic
                pass
            finally:
                self._propagating = False

        try:
            layer.events.contrast_limits.connect(_propagate)
        except Exception:                               # noqa: BLE001
            pass

    def _drop(self, key) -> None:
        layer = self._layers.pop(key, None)
        self._steps.pop(key, None)
        if layer is None:
            return
        try:
            # Swapping for a stub before removing is gallery-view's memory patch and it matters
            # here for the same reason: vispy holds a C-level reference to the uploaded volume, so
            # removing the layer alone can leave the buffer resident.
            data = getattr(layer, "data", None)
            if isinstance(data, np.ndarray):
                layer.data = np.zeros((1, 1, 1), dtype=data.dtype)
            self._viewer.layers.remove(layer)
        except Exception:                               # noqa: BLE001 - already gone
            pass

    # -- timing, so the claims about this are measured rather than asserted ---------------
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

    def timings(self) -> dict:
        return {"first_pixels_s": self._t_first, "resolved_s": self._t_settled,
                "step": self._step, "layers": len(self._layers),
                "resident_bytes": self.resident_bytes, "bricks_planned": len(self._bricks)}
