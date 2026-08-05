"""Gallery View: the selected Regions tiled side by side, one row each, one column per channel.

The Qt half of :mod:`squidmip._gallery` — layout and nothing else. Every pixel this window shows
was fused, windowed and handed over by :class:`GalleryWorker` on its own thread; the only numpy
this thread runs is :func:`squidmip._montage.composite` over an array that is ALREADY in RAM at
cell resolution (~512 px), which is the same multiply-sum the plate overview does on every contrast
tick. Nothing here decodes, and ``test_gallery.py`` pins that by failing if the window's thread
ever calls ``reader.read``.

WHY THAT RULE IS WRITTEN DOWN RATHER THAN ASSUMED
--------------------------------------------------
``_contrast.py``'s ``sample_plane`` (line 157) costs a measured **493 ms per region** when its
caller materialises a dask level on the Qt thread just to pick contrast limits, because that
materialisation decodes every FOV of the region at full resolution. A gallery is N of those. The
gallery therefore computes its contrast where it computes its pixels — in the worker, off one
already-decimated array — and the window receives ``(lo, hi)`` as data.

THE LAYOUT IS gallery-view's, NOT A REFLOWING GRID
---------------------------------------------------
hongquanli/gallery-view's gallery is a TABLE: one row per sample, one column per wavelength, in a
``QScrollArea``, with a fixed-size ``QLabel`` per cell and a ghost cell where a sample lacks a
channel so the columns stay aligned. That is ported, with Region for sample: a channel reads DOWN
the page across regions, which is what makes two wells comparable. A reflowing "n columns from the
window width" grid would put two regions' 488 nm cells in different columns at different window
sizes, and comparison is the entire purpose of the view.

Ported alongside it: the three fixed thumbnail presets (80 / 160 / 320), the numpy -> ``QImage``
-> scaled ``QPixmap`` path with its ``.copy()`` and its explicit stride, and rebuilding every cell
from arrays already in RAM whenever a display setting changes (a size or contrast change re-renders,
it never re-reads).

SUBSET-NATIVE, AND IT DOES NOT OWN A SELECTION
-----------------------------------------------
The window is handed a :class:`squidmip._gallery.GalleryScope` and never asks the plate anything.
The scope's FOV mapping is the same ``{region: [fov, ...]}`` shape ``stitch_plate(regions=...)``
takes, so "run the gallery on the marquee" and "run the gallery on the whole acquisition" are one
code path with two scopes. See ``PlateWindow._open_gallery_view``.

NOT A VIEWER WINDOW, SO NO WINDOW ID
-------------------------------------
CONTEXT.md reserves **Window id** for a viewer window in ``ViewerManager``'s registry — the integer
the console prints in brackets, that contrast inheritance and the navigator rows key on. A gallery
has no region cursor, no napari pane and no contrast to inherit; registering one would put a row in
the navigator that none of those lookups can answer. It is owned by the root ``PlateWindow``
instead, one at a time, and re-opening rescopes and raises the existing one.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any, Optional

import numpy as np
from qtpy.QtCore import Qt, QThread, QTimer, Signal
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from squidmip._contrast import dtype_range
from squidmip._gallery import (
    MAX_GALLERY_CELLS,
    GalleryCell,
    GalleryScope,
    channel_field,
    fuse_gallery_cell,
    shared_windows,
)
from squidmip._logpane import get_logger
from squidmip._montage import _hex_to_rgb01, composite

log = get_logger("gallery")

#: gallery-view's three presets, unchanged. Not a continuous slider: a cell is a fixed-size QLabel
#: and the columns only stay aligned because every cell in a column is the same width.
CELL_SIZE_PRESETS = (("Small", 80), ("Medium", 160), ("Large", 320))
DEFAULT_CELL_PX = 160

#: Contrast modes. "shared" is the default and the divergence from gallery-view; see
#: :func:`squidmip._gallery.shared_windows` for why comparison needs it.
CONTRAST_MODES = (("Shared per channel", "shared"), ("Per cell", "cell"))


class GalleryWorker(QThread):
    """Fuses gallery cells off the Qt thread and emits them ONE AT A TIME, as they land.

    One long-lived thread over a ``queue.Queue``, which is gallery-view's ``MipLoader`` shape
    rather than a ``QThreadPool``: a pool would give up the cheap cooperative cancel (a flag the
    loop checks between cells) and the ordering, and the ordering is the feature — cells are queued
    region-major so a whole ROW completes early instead of every region's first channel.

    Emits per cell rather than per gallery because the owner asked for exactly that: "populate each
    channel as soon as it is ready". A gallery that waited for all of them would show nothing for
    the length of the slowest region.
    """

    cellReady = Signal(object)          # GalleryCell
    problem = Signal(str)               # one cell failed, by name; the gallery keeps going
    progress = Signal(int, int)         # (done, total)
    finishedAll = Signal(int)           # cells actually produced

    def __init__(self, reader, meta, scope: GalleryScope, parent=None) -> None:
        super().__init__(parent)
        self._reader, self._meta, self._scope = reader, meta, scope
        self._stop = threading.Event()
        self._queue: "queue.Queue" = queue.Queue()
        for job in scope.cells():
            self._queue.put(job)
        self._total = self._queue.qsize()
        # The SHARED byte budget, not a new cache. `_budget.cache_budget()` already decided how
        # much this machine can spend on preview pixels; a gallery-private cache would spend it
        # twice and the two would evict against each other.
        self._cache, self._token = self._shared_cache(reader)

    @staticmethod
    def _shared_cache(reader):
        """``(cache, token)`` into ``_mosaic_source``'s process-wide plane cache, or ``(None, None)``.

        A reader with no stable identity (a test stub) cannot be given cache keys without risking
        another acquisition's pixels, so it runs UNCACHED and says so once, rather than either
        crashing the gallery or quietly keying on ``id()``.
        """
        from squidmip._mosaic_source import plane_cache, source_token

        try:
            return plane_cache(), source_token(reader)
        except Exception as exc:                            # noqa: BLE001 - stated, not swallowed
            log.info("gallery cells will not be cached: %s", exc)
            return None, None

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:                                  # pragma: no cover - Qt thread
        done = made = 0
        while not self._stop.is_set():
            try:
                region, channel = self._queue.get_nowait()
            except queue.Empty:
                break
            try:
                cell = fuse_gallery_cell(
                    self._reader, self._meta, region, self._scope.fovs_of(region), channel,
                    t=self._scope.t, projection=self._scope.projection,
                    cache=self._cache, token=self._token,
                    should_stop=self._stop.is_set,
                )
            except Exception as exc:                        # noqa: BLE001 - reported per cell
                self.problem.emit(f"{region}/{channel}: {type(exc).__name__}: {exc}")
                cell = None
            done += 1
            if cell is not None:
                made += 1
                self.cellReady.emit(cell)
            elif not self._stop.is_set():
                self.problem.emit(
                    f"{region}/{channel}: no derivable mosaic (no stage positions, no pixel size, "
                    "or no FOV in scope)")
            self.progress.emit(done, self._total)
        self.finishedAll.emit(made)


class GalleryWindow(QMainWindow):
    """The gallery itself: Regions down, channels across, cells filling in as they are fused."""

    def __init__(self, reader, meta, scope: GalleryScope, *, title: str = "Gallery View",
                 parent=None) -> None:
        super().__init__(parent)
        self._reader, self._meta = reader, meta
        self._scope, self._dropped = scope.capped(MAX_GALLERY_CELLS)
        self._acq_title = title
        self._cells: "dict[tuple[str, str], GalleryCell]" = {}
        self._labels: "dict[tuple[str, str], QLabel]" = {}
        self._cell_px = DEFAULT_CELL_PX
        self._contrast_mode = "shared"
        self._shared: "dict[str, tuple[float, float]]" = {}
        self._worker: Optional[GalleryWorker] = None
        self._started_at = 0.0
        self._first_paint_ms: Optional[float] = None
        self._colors = self._channel_colors(meta)

        self.setWindowTitle(f"{title} — Gallery View")
        self._build()
        self.restart()

    # -- construction ---------------------------------------------------------------------------

    @staticmethod
    def _channel_colors(meta) -> dict:
        """``{channel: (3,) float RGB}`` from the RESOLVED ``display_color``.

        The acquisition's own colour, resolved by ``resolve_channels`` (YAML first, wavelength
        fallback second) — the same source every other compositing site in this product tints
        from, so a gallery cell and the plate cell of the same well are the same colour.
        """
        out = {}
        for ch in (meta.get("channels") or []):
            name = channel_field(ch, "name")
            color = channel_field(ch, "display_color")
            if name is not None and color:
                out[str(name)] = _hex_to_rgb01(str(color))
        return out

    def _build(self) -> None:
        central = QWidget(self)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        outer.addLayout(self._controls())

        self._grid_host = QWidget()
        self._grid = QGridLayout(self._grid_host)
        self._grid.setHorizontalSpacing(6)
        self._grid.setVerticalSpacing(6)
        self._grid.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(self._grid_host)
        outer.addWidget(scroll, 1)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        outer.addWidget(self._status)
        self.setCentralWidget(central)

        # Repaints are COALESCED. A shared window widens as cells land, and each widening
        # invalidates every other cell of that channel; repainting synchronously would make the
        # last cell of a 96-region gallery pay for 96 repaints in one slot.
        self._repaint_timer = QTimer(self)
        self._repaint_timer.setSingleShot(True)
        self._repaint_timer.setInterval(120)
        self._repaint_timer.timeout.connect(self._flush_repaints)
        self._dirty_channels: set = set()

        self._lay_out_grid()

    def _controls(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(10)

        self._proj = QComboBox()
        self._proj.addItem("MIP", "mip")
        self._proj.addItem("Single plane", "plane")
        self._proj.setCurrentIndex(0 if self._scope.projection == "mip" else 1)
        self._proj.currentIndexChanged.connect(lambda _i: self.restart())
        row.addWidget(QLabel("Projection:"))
        row.addWidget(self._proj)

        self._contrast = QComboBox()
        for label, key in CONTRAST_MODES:
            self._contrast.addItem(label, key)
        self._contrast.currentIndexChanged.connect(self._on_contrast_mode)
        row.addWidget(QLabel("Contrast:"))
        row.addWidget(self._contrast)

        self._size = QComboBox()
        for label, px in CELL_SIZE_PRESETS:
            self._size.addItem(label, px)
        self._size.setCurrentIndex(1)
        self._size.currentIndexChanged.connect(self._on_cell_size)
        row.addWidget(QLabel("Cell:"))
        row.addWidget(self._size)

        # The timepoint control exists ONLY when there is more than one. A spin box pinned to 0..0
        # is a control that implies an axis the acquisition does not have.
        n_t = int((self._meta or {}).get("n_t", 1) or 1)
        self._t = QSpinBox()
        self._t.setRange(0, max(0, n_t - 1))
        self._t.setValue(min(self._scope.t, n_t - 1))
        self._t.valueChanged.connect(lambda _v: self.restart())
        self._t_label = QLabel("Timepoint:")
        row.addWidget(self._t_label)
        row.addWidget(self._t)
        self._t_label.setVisible(n_t > 1)
        self._t.setVisible(n_t > 1)

        self._holes = QCheckBox("Mark unread FOVs")
        self._holes.setChecked(True)
        self._holes.setToolTip(
            "A FOV that could not be read leaves a hole in the mosaic at its own position. "
            "Ticked, the cell's caption names how many.")
        self._holes.stateChanged.connect(lambda _s: self._refresh_captions())
        row.addWidget(self._holes)

        row.addStretch(1)
        return row

    # -- the grid -------------------------------------------------------------------------------

    def _lay_out_grid(self) -> None:
        """Build every row and column up front, empty. Cells are filled in as they arrive.

        Up front, because the SHAPE of the gallery is known from the scope alone and a grid that
        grew as cells landed would reflow under the user's cursor on every arrival.
        """
        while self._grid.count():
            item = self._grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._labels.clear()

        for col, channel in enumerate(self._scope.channels, start=1):
            head = QLabel(self._channel_label(channel))
            head.setAlignment(Qt.AlignCenter)
            head.setStyleSheet("font-weight:600;")
            self._grid.addWidget(head, 0, col)

        for r, region in enumerate(self._scope.regions, start=1):
            name = QLabel(self._region_label(region))
            name.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            name.setMinimumWidth(90)
            self._grid.addWidget(name, r, 0)
            for c, channel in enumerate(self._scope.channels, start=1):
                cell = QLabel("")
                cell.setAlignment(Qt.AlignCenter)
                cell.setFixedSize(self._cell_px, self._cell_px)
                cell.setStyleSheet(
                    "background-color:#141414; border:1px solid #2a2a2a; border-radius:3px;")
                self._grid.addWidget(cell, r, c)
                self._labels[(region, channel)] = cell
        self._grid.setRowStretch(len(self._scope.regions) + 1, 1)
        self._grid.setColumnStretch(len(self._scope.channels) + 1, 1)

    def _region_label(self, region: str) -> str:
        """The Region, and the FOV count IN SCOPE when that is fewer than the region has.

        Naming the crop on the row is the only way a user can tell a marquee'd corner of a well
        from the whole well: both render as a picture, and one of them is a smaller picture of the
        same tissue.
        """
        in_scope = len(self._scope.fovs_of(region))
        whole = len((self._meta.get("fovs_per_region") or {}).get(region) or ())
        if whole and in_scope < whole:
            return f"{region}\n{in_scope}/{whole} FOV"
        return f"{region}\n{in_scope} FOV"

    def _channel_label(self, channel: str) -> str:
        for ch in (self._meta.get("channels") or []):
            if str(channel_field(ch, "name")) == channel:
                return str(channel_field(ch, "display_name") or channel)
        return channel

    # -- the run --------------------------------------------------------------------------------

    def current_scope(self) -> GalleryScope:
        """The scope as the controls currently state it — projection and timepoint are live."""
        return GalleryScope(
            regions=self._scope.regions,
            fovs=self._scope.fovs,
            channels=self._scope.channels,
            t=int(self._t.value()),
            projection=str(self._proj.currentData()),
            from_selection=self._scope.from_selection,
        )

    def restart(self) -> None:
        """(Re)fuse every cell for the current scope. Idempotent; cancels any run in flight."""
        self._stop_worker()
        self._scope = self.current_scope()
        self._cells.clear()
        self._shared.clear()
        self._first_paint_ms = None
        self._lay_out_grid()
        if self._scope.is_empty():
            self._say("Nothing in scope — select some wells on the plate, or open an acquisition.")
            return
        self._started_at = time.perf_counter()
        log.info("gallery: %s", self._scope.describe(self._meta))
        self._worker = GalleryWorker(self._reader, self._meta, self._scope, parent=self)
        self._worker.cellReady.connect(self._on_cell)
        self._worker.problem.connect(self._on_problem)
        self._worker.progress.connect(self._on_progress)
        self._worker.finishedAll.connect(self._on_finished)
        self._worker.start()
        self._say(f"Building {self._scope.cell_count} cell(s)… {self._scope.describe(self._meta)}")

    def _stop_worker(self) -> None:
        w, self._worker = self._worker, None
        if w is None:
            return
        w.stop()
        # Bounded, and it does not block the close: a cell is one region's reads, and the worker
        # checks the flag between FOVs.
        if not w.wait(4000):
            log.warning("gallery worker did not stop within 4 s; leaving it to finish")

    def _on_cell(self, cell: GalleryCell) -> None:
        if self._first_paint_ms is None:
            self._first_paint_ms = (time.perf_counter() - self._started_at) * 1000.0
            log.info("gallery first paint: %.0f ms (%s/%s, %dx%d, step %g)",
                     self._first_paint_ms, cell.region, cell.channel,
                     cell.shape[0], cell.shape[1], cell.step)
        self._cells[(cell.region, cell.channel)] = cell
        if self._contrast_mode == "shared":
            before = self._shared.get(cell.channel)
            self._shared = shared_windows(self._cells.values())
            if self._shared.get(cell.channel) != before:
                self._dirty_channels.add(cell.channel)   # every cell of it now reads differently
                self._repaint_timer.start()
        self._paint(cell)

    def _on_problem(self, msg: str) -> None:
        log.warning("gallery: %s", msg)
        self._say(msg)

    def _on_progress(self, done: int, total: int) -> None:
        if done < total:
            self._say(f"{done}/{total} cells… {self._scope.describe(self._meta)}")

    def _on_finished(self, made: int) -> None:
        first = "" if self._first_paint_ms is None else f"first paint {self._first_paint_ms:.0f} ms; "
        note = ""
        if self._dropped:
            note = (f" — {self._dropped} further region(s) are NOT shown "
                    f"(cap {MAX_GALLERY_CELLS} cells); select fewer wells to choose which.")
        holes = sum(len(c.unreadable) for c in self._cells.values())
        if holes:
            note += f" {holes} FOV(s) could not be read and are black holes in their own places."
        self._say(f"{made} cell(s); {first}{self._scope.describe(self._meta)}.{note}")
        log.info("gallery complete: %d cell(s), %s%s", made, first, self._scope.describe(self._meta))

    # -- rendering (arrays already in RAM; nothing here reads) ----------------------------------

    def _window_for(self, cell: GalleryCell):
        """The ``(lo, hi)`` this cell is drawn at, honouring the contrast mode.

        Falls back to the dtype's FULL range when no window could be derived, which draws a blank
        channel BLACK. A fabricated narrow window would draw its noise at full intensity, i.e. it
        would read as signal — ``_contrast.auto_contrast`` returns None rather than guess for
        exactly this reason and the gallery must not undo that.
        """
        if self._contrast_mode == "shared":
            win = self._shared.get(cell.channel)
            if win is not None:
                return win
        if cell.window is not None:
            return cell.window
        return dtype_range(cell.image.dtype)

    def _paint(self, cell: GalleryCell) -> None:
        label = self._labels.get((cell.region, cell.channel))
        if label is None:
            return
        lo, hi = self._window_for(cell)
        color = self._colors.get(cell.channel)
        if color is None:
            color = np.ones(3, dtype=np.float32)      # unresolved channel: grey, never invisible
        # THE single home of the window-multiply-sum (`_montage.composite`), over a one-channel
        # store. Not a private ramp: what the gallery shows and what the montage exports have to
        # be the same arithmetic or they will drift.
        rgb = composite(cell.image[None, ...], np.asarray(color, np.float32)[None, :], [(lo, hi)])
        h, w = rgb.shape[:2]
        # `.copy()` is load-bearing (QImage does not own a buffer it is handed) and the explicit
        # 3*w stride is too (a width that is not a multiple of 4 corrupts without it). Both are
        # gallery-view's, and `_layer_tree.py` already states the same rule for the same reason.
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        label.setPixmap(QPixmap.fromImage(img).scaled(
            self._cell_px, self._cell_px, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        label.setToolTip(self._caption(cell, lo, hi))

    def _caption(self, cell: GalleryCell, lo: float, hi: float) -> str:
        bits = [f"{cell.region} · {self._channel_label(cell.channel)}",
                f"{cell.shape[1]}x{cell.shape[0]} px (1/{cell.step:g} of "
                f"{cell.full_shape[1]}x{cell.full_shape[0]})",
                f"{cell.n_fovs} FOV in scope",
                f"window {lo:.0f}–{hi:.0f} ({self._contrast_mode})"]
        if cell.unreadable and self._holes.isChecked():
            bits.append(f"{len(cell.unreadable)} FOV unreadable: {list(cell.unreadable)[:5]}")
        return "\n".join(bits)

    def _flush_repaints(self) -> None:
        channels, self._dirty_channels = self._dirty_channels, set()
        for (region, channel), cell in self._cells.items():
            if channel in channels:
                self._paint(cell)

    def _refresh_captions(self) -> None:
        for cell in self._cells.values():
            self._paint(cell)

    def _on_contrast_mode(self, _index: int) -> None:
        self._contrast_mode = str(self._contrast.currentData())
        self._shared = shared_windows(self._cells.values()) if self._contrast_mode == "shared" else {}
        self._refresh_captions()

    def _on_cell_size(self, _index: int) -> None:
        """A size change RE-RENDERS from the arrays in RAM. It never re-reads — gallery-view's rule."""
        self._cell_px = int(self._size.currentData() or DEFAULT_CELL_PX)
        for label in self._labels.values():
            label.setFixedSize(self._cell_px, self._cell_px)
        self._refresh_captions()

    def _say(self, msg: str) -> None:
        self._status.setText(msg)

    # -- lifetime -------------------------------------------------------------------------------

    def first_paint_ms(self) -> Optional[float]:
        """First paint for this gallery: scope accepted -> first cell on screen. None until then."""
        return self._first_paint_ms

    def rescope(self, scope: GalleryScope, *, title: Optional[str] = None) -> None:
        """Point an OPEN gallery at a new scope. The re-open gesture, so ids and geometry survive."""
        self._scope, self._dropped = scope.capped(MAX_GALLERY_CELLS)
        if title:
            self._acq_title = title
            self.setWindowTitle(f"{title} — Gallery View")
        n_t = int((self._meta or {}).get("n_t", 1) or 1)
        self._t.setRange(0, max(0, n_t - 1))
        self._t.setValue(min(scope.t, n_t - 1))
        self._t_label.setVisible(n_t > 1)
        self._t.setVisible(n_t > 1)
        self._proj.setCurrentIndex(0 if scope.projection == "mip" else 1)
        self.restart()

    def closeEvent(self, event):                              # noqa: N802 - Qt name
        self._stop_worker()
        super().closeEvent(event)
