"""Gallery View: the selected Regions tiled side by side, one row each, one column per channel.

The Qt half of :mod:`squidxplorer._gallery` — layout only. All fusing and contrast happens in
:class:`GalleryWorker` on its own thread; this thread never decodes.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Optional

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

from squidxplorer._contrast import dtype_range
from squidxplorer._gallery import (
    MAX_GALLERY_CELLS,
    PROJECTIONS,
    GalleryCell,
    GalleryScope,
    channel_field,
    fuse_gallery_cell,
    shared_windows,
)
from squidxplorer._logpane import get_logger
from squidxplorer._montage import _hex_to_rgb01, composite

log = get_logger("gallery")

#: gallery-view's three fixed presets; cells are fixed-size QLabels so columns stay aligned.
CELL_SIZE_PRESETS = (("Small", 80), ("Medium", 160), ("Large", 320))
DEFAULT_CELL_PX = 160

CONTRAST_MODES = (("Shared per channel", "shared"), ("Per cell", "cell"))

#: Labels only — the keys and their order come from `PROJECTIONS`.
PROJECTION_LABELS = {"mip": "MIP", "plane": "Single plane"}

#: Fixed row-label column width, so every row's cells start at the same x.
_ROW_LABEL_PX = 96

#: Cells redrawn per repaint tick; bounds the worst stall regardless of gallery size.
REPAINT_BUDGET = 12


class GalleryWorker(QThread):
    """Fuses gallery cells off the Qt thread and emits them one at a time, as they land."""

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
        self._cache, self._token = self._shared_cache(reader)

    @staticmethod
    def _shared_cache(reader):
        """``(cache, token)`` into the process-wide plane cache, or ``(None, None)`` uncached."""
        from squidxplorer._mosaic_source import plane_cache, source_token

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
                    time_point=self._scope.time_point, projection=self._scope.projection,
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
        self._headers: "list[QLabel]" = []
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
        """``{channel: (3,) float RGB}`` from the resolved ``display_color``."""
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

        # Repaints are coalesced and drained a few per tick; see _flush_repaints.
        self._repaint_timer = QTimer(self)
        self._repaint_timer.setSingleShot(True)
        self._repaint_timer.setInterval(120)
        self._repaint_timer.timeout.connect(self._flush_repaints)
        self._dirty: set = set()
        # No `_lay_out_grid()` here: `restart()`, called next by __init__, owns the grid's contents.

    def _controls(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(10)

        self._proj = QComboBox()
        # Built from `_gallery.PROJECTIONS`, so a new projection appears here with no edit.
        for key in PROJECTIONS:
            self._proj.addItem(PROJECTION_LABELS.get(key, key), key)
        self._select_projection(self._scope.projection)
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

        # The timepoint control is shown only when the acquisition has more than one.
        n_t = int((self._meta or {}).get("n_t", 1) or 1)
        self._t = QSpinBox()
        self._t.setRange(0, max(0, n_t - 1))
        self._t.setValue(min(self._scope.time_point, n_t - 1))
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
        """Build every row and column up front, empty; cells are filled in as they arrive."""
        # setParent(None) BEFORE deleteLater(): taking a widget out of a layout does not
        # reparent it, so it keeps painting at its old geometry until unparented.
        while self._grid.count():
            item = self._grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._labels.clear()
        self._headers = []

        for col, channel in enumerate(self._scope.channels, start=1):
            head = QLabel(self._channel_header(channel))
            head.setAlignment(Qt.AlignCenter)
            head.setStyleSheet("font-weight:600;")
            # Fixed to the cell width, or a long channel name overruns its column.
            head.setFixedWidth(self._cell_px)
            head.setWordWrap(True)
            head.setToolTip(self._channel_label(channel))
            self._grid.addWidget(head, 0, col)
            self._headers.append(head)

        for r, region in enumerate(self._scope.regions, start=1):
            name = QLabel(self._region_label(region))
            name.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            name.setFixedWidth(_ROW_LABEL_PX)
            name.setWordWrap(True)
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

    def _select_projection(self, projection: str) -> None:
        """Point the combo at *projection* by data, never by index."""
        index = self._proj.findData(str(projection))
        self._proj.setCurrentIndex(index if index >= 0 else 0)

    def _region_label(self, region: str) -> str:
        """The Region, and the FOV count in scope when that is fewer than the region has."""
        in_scope = len(self._scope.fovs_of(region))
        whole = len((self._meta.get("fovs_per_region") or {}).get(region) or ())
        if whole and in_scope < whole:
            return f"{region}\n{in_scope}/{whole} FOV"
        return f"{region}\n{in_scope} FOV"

    def _channel_record(self, channel: str):
        for ch in (self._meta.get("channels") or []):
            if str(channel_field(ch, "name")) == channel:
                return ch
        return None

    def _channel_label(self, channel: str) -> str:
        """The channel's full display name — for tooltips and captions, where there is room."""
        rec = self._channel_record(channel)
        return channel if rec is None else str(channel_field(rec, "display_name") or channel)

    def _channel_header(self, channel: str) -> str:
        """The short column heading: the excitation wavelength when the channel has one."""
        rec = self._channel_record(channel)
        nm = None if rec is None else channel_field(rec, "excitation_nm")
        if nm:
            return f"{int(round(float(nm)))} nm"
        return self._channel_label(channel)

    # -- the run --------------------------------------------------------------------------------

    def current_scope(self) -> GalleryScope:
        """The scope as the controls currently state it — projection and timepoint are live."""
        return GalleryScope(
            regions=self._scope.regions,
            fovs=self._scope.fovs,
            channels=self._scope.channels,
            time_point=int(self._t.value()),
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
        # Bounded wait; the worker checks the stop flag between FOVs.
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
                # Every other cell of this channel now reads differently; queued, not painted.
                self._queue_repaint(k for k in self._cells if k[1] == cell.channel)
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
        """The ``(lo, hi)`` this cell is drawn at, honouring the contrast mode."""
        if self._contrast_mode == "shared":
            win = self._shared.get(cell.channel)
            if win is not None:
                return win
        if cell.window is not None:
            return cell.window
        return dtype_range(cell.image.dtype)

    def _draw_stride(self, cell: GalleryCell) -> int:
        """Integer decimation that still leaves >= 1 source pixel per displayed pixel."""
        h, w = cell.shape
        return max(1, min(h // max(1, self._cell_px), w // max(1, self._cell_px)))

    def _paint(self, cell: GalleryCell) -> None:
        label = self._labels.get((cell.region, cell.channel))
        if label is None:
            return
        lo, hi = self._window_for(cell)
        color = self._colors.get(cell.channel)
        if color is None:
            color = np.ones(3, dtype=np.float32)      # unresolved channel: grey, never invisible
        # Composite at display resolution: stride first (exact — windowing is per-pixel),
        # then window and scale.
        k = self._draw_stride(cell)
        view = cell.image[::k, ::k]
        rgb = composite(np.ascontiguousarray(view)[None, ...],
                        np.asarray(color, np.float32)[None, :], [(lo, hi)])
        h, w = rgb.shape[:2]
        # `.copy()` is load-bearing (QImage does not own a buffer it is handed) and so is the
        # explicit 3*w stride (a width not a multiple of 4 corrupts without it).
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

    def _queue_repaint(self, keys) -> None:
        """Mark cells as needing a redraw. Drained by :meth:`_flush_repaints`, a few per tick."""
        self._dirty.update(keys)
        if self._dirty and not self._repaint_timer.isActive():
            self._repaint_timer.start()

    def _flush_repaints(self) -> None:
        """Redraw at most :data:`REPAINT_BUDGET` cells, then hand the event loop back."""
        for _ in range(REPAINT_BUDGET):
            if not self._dirty:
                break
            cell = self._cells.get(self._dirty.pop())
            if cell is not None:
                self._paint(cell)
        if self._dirty:
            self._repaint_timer.start()

    def _refresh_captions(self) -> None:
        """Redraw everything, through the same budgeted queue a contrast widening uses."""
        self._queue_repaint(self._cells.keys())

    def _on_contrast_mode(self, _index: int) -> None:
        self._contrast_mode = str(self._contrast.currentData())
        self._shared = shared_windows(self._cells.values()) if self._contrast_mode == "shared" else {}
        self._refresh_captions()

    def _on_cell_size(self, _index: int) -> None:
        """A size change re-renders from the arrays in RAM; it never re-reads."""
        self._cell_px = int(self._size.currentData() or DEFAULT_CELL_PX)
        for label in self._labels.values():
            label.setFixedSize(self._cell_px, self._cell_px)
        for head in self._headers:
            head.setFixedWidth(self._cell_px)   # headers are pinned to the column, so they follow
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
        self._t.setValue(min(scope.time_point, n_t - 1))
        self._t_label.setVisible(n_t > 1)
        self._t.setVisible(n_t > 1)
        self._select_projection(scope.projection)
        self.restart()

    def closeEvent(self, event):                              # noqa: N802 - Qt name
        self._stop_worker()
        super().closeEvent(event)
