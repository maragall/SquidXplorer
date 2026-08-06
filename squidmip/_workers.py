"""The eight background threads the plate window runs its long work on.

Gap 6 of the GUI backlog plan (2026-07-29), step 2 of the split of ``squidmip/_viewer.py``.

WHY THIS WAS CUT, AND WHY HERE
------------------------------
Every class in here is the same three things and nothing else: an ``__init__`` that stores its
arguments, a set of ``Signal`` declarations, and a ``run()`` that does the work and emits. None
of them touches a widget, reads a layout, or knows what a dock is, because a QThread that touched a
widget would be a bug: Qt owns widgets on the GUI thread only. That constraint had already made
them self-contained; it just had not made them separately FILED.

So the seam here is not a judgement call about cohesion. It is the Qt threading rule read as an
architectural boundary. What is on the far side of it cannot, by construction, need the window.

WHAT IS IN HERE
---------------
* :class:`_OperatorWorker` streams an operator over the plate (``project_plate`` for a z-reducer,
  ``stitch_plate`` for a region operator) and persists it as a navigable OME-zarr plate.
* :class:`_MinervaWorker`, :class:`_MosaicWorker`, :class:`_FocusWorker`, :class:`_SpotWorker`,
  :class:`_FlatfieldWorker`: one long operation each, off the GUI thread.
* :class:`_PreviewWorker` mosaics the RAW plate thumbnail before any operator has run.
* :class:`_ComputedPlateWorker` re-reads an ALREADY written plate back into the overview.
* the helpers those runs call: :func:`_spot_stages`, :func:`_full_res_plane`, :func:`_full_res_mip`,
  and the three tuning constants ``_VIEWER_WORKERS``, ``_MIN_PREVIEW_BOX_PX``, ``_CACHE_AUTO``.

WHAT IS DELIBERATELY NOT IN HERE
--------------------------------
``_LoupeWorker`` and ``_TileFetcher``, the plate overview's own two threads, live in
:mod:`squidmip._plate_overview` beside their only caller. They are the reason the arrows point the
way they do: this module IMPORTS the plate geometry (``_fit_cell``, ``_CELL``, ``push_shape_for``)
to fill its tiles, so filing PlateOverview's threads here would have made the two modules import
each other. A cycle is a worse outcome than two threads sitting next to their owner.

``_viewer`` -> ``_workers`` -> ``_plate_overview`` -> the domain layer. One direction, no cycles.
Nothing here imports ``_viewer``.

Behaviour is unchanged by the move: every class below is byte-identical to the ``_viewer.py`` it
came from, and ``_viewer.py`` re-exports all of them, which matters more here than it looks.
Several tests swap a worker out with ``monkeypatch.setattr(V, "_OperatorWorker", ...)`` and
``PlateWindow`` still resolves the name in ``_viewer``'s namespace, so those spies keep working
exactly as before. ``squidmip._region_viewer`` also imports ``_SpotWorker``, ``_FocusWorker`` and
``_MosaicWorker`` from ``_viewer`` inside function bodies; those keep working too.

This removed 949 lines from ``_viewer.py``, which went from 5,940 lines to 4,994 (the balance is
the re-export block and the imports the move made dead).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
from qtpy.QtCore import QThread, Signal

from squidmip import _explore
from squidmip._engine import _default_workers
from squidmip._logpane import capture_stdout_to_log, get_logger
from squidmip._measure import (
    FAILED as _MEASURE_FAILED, OK as _MEASURE_OK, PARTIAL as _MEASURE_PARTIAL,
    STOPPED as _MEASURE_STOPPED, measure_run,
)
from squidmip._montage import _area_downsample
from squidmip._operations import operator_label
from squidmip._plate_overview import (
    _CELL, _PUSH_PX, _box_union, _fit_box, _fit_cell, _fit_letterboxed, _mosaic_boxes,
    content_box, push_shape_for, region_mosaic_extent_px,
)
from squidmip._progress import FOV_UNIT, PREVIEW_LABEL, RunProgress, unit_plan
from squidmip._tsctx import HANDLES
from squidmip.contract import field_path

#: Same logger name these runs logged under before the move, so a log line reads identically.
log = get_logger("viewer")

_VIEWER_WORKERS = min(6, _default_workers())   # adapt to the machine, but CAP at 6: the producer's
                           # peak RAM is ~workers x one-well (~139 MB each on a 1536wp), and projection
                           # throughput scales only sublinearly past ~6 threads — so more workers buys
                           # little speed for linearly more memory. 6 balances both, leaves GUI cores.

_MIN_PREVIEW_BOX_PX = 4    # smallest FOV box (of _CELL) the RAW preview will bother mosaicking
#                            (IMA-253): below this a field is a speck, and reading one plane per
#                            field to draw specks is pure cost. The operator path is unaffected.

#: "build the cache yourself from the reader" as a DEFAULT that ``None`` can override. A plain
#: ``None`` default would make "no cache" unsayable, which is exactly what the uncached-path tests
#: and ``SQUIDMIP_PLATE_CACHE=0`` need to be able to say.
_CACHE_AUTO = object()


# --- operator worker: stream a projection over the plate, fill row-major -------------------

class _OperatorWorker(QThread):
    """Runs an operator (MIP) over the plate AND persists it as a navigable multiscale OME-Zarr plate
    (``write_plate``), filling one thumbnail per well as each is written. Projection + pyramid write
    run in write_plate's bounded producer/writer pools; our ``_on_well`` renders the plate tile and
    is called FROM THOSE WRITER THREADS, several at once — so only the done-counter needs ``_lock``
    (the expensive per-channel downsample happens OUTSIDE it, so downsampling still parallelises).
    The worker is deliberately THIN: it emits one native-dtype tile per FIELD — the whole 88x88 cell
    for a single-FOV well, or just that FOV's sub-cell box for a mosaic (IMA-187) — and keeps no
    pixels of its own. PlateOverview owns the per-channel store, the contrast and the compositing,
    so the channel toggle works on every entry path, not only after a run. Memory stays O(engine +
    write workers) wells in flight. The written ``plate.ome.zarr`` is the durable, re-openable artifact.
    """

    tileReady = Signal(int, int, str, object, object)   # (ri, ci, well_id, (C,h,w) native tile,
    #                                                          box=(top,left,h,w) in cell px | None)
    progress = Signal(int, int)                 # (done, total) WELLS — the plate's header/tab unit
    # ...and the same run counted in the ENGINE'S unit (FOVs for a per-FOV operator, regions for a
    # region operator), carried as one immutable ``squidmip._progress.ProgressReport``. A separate
    # signal rather than a widened ``progress`` because the two have different DENOMINATORS and both
    # are wanted: "3 of 1536 wells" is the right sentence for a plate run's header, and it is the
    # wrong one for the case this exists to fix — a decon over ONE region, where the well counter
    # reads 0 of 1 for seven minutes while 27 FOVs go past underneath it (Julio, 2026-08-03).
    # One object, not (done, total, unit, eta): four positional ints and strings crossing a thread
    # boundary is four chances for a label to show a count from one moment and a total from another.
    runProgress = Signal(object)                # ProgressReport
    streamEnded = Signal()                      # every well landed -> recomposite the whole plate
    writtenReady = Signal(str)                  # path of the written plate.ome.zarr
    wellFailed = Signal(int, int)               # (ri, ci) of a well SKIPPED on a read error
    pushReady = Signal(int, object)             # (fov_idx, [per-channel ~512px plane]) for the slider
    # FULL-RESOLUTION result pixels, per FOV, for the napari layer group (Defect 3). Separate
    # from pushReady because that one is the ~512px ndviewer slider feed: a downsampled,
    # letterboxed preview. A processing LAYER has to be the operator's actual output, in the
    # raw mosaic's frame, or the before/after toggle compares a thumbnail against a pyramid.
    resultReady = Signal(str, int, object)      # (region, fov, (C, Y, X) native dtype)
    failed = Signal(str)                        # whole-run failure (not a per-well skip)
    finished_ok = Signal()

    def __init__(self, operator: str, reader, meta, fov_index: dict, out_dir: str,
                 regions=None, save: bool = True, n_fovs=1, operator_kwargs=None):
        # No nr/nc: the worker no longer builds a plate-sized montage of its own (IMA-206 moved the
        # canvas into PlateOverview), so it has no use for the plate's shape.
        super().__init__()
        self._operator = operator
        self._reader, self._meta = reader, meta
        self._fov_index = fov_index
        self._out_dir = out_dir
        self._regions = regions          # None = whole plate; a list = subset preview (those wells only)
        self._save = save                # False = PREVIEW: compute + push to the viewer, write NOTHING
        self._n_fovs = n_fovs            # None = every FOV per well -> coordinate-placed mosaic tiles
        # Per-run parameters for the operator, whichever table it is in -- a REGION operator's
        # (registration on/off, registration channel, feather width, blunder thresholds, channel
        # subset) from the stitcher panel, or a PROJECTOR's declared `params` from the panel
        # `_param_panel` builds out of them. Carried on BOTH branches of run(): a setting tuned on
        # a preview and then dropped on the save would be thrown away at exactly the moment it is
        # written to disk, and a setting dropped on the PREVIEW is worse still -- the preview is
        # where the value is judged.
        #
        # The second half of that used to read "a projector has no equivalent seam (its parameters
        # are baked in at registration)", and the preview branch of run() matched it by omitting
        # these. `Operator.params` / `Operator.factory` ended that, and an operator declaring four
        # parameters (spot, cellpose) then ran at its defaults while everything on screen said
        # otherwise.
        self._operator_kwargs = dict(operator_kwargs or {})
        # Per-region FOV boxes inside the _CELL thumbnail (IMA-187). Computed ONCE up front from
        # the reader's stage positions, because every arriving FOV needs its box and the geometry
        # never changes mid-run. Empty dict => no positions => the historical single-tile path.
        # IMA-222: a REGION operator (stitch) returns ONE fused mosaic per well, not one array
        # per FOV, so there are no per-FOV sub-boxes to composite into -- the mosaic IS the cell.
        # A non-empty _boxes here would slot a whole-well mosaic into a single FOV's sub-rectangle.
        from squidmip import available_region_operators

        self._region_op = self._operator in available_region_operators()
        self._boxes = {} if (self._region_op or n_fovs == 1) else _mosaic_boxes(meta)
        # IMA-245: the shape of what this run PUSHES to the array viewer. A region operator pushes
        # a whole-region mosaic, so the surface is the mosaic extent (aspect preserved), not the
        # frame. Computed here, once, and read back by the window through `push_shape` — the window
        # declares the viewer's canvas from the SAME number the worker fills, so the producer and
        # the consumer cannot describe two different rectangles.
        self._push_shape = push_shape_for(meta, self._region_op, regions)
        # True when a region run wanted the mosaic extent and could not derive it (no stage
        # positions / no pixel size), so the push falls back to the square frame surface. The
        # window turns this into a readout line: a squashed mosaic must not look like a correct one.
        self._push_shape_estimated = bool(self._region_op
                                          and region_mosaic_extent_px(meta, regions) is None)
        self._total = len(regions) if regions is not None else len(meta["regions"])
        # The ENGINE's unit and its total, known here because the iteration is known here:
        # project_plate walks (region, fov) pairs and stitch_plate walks regions, and both draw
        # their scope from the same `fovs_per_region` table this reads. Computed ONCE, before the
        # run, so the bar is determinate from its first frame instead of growing a denominator.
        # `unit_plan` returns None for the total when the metadata has no FOV table, and the
        # window then draws an INDETERMINATE bar rather than a fabricated percentage.
        _units, _unit = unit_plan(meta, regions, region_op=self._region_op, n_fovs=n_fovs)
        self._progress = RunProgress(operator_label(operator), _units, _unit)
        self._channels = [c["name"] for c in meta["channels"]]
        self._dtype = np.dtype(meta["dtype"])
        self._lock = threading.Lock()             # guards _done (on_well runs on writer threads)
        self._done = 0
        self._seen_fovs: dict[tuple, set] = {}    # (ri,ci) -> FOVs composited so far, for progress
        self._failed_regions: set = set()         # regions whose fields raised (IMA-226: report it)
        self._stop = threading.Event()            # set by the window to end the run cleanly

    @property
    def mosaic_boxes(self) -> dict:
        """``{(region, fov): (top, left, h, w)}`` this run composites into ({} = single-tile path).

        The plate view must hit-test against the SAME boxes the worker paints into, so it reads
        them from here rather than recomputing — a second `_mosaic_boxes(meta)` call would be a
        second chance to disagree, and a disagreement opens a different FOV than the one clicked.
        """
        return self._boxes

    @property
    def push_shape(self) -> tuple:
        """``(h, w)`` of every plane this run pushes to the array viewer (IMA-245).

        Read by the window to size ``start_acquisition``. Same reasoning as ``mosaic_boxes``:
        recomputing it there would be a second chance to disagree, and a disagreement here is a
        black viewer — the push is rejected for the wrong shape and the rejection is invisible.
        """
        return self._push_shape

    @property
    def push_shape_estimated(self) -> bool:
        """True when a REGION run could not derive its mosaic extent and fell back to the square
        frame surface. The window reports it; a squashed mosaic must not pass for a correct one."""
        return self._push_shape_estimated

    @property
    def landed(self) -> int:
        """Wells that actually produced pixels. IMA-226: a live run whose every well raised used to
        finish "✓ · 1 well" with an empty plate behind it — flat-field with no illumination profile
        raises per field, `_on_error` painted the dots red, and the success message printed anyway.
        The status line must not claim a result the plate does not have."""
        with self._lock:
            return self._done

    @property
    def skipped(self) -> int:
        """Regions where at least one field raised and was skipped."""
        with self._lock:
            return len(self._failed_regions)

    def stop(self):
        """Ask the run to stop; write_plate polls this and abandons after in-flight wells drain."""
        self._stop.set()

    def _on_well(self, region, fov, image):
        """Called per written FIELD (on a write_plate WRITER THREAD): composite the plate thumbnail.

        Single-FOV (the historical path) fills the whole _CELL tile. Multi-FOV composites this
        FOV into its coordinate-derived box inside the SAME cell, accumulating across the calls
        for that region — so a 36-FOV well is built from 36 arrivals rather than 36 overwrites.
        Compositing is at THUMBNAIL scale throughout: the cell is _CELL x _CELL no matter how
        many FOVs land in it, so a mosaic well costs the same memory as a single-FOV well.
        """
        info = self._fov_index[region]
        ri, ci, well_id = *info["rc"], info["well_id"]
        well = image[0, :, 0]  # (C, Y, X)
        box = self._boxes.get((region, fov))
        n_c = len(self._channels)

        # Downsample OUTSIDE the lock (the expensive part stays parallel). A mosaic FOV is fitted
        # to its own sub-cell box; a field that is the WHOLE cell's content (a single-FOV well, or
        # a REGION operator's fused mosaic) gets the box its own aspect ratio implies. Either way
        # the box travels with the tile, so the widget slots it in without knowing the geometry.
        #
        # This branch used to call `_fit_cell`, which resizes to EXACTLY (_CELL, _CELL) — fine for
        # a square frame, a STRETCH for a fused mosaic. `_boxes` is deliberately empty for a region
        # operator (there are no per-FOV sub-boxes when the mosaic IS the cell), so every stitched
        # well was squashed into the square: on a 2x5 mosaic that is 2.4x vertically, against a raw
        # preview of the same well letterboxed by `cell_boxes` into the middle band of the cell.
        # Same cell, same size, different geometry — Julio: "raw vs stitched ... are not registered
        # ... for stitching it gets warped". `content_box` is `cell_boxes`' own rule, so the two
        # now land identically.
        if box is None:
            box = content_box(well.shape[1:], _CELL, _CELL)
        top, left, bh, bw = box
        tiles = [_fit_box(well[c_i], bh, bw) for c_i in range(n_c)]
        raw = np.empty((n_c, bh, bw), self._dtype)   # native dtype (half the RAM)
        for c_i, ds in enumerate(tiles):
            raw[c_i] = ds
        with self._lock:                          # shared counter -> serialize (the cheap part)
            seen = self._seen_fovs.setdefault((ri, ci), set())
            was_empty = not seen
            seen.add(fov)
            if was_empty:                          # count WELLS, not fields, so the bar still
                self._done += 1                    # reads "n of n wells" on a 36-FOV plate
            done = self._done
            # ...and count the ENGINE'S unit, which ticks on EVERY call: one per FOV for a per-FOV
            # operator, one per region for a region operator (which is called once per region).
            # Under the same lock as `_done` because `_on_well` runs on several writer threads at
            # once; the report is taken here and emitted below, outside it.
            self._progress.tick(time.monotonic())
            report = self._progress.report()
        # per-channel + its box; the widget windows, places and composites (IMA-206 + IMA-187)
        self.tileReady.emit(ri, ci, well_id, raw, box)
        self.progress.emit(done, self._total)
        self.runProgress.emit(report)
        # feed the ndviewer growing slider: one ~512px plane per channel, in memory (register_array),
        # so scrubbing the processed wells is instant and z-collapsed (nz=1). Downsampled -> bounded.
        # ...at `push_shape`: the frame square for a per-FOV operator, the aspect-preserved mosaic
        # extent for a REGION operator (IMA-245). Squashing a region mosaic into the frame square
        # is what put a whole-well stitch into the array viewer as an unreadable rectangle.
        ph, pw = self._push_shape
        push = [_fit_letterboxed(well[c_i], ph, pw, self._dtype)
                for c_i in range(len(self._channels))]
        self.pushReady.emit(info["idx"], push)
        # The operator's own pixels, undownsampled. `well` is a view into `image`; the slot
        # copies what it keeps and drops the rest, so a plate-wide run does not accumulate.
        self.resultReady.emit(region, fov, well)

    def _on_error(self, region, fov, exc):
        """A well's projection failed (corrupt/missing plane): SKIP it, mark its dot failed, keep the
        run alive. One bad file must not abort a whole-plate run."""
        with self._lock:
            self._failed_regions.add(region)
        info = self._fov_index.get(region)
        if info is not None:
            self.wellFailed.emit(*info["rc"])

    #: The in-flight run's recorder, or None between runs. A CLASS-level default rather than an
    #: __init__ assignment so that a first-paint report arriving before ``run`` has started, or
    #: after it has finished, finds None instead of raising on the GUI thread.
    _recorder = None

    def run(self):
        # TIME AND MEASURE THIS RUN. The same measurement the CLI's EngineExecutor makes, at the
        # GUI's own operator-run path, into the same METRICS log — so the comparison table sees
        # both surfaces' runs and the one line per run reaches the log panel (measure_run logs at
        # INFO to the root logger, which the panel is a sink of). One measurement, three consumers.
        target = _explore.describe_run_target(self._regions, total=self._total) or self._operator
        # CAPTURE print() FOR THE DURATION OF THE RUN. tilefusion says what it is doing with bare
        # print (registration.py:274, optimization.py:254, distortion.py:245, fusion.py:358), not
        # through its loggers, so the panel showed none of it while maragall/stitcher's own GUI
        # showed all of it -- it swaps sys.stdout for the same purpose (gui/app.py:580).
        #
        # It sits HERE and not inside _stitch.py deliberately, in both directions. Not lower,
        # because a CLI or tools/stitch_demo.py run must keep printing to the terminal it was
        # started from, and routing its lines into a logger nobody configured would SWALLOW them.
        # Not narrower, because stitch_plate hands each region to a ThreadPoolExecutor, so the
        # prints land on a pool thread rather than on this QThread: the capture is scoped to the
        # RUN's duration, not to a thread (see squidmip._logpane.capture_stdout_to_log).
        with capture_stdout_to_log(), \
                measure_run(self._operator, target, n_targets=self._total) as _run_metrics:
            _run_metrics.note(surface="gui", save=self._save)
            # FIRST PAINT is measured by the WINDOW, not here: only the GUI thread knows when a
            # tile was actually drawn, and the gap between emitting one and drawing it is the
            # queue delay the metric exists to expose. Publishing the recorder for the lifetime of
            # the run is what lets the window report into the record before it is written. Cleared
            # in a finally so a tile arriving late finds nothing rather than a stale run's record.
            self._recorder = _run_metrics
            try:
                self._run_body(_run_metrics)
            finally:
                self._recorder = None

    def report_first_paint(self, seconds: float) -> None:
        """This run's first tile has been drawn, ``seconds`` after the user asked for it.

        Called from the GUI thread. Safe against the run finishing underneath it: the recorder
        takes the first report only and refuses any report after its block has exited, so a tile
        landing during teardown cannot alter a record that is already written.
        """
        r = self._recorder
        if r is not None:
            r.first_paint(seconds)

    #: The run's progress so far, for a consumer that arrives mid-run (a window connected after
    #: start, or one asking again after a repaint). Reading the tracker without the lock is safe:
    #: the snapshot is built from ints the GIL makes atomic, and a report one tick stale is a
    #: report, where refusing to answer would be a blank bar over a running job.
    @property
    def progress_report(self):
        return self._progress.report()

    def _run_body(self, _run_metrics):
        # SAY 0 OF N BEFORE ANY WORK. The total is known from the metadata, so the window can draw
        # a real bar from the first second rather than after the first unit lands — and on decon,
        # where one unit is minutes, "after the first unit" is most of the wait Julio described.
        self.runProgress.emit(self._progress.report())
        try:
            projector = self._operator
            if self._save:
                # write_plate picks its own stream from the operator, so a region operator (stitch)
                # persists through the same call: both twins yield (region, fov, (T,C,1,Y,X)), and
                # the disk guard sizes a region write from real mosaic extents rather than frames.
                from squidmip import write_plate  # persist + project in one bounded, streaming pass

                write_plate(self._reader, self._out_dir, n_fovs=self._n_fovs, workers=_VIEWER_WORKERS,
                            projector=projector, tiff=False, on_well=self._on_well,
                            stop=self._stop.is_set, on_error=self._on_error, regions=self._regions,
                            operator_kwargs=self._operator_kwargs or None)
                if self._stop.is_set():
                    _run_metrics.finish(_MEASURE_STOPPED, "stopped by the window")
                    return  # window closing / re-opening; drop out cleanly (no final/written emit)
                self.streamEnded.emit()
                self.writtenReady.emit(str(Path(self._out_dir) / "plate.ome.zarr"))
            else:
                # PREVIEW: run the engine over the subset and push each result to the plate + slider,
                # writing NOTHING to disk (so testing an operator on a few wells costs no disk + only
                # the subset's compute). Same math as the saved run — a faithful preview.
                if self._region_op:
                    # IMA-222: a region operator's unit of work is the WELL, so stitch_plate yields
                    # one fused mosaic per region. It mirrors project_plate's contract exactly
                    # (bounded in-flight window, regions=, on_error=, and the same
                    # (region, fov, (T, C, 1, Y, X)) yield), so the loop below is UNCHANGED.
                    # workers=1: peak memory is workers x one fused mosaic (~0.9 GB on a 27-FOV 10x
                    # well), not the ~139 MB of one projected FOV. Saving takes the write_plate
                    # branch above, which dispatches to stitch_plate itself.
                    from squidmip import stitch_plate

                    stream = stitch_plate(self._reader, workers=1, operator=projector,
                                          n_fovs=None, on_error=self._on_error,
                                          regions=self._regions, **self._operator_kwargs)
                else:
                    from squidmip import project_plate

                    # operator_kwargs ON THE PREVIEW TOO. This call used to omit it while the SAVE
                    # branch above passed it and this class's docstring claimed both branches
                    # carried it. It was true when written -- a projector's parameters were baked
                    # in at registration and only a REGION operator took kwargs -- and it stopped
                    # being true the moment `Operator.params`/`factory` landed. The symptom is the
                    # worst shape there is: the panel says min_area_px=400, the console line says
                    # min_area_px=400, and the pixels are the ones min_area_px=30 produces.
                    stream = project_plate(self._reader, workers=_VIEWER_WORKERS, projector=projector,
                                           n_fovs=self._n_fovs, on_error=self._on_error,
                                           regions=self._regions,
                                           operator_kwargs=self._operator_kwargs or None)
                try:
                    for region, fov, image in stream:
                        if self._stop.is_set():
                            _run_metrics.finish(_MEASURE_STOPPED, "stopped by the window")
                            return
                        self._on_well(region, fov, image)
                finally:
                    close = getattr(stream, "close", None)
                    if callable(close):
                        close()
                if self._stop.is_set():
                    _run_metrics.finish(_MEASURE_STOPPED, "stopped by the window")
                    return
                self.streamEnded.emit()
            # Name the OUTCOME for the record, the same rule the status line follows: landed==0 is
            # a partial result however politely we got here (per-well fault isolation keeps one bad
            # file from aborting a plate), and a skip on any well is partial too.
            if self.landed == 0 and self._total:
                _run_metrics.finish(_MEASURE_PARTIAL,
                                    f"produced nothing — all {self._total} target(s) skipped")
            elif self.skipped:
                _run_metrics.finish(_MEASURE_PARTIAL, f"{self.skipped} well(s) skipped")
            else:
                _run_metrics.finish(_MEASURE_OK)
            self.finished_ok.emit()
        except Exception as e:
            # measure_run records this as failed with the exception name and re-raises; catch it
            # here so the QThread still ends via `failed` rather than an unhandled thread exception.
            _run_metrics.finish(_MEASURE_FAILED, f"{type(e).__name__}: {e}")
            self.failed.emit(f"{type(e).__name__}: {e}")


class _MinervaWorker(QThread):
    """Export the selection to Minerva-ingestable files, then start Minerva Author (IMA-228).

    Two stages with deliberately different failure semantics::

        export  ──ok──▶  launch  ──ok──▶  exported(paths) + launched(True)
           │                │
           │                └──fail──▶  exported(paths) + launched(False)   ← files still good
           └──fail──▶  failed(msg)                                          ← nothing written

    A launch failure must NEVER invalidate a successful export: Minerva Author lives in a
    separate checkout that may not be installed, and the OME-TIFF on disk is the deliverable.
    The user always gets the story path either way, because Minerva has no deep link — the
    file is picked by hand in its "Select File" dialog.
    """
    progress = Signal(int, int)          # (done, total) REGIONS exported — the export unit is one
    #                                      fused mosaic per region, and `run()` below counts
    #                                      `grouped`, not the selection. The comment said "FOVs".
    exported = Signal(object)            # [(ome_path, story_path), ...]
    launched = Signal(bool)              # did a Minerva server end up answering?
    failed = Signal(str)
    finished_ok = Signal()

    def __init__(self, reader, selection, out_dir, projector: str, t: int = 0, launch: bool = True,
                 luts=None):
        super().__init__()
        self._reader = reader
        self._selection = list(selection)
        self._out_dir = out_dir
        self._projector = projector
        self._t = t
        self._launch = launch
        # Snapshotted by the CALLER on the GUI thread and handed over as plain data. This thread
        # must never reach into a napari layer to read a colormap: that is a Qt object owned by
        # the main thread. ``None`` = use the export's percentile defaults.
        self._luts = dict(luts) if luts else None
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        from squidmip import _minerva
        try:
            pairs = []

            def on_progress(done, total):
                self.progress.emit(done, total)

            # Export REGION by REGION — the export unit is a fused mosaic per region, so this
            # is also the finest granularity a stop can act on. A stop between regions takes
            # effect promptly; every file already written stays on disk and is reported.
            grouped = _minerva.group_selection(self._selection)
            for i, (region, fovs) in enumerate(grouped.items()):
                if self._stop.is_set():
                    break
                pairs.extend(
                    _minerva.export_selection(
                        self._reader, [(region, f) for f in fovs], self._out_dir,
                        t=self._t, projector=self._projector, luts=self._luts,
                    )
                )
                on_progress(i + 1, len(grouped))
            self.exported.emit(pairs)
            if pairs and self._launch and not self._stop.is_set():
                # should_stop: the liveness wait is up to 90 s and closeEvent joins this thread.
                # Without it, closing mid-poll froze the GUI for the rest of the wait (84 s).
                self.launched.emit(
                    _minerva.launch_minerva(pairs[0][1], should_stop=self._stop.is_set))
            self.finished_ok.emit()
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class _MinervaRenderWorker(QThread):
    """Render already-exported pairs into viewable Minerva exhibits, OFF the GUI thread.

    The second Minerva destination, and the only one that needs no file picking: Author's editor
    cannot be pointed at a file (``app.py`` reads ``sys.argv`` once, for ``--dev``), but
    ``src/render.py`` is a real CLI over the same ``.story.json`` we already write. See
    :func:`squidmip._minerva.render_exhibit`.

    Off the GUI thread because it is MEASURED in minutes, not guessed at: 132 s for one real
    4-channel 11535x9635 region on this machine. ``stop()`` terminates the child process, because
    a 90 s port poll was already too long to hold ``closeEvent`` and two minutes is worse.

    Failure semantics differ from :class:`_MinervaWorker`'s on purpose. There is no half-success
    to protect here: the export already happened and is already reported, so a render either
    produces an exhibit or is a plain failure with ``render.py``'s own stderr attached.
    """

    progress = Signal(int, int)          # (done, total) exhibits rendered
    rendered = Signal(object)            # [index_html_path, ...] in the order asked for
    failed = Signal(str)

    def __init__(self, pairs, out_root=None, threads=None, parent=None):
        super().__init__(parent)
        self._pairs = [(str(o), str(s)) for o, s in pairs]
        self._out_root = out_root
        self._threads = threads
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        from squidmip import _minerva
        done = []
        try:
            for i, (ome, story) in enumerate(self._pairs):
                if self._stop.is_set():
                    break
                # Beside the export, named for it: ~/minerva_export/<acq>/<stem>_rendered/. The
                # exhibit is a directory, so it cannot sit next to the files as a sibling FILE,
                # and putting it under the export keeps one place to delete.
                stem = Path(ome).name
                for suffix in (".ome.tiff", ".ome.tif"):
                    if stem.lower().endswith(suffix):
                        stem = stem[: -len(suffix)]
                        break
                root = Path(self._out_root) if self._out_root else Path(ome).parent
                index = _minerva.render_exhibit(
                    ome, story, root / f"{stem}_rendered",
                    threads=self._threads, should_stop=self._stop.is_set,
                )
                done.append(index)
                self.progress.emit(i + 1, len(self._pairs))
            self.rendered.emit(done)
        except Exception as e:
            # Report what LANDED as well as what broke: rendering three regions and failing on
            # the third still leaves two exhibits the user can open.
            self.rendered.emit(done)
            self.failed.emit(f"{type(e).__name__}: {e}")


class _MosaicWorker(QThread):
    """Fuse one region's FOVs into a mosaic per channel, OFF the GUI thread.

    A 28-FOV tissue region is ~940 MB of TIFF to read. Doing that in ``ingest`` would freeze the
    window for seconds on open, which is precisely the "opens instantly" property IMA-260 bought.
    Results arrive per channel so the first channel paints while the rest are still being read.
    """

    ready = Signal(str, str, object, object, object)
    #        region, channel, LEVELS (pyramid), bbox_um|None, contrast window (lo, hi)|None
    problem = Signal(str)
    finished_count = Signal(int)

    def __init__(self, reader, meta, region, channels, z_index=0, parent=None, t=0):
        super().__init__(parent)
        self._reader, self._meta = reader, meta
        self._region = region
        self._channels = list(channels)
        self._z_index = int(z_index)
        #: WHICH TIMEPOINT this mosaic is of. It used to be nothing at all: ``run`` called
        #: ``fuse_region_pyramid`` without a ``t`` and the signature defaults it to 0, so a window
        #: whose timepoint slider said 3 rendered timepoint 0, and moving that slider re-read the
        #: whole region to repaint byte-identical pixels. Invisible on this machine, where every
        #: acquisition is n_t=1 — see tests/test_time_point.py, which drives an Nt=3 fixture.
        self._t = int(t)
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        from squidmip._mosaic_source import fuse_region_pyramid, mosaic_bbox_um
        # THE SAME seeding function `add_mosaic` calls, imported rather than reimplemented: two
        # contrast rules over one quantity is this codebase's most-repeated defect shape, and
        # `_auto_window_for` already degrades to "let napari autoscale" on any failure, so nothing
        # raised here can cost the layer. It carries no Qt and imports napari nowhere.
        from squidmip._napari_view import _auto_window_for

        try:
            bbox = mosaic_bbox_um(self._meta, self._region)
        except Exception as exc:                    # noqa: BLE001
            self.problem.emit(f"mosaic placement failed: {exc}")
            bbox = None

        n = 0
        for ch in self._channels:
            if self._stop.is_set():
                break
            try:
                # A LAZY MULTISCALE PYRAMID of (z, y, x) levels — the same shape of data the
                # written-OME-Zarr path has always handed napari via open_pyramid. napari fetches
                # only the clipped visible region of the level matching the current zoom, so a
                # fit-to-window view costs a coarse level (~0.9 MB) instead of a full-resolution
                # fused plane (54.9 MB on the real 10x region), per channel, per z step.
                #
                # napari's own dimension slider is still the z control: every level keeps the z
                # axis at full length and only y/x are coarsened. Only the visible (level, z) is
                # ever materialised, and _mosaic_source's bounded cache keeps a revisited one.
                res = fuse_region_pyramid(self._reader, self._meta, self._region, ch,
                                          t=self._t)
            except Exception as exc:                # noqa: BLE001 - reported, never swallowed
                self.problem.emit(f"{self._region}/{ch}: {type(exc).__name__}: {exc}")
                continue
            if res is None:
                self.problem.emit(
                    f"{self._region}: no stage positions / pixel size — mosaic not derivable."
                )
                continue
            levels, _step, _nz = res
            # THE CONTRAST SEED IS COMPUTED HERE, ON THIS THREAD. It is the one piece of work in
            # this path that is not lazy, and until now it happened on the Qt UI thread.
            #
            # An earlier comment here said there was "no contrast window on the wire" because
            # "napari autoscales from the small level it actually renders". Half of that is true
            # and the half that matters is not: `MosaicLayers.add_mosaic` does NOT let napari
            # autoscale, it seeds the fluorescence window itself (`_auto_window_for` ->
            # `_contrast.sample_plane`), and it did so inside the `ready` slot. `sample_plane`
            # already picks the COARSEST pyramid level, so this is not about pixel count: every
            # level of this pyramid is fused from the FOV TIFFs at its own decimation, so
            # materialising even the smallest rung decodes every FOV of the region. Measured on a
            # 27-FOV x 4-channel region: 2.2 ms in this worker and 128 ms of frozen UI, against
            # 493-604 ms on the machine the defect was reported from.
            #
            # Nothing extra is computed by moving it. The decode is one napari needs anyway to
            # draw the layer, and `_mosaic_source`'s bounded plane cache keeps it, so the first
            # paint is served from what this line just warmed.
            window = _auto_window_for(levels, True)
            self.ready.emit(self._region, ch, levels, bbox, window)
            n += 1
        self.finished_count.emit(n)

class _FocusWorker(QThread):
    """Rank one FOV's z planes by Tenengrad sharpness, OFF the GUI thread.

    The reference-plane scan reads every z plane of one FOV — ~40-50 TIFFs on the tissue set.
    It used to run inside the button's ``clicked`` slot, which froze the window solid for the
    whole scan with no progress and no cancel; it was the only long operation in the app without
    a QThread. Planes are area-downsampled to 512 px before scoring, so the cost is dominated by
    the reads rather than the metric.
    """

    ready = Signal(int, str)          # (z index of the sharpest plane, a note or "")
    problem = Signal(str)

    def __init__(self, reader, meta, region, fov, channel, parent=None):
        super().__init__(parent)
        self._reader, self._meta = reader, meta
        self._region, self._fov, self._channel = region, fov, channel
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        from squidmip.projection import _tenengrad

        best_z_i, best_f, read = 0, -1.0, 0
        failures = []
        for z_i, z in enumerate(self._meta["z_levels"]):
            if self._stop.is_set():
                return
            try:
                plane = self._reader.read(self._region, self._fov, self._channel, z)
            except Exception as exc:              # noqa: BLE001 - counted and REPORTED below
                failures.append(f"z={z}: {type(exc).__name__}")
                continue
            read += 1
            f = _tenengrad(_area_downsample(plane, 512, 512).astype(np.float32))
            if f > best_f:
                best_f, best_z_i = f, z_i
        if read == 0:
            # NEVER return a default. Reporting "focused on z=0" when nothing could be read is
            # the log-and-continue failure this project has six confirmed instances of.
            self.problem.emit(
                f"{self._region}:{self._fov} — not one z plane of {self._channel} could be "
                f"read, so there is no sharpest plane. ({'; '.join(failures[:3])})")
            return
        note = ("" if not failures else
                f" ({len(failures)} of {len(self._meta['z_levels'])} planes were unreadable "
                f"and were skipped)")
        self.ready.emit(int(best_z_i), note)


class _SpotWorker(QThread):
    """Run spot detection on the plane pane 2 is CURRENTLY showing, off the GUI thread.

    Spencer: *"responsiveness is important. And an indicator when its working."* Both halves are
    structural here rather than cosmetic:

    * **Responsive.** Same ``QThread`` shape as ``_MosaicWorker``/``_PreviewWorker``. The click
      handler builds this and calls ``start()``, which returns immediately; every pixel is
      touched on this thread. Measured on region ``manual0`` of the 10x tissue slide
      (5731 x 4793 fused mosaic, 405 nm): ~7.3 s total, of which the watershed is ~4.8 s.
    * **Indicator.** ``progress(done, total)`` counts STAGES of the recipe, matching the
      ``Signal(int, int)`` convention every other worker here uses, so an existing indicator
      binds to it unchanged. That signal has no text channel, so the stage NAME goes out
      separately on ``stageChanged(str)`` rather than being smuggled into an int.
    * **Cancellable.** ``stop()`` sets an Event that ``detect_spots`` polls between stages. The
      cancel is honoured at the next stage boundary (worst case one watershed), and a cancelled
      run emits ``cancelled`` and NO result — never a half-finished mask presented as an answer.

    The plane is taken from the layer already on the canvas, not re-read from disk, so what was
    counted is exactly what the user is looking at.
    """

    progress = Signal(int, int)                # (stages done, stages total) — the convention
    stageChanged = Signal(str)                 # the TEXT channel progress(int,int) cannot carry
    ready = Signal(str, str, object, object, object, int)
    # ^ (region, channel, labels (H,W) int32, centroids (N,2) float, bbox_um|None, count)
    problem = Signal(str)                      # a NAMED failure: "<region>/<channel>: ..."
    cancelled = Signal()
    finished_count = Signal(str, str, int)     # (region, channel, count) — the run's answer

    def __init__(self, region, channel, data, z_index, bbox_um, params=None, parent=None):
        super().__init__(parent)
        self._region, self._channel = region, channel
        self._data, self._z = data, z_index
        self._bbox_um = bbox_um
        self._params = params
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        from squidmip._spots import SpotDetectionCancelled, detect_spots, preferred_segmenter

        where = f"{self._region}/{self._channel}"
        algorithm = preferred_segmenter()
        # THE DENOMINATOR IS WHATEVER THE RUNNING ALGORITHM REPORTS. `_spots.STAGES` is the
        # otsu-watershed recipe's 7 stages, and it used to be emitted as the final total no matter
        # which segmenter ran. Cellpose is the preferred nuclei segmenter now and reports a total of
        # 1 ("running cellpose", 0/1 then 1/1), so the closing emit of 7/7 changed the denominator
        # mid-run: a progress bar completed at 1/1 and then jumped backwards to 7/7. `_spots`:
        # "the list is the progress DENOMINATOR ... there is no second hardcoded total to keep in
        # sync". This records the total the algorithm actually reported so there is no second one.
        reported_total = [0]

        def _stage(name, done, total):
            reported_total[0] = int(total)
            self.stageChanged.emit(name)
            self.progress.emit(int(done), int(total))

        try:
            plane = _full_res_mip(self._data)          # segment the MIP over z, not one z-plane
            log.info("%s: detecting nuclei with %s on a %s MIP", where, algorithm, plane.shape)

            res = detect_spots(
                plane, self._params, algorithm=algorithm,
                on_stage=_stage,
                should_stop=self._stop.is_set,
            )
        except SpotDetectionCancelled:
            self.cancelled.emit()
            return
        except Exception as exc:                   # noqa: BLE001 - NAMED, never swallowed
            # Log AND banner. The banner shows it where the user is looking; the log gives it a
            # permanent, copyable line in the panel — a user who clicked the banner away still has
            # the record. (This is the "logger didn't show the detect-nuclei error" gap.)
            log.error("%s: spot detection failed — %s: %s", where, type(exc).__name__, exc)
            self.problem.emit(f"{where}: spot detection failed — {type(exc).__name__}: {exc}")
            return

        # Close on the SAME denominator the run reported. A segmenter that reported no stage at all
        # (none is registered without one, but the seam permits it) falls back to the stage list, so
        # the bar still reaches its total rather than never being filled.
        total = reported_total[0] or len(_spot_stages())
        self.progress.emit(total, total)
        self.stageChanged.emit("done")
        # Success goes to the LOG too, not only the napari readout — the user saw the count in the
        # viewer but nothing in the panel. A run that produced a number the user can act on should
        # leave a copyable line in the log like every other operator.
        log.info("%s: %d nuclei detected (%s)", where, res.count, algorithm)
        self.ready.emit(self._region, self._channel, res.labels, res.centroids,
                        self._bbox_um, res.count)
        self.finished_count.emit(self._region, self._channel, res.count)


def _spot_stages():
    """The stage list, imported lazily so ``_viewer`` keeps no second copy of the denominator."""
    from squidmip._spots import STAGES

    return STAGES


class _FlatfieldWorker(QThread):
    """Estimate an illumination profile LIVE from plate tiles, off the GUI thread.

    All processing comes from maragall/: this calls ``_flatfield.estimate_profile`` which is
    ``tilefusion.flatfield.estimate_flatfield_channel`` (the numpy BaSiC port), NOT a reimplemented
    estimator. The BaSiC solve is seconds-to-minutes, so it must not run on the GUI thread. Reads a
    SPREAD sample of FOVs across the plate (decorrelated content makes the low-rank/sparse split
    better than the first N tiles of one well). Fails to the LOG by name, never silently.
    """

    done = Signal(object)     # FlatfieldProfile
    problem = Signal(str)
    stage = Signal(str)

    def __init__(self, reader, meta, channel, *, max_tiles=48, use_darkfield=False, parent=None):
        super().__init__(parent)
        self._reader, self._meta, self._channel = reader, meta, channel
        self._max_tiles = int(max_tiles)
        self._use_dark = bool(use_darkfield)

    def run(self):                                    # pragma: no cover - Qt thread
        try:
            from squidmip._flatfield import estimate_profile

            meta = self._meta
            z0 = (meta.get("z_levels") or [0])[0]
            fpr = meta.get("fovs_per_region") or {}
            pairs = [(region, int(fov))
                     for region in (meta.get("regions") or [])
                     for fov in (fpr.get(region) or [])]
            if not pairs:
                self.problem.emit("no FOVs to estimate a flat-field from.")
                return
            # Spread the sample across the plate, not the first N of one well.
            step = max(1, len(pairs) // self._max_tiles)
            sample = pairs[::step][: self._max_tiles]
            tiles = []
            for region, fov in sample:
                try:
                    tiles.append(np.asarray(self._reader.read(region, fov, self._channel, int(z0))))
                except Exception:                     # noqa: BLE001 - one bad tile is not fatal
                    continue
                self.stage.emit(f"read {len(tiles)}/{len(sample)} tiles for {self._channel}…")
            if len(tiles) < 3:
                self.problem.emit(
                    f"flat-field estimate needs at least 3 readable tiles for {self._channel}, "
                    f"got {len(tiles)}.")
                return
            self.stage.emit(f"estimating illumination (tilefusion BaSiC) from {len(tiles)} tiles…")
            profile = estimate_profile(np.stack(tiles), use_darkfield=self._use_dark)
            log.info("flat-field: estimated a %s profile from %d tiles (tilefusion BaSiC)",
                     self._channel, len(tiles))
            self.done.emit(profile)
        except Exception as exc:                      # noqa: BLE001 - NAMED to the log, not swallowed
            log.error("flat-field estimate failed for %s: %s", self._channel, exc)
            self.problem.emit(f"{type(exc).__name__}: {exc}")


def _full_res_plane(data, z_index):
    """The FULL-RESOLUTION 2-D plane behind a napari layer's ``data``, whatever shape it is in.

    A napari layer's ``data`` is one of three things here, and counting cells on the wrong one
    gives a wrong number that looks entirely plausible:

    * a **list of pyramid levels** (a multiscale mosaic — level 0 is full resolution, and every
      later level has fewer, larger-looking nuclei). ALWAYS level 0: counting a 4x-downsampled
      level would merge touching nuclei and silently under-report.
    * a **(z, y, x) stack** — take the z the user is actually looking at.
    * a plain **(y, x)** plane.

    Only the ONE plane asked for is ever materialised; a lazy pyramid stays lazy until the
    ``np.asarray`` at the end.
    """
    # A pyramid arrives as a list/tuple whose ELEMENTS are arrays (level 0 is full resolution). A
    # plain nested Python list whose elements are lists/scalars is NOT a pyramid — it merely
    # encodes one array — so it is converted whole. `ndim` on the first element is the
    # discriminator: >=2 means "element is an array" (pyramid); otherwise it is a nested list.
    if isinstance(data, (list, tuple)):
        if not data:
            raise ValueError("the layer holds an EMPTY multiscale pyramid — nothing to count.")
        data = data[0] if getattr(data[0], "ndim", 0) >= 2 else np.asarray(data)

    # Trust ``.ndim`` when present (keeps a lazy dask/zarr level lazy until the final asarray). When
    # it is ABSENT — a container whose ndim defaulted to 2 is exactly what let a (z, y, x) stack
    # skip the reduction and reach the raise as a 3-D "plane" — materialise once and read the real
    # ndim, so the shape of the container never decides whether the z reduction runs.
    ndim = getattr(data, "ndim", None)
    if ndim is None:
        data = np.asarray(data)
        ndim = data.ndim

    # A 3-D (z, y, x) stack is indexed at the z the user is looking at. Anything with MORE leading
    # axes is genuinely ambiguous — we cannot know which is z, which is channel, which is time — so
    # it is REFUSED by name rather than silently counting the middle of the wrong axis.
    if ndim == 3:
        n_z = int(data.shape[0])
        z = n_z // 2 if z_index is None else int(z_index)
        data = data[min(max(z, 0), n_z - 1)]

    plane = np.asarray(data)
    if plane.ndim != 2:
        raise ValueError(
            f"expected a 2-D plane to count on, got shape {plane.shape!r}. The layer's data is "
            "neither a pyramid level list, a (z, y, x) stack, nor a (y, x) plane."
        )
    return plane


def _full_res_mip(data):
    """The full-resolution MIP (max over z) behind a napari layer's ``data`` — what cellpose / spot
    detection segments now (Julio: "run cellpose on a MIP instead of the current z-plane").

    Level 0 always (a downsampled level merges touching nuclei). A (z, y, x) stack is reduced by
    max over z; a plain (y, x) plane IS its own MIP. The max stays lazy on a dask level until the
    final asarray, so only the 2-D result is materialised, not the whole stack at once."""
    if isinstance(data, (list, tuple)):
        if not data:
            raise ValueError("the layer holds an EMPTY multiscale pyramid — nothing to count.")
        data = data[0] if getattr(data[0], "ndim", 0) >= 2 else np.asarray(data)
    ndim = getattr(data, "ndim", None)
    if ndim is None:
        data = np.asarray(data)
        ndim = data.ndim
    if ndim == 3:
        data = data.max(axis=0)                        # MIP over z (lazy on a dask level)
    plane = np.asarray(data)
    if plane.ndim != 2:
        raise ValueError(
            f"expected a 2-D MIP to count on, got shape {plane.shape!r} after the z reduction."
        )
    return plane


class _PreviewWorker(QThread):
    """Fast RAW preview so the plate shows imagery the moment it opens — before any operator runs
    (the "downsample the plate before opening" step). Reads ONE representative z-plane per channel
    per FOV (not the whole stack), area-downsamples, and hands the per-channel tile to the plate.
    Cheap relative to a full projection; parallel reads. Status stays 'empty' (grey frame) — this is
    a preview, not a processed result. A later operator overwrites each tile. Like the other
    producers it keeps the CHANNEL AXIS intact all the way to the widget, so the channel toggle
    works on a freshly-opened acquisition, before any operator has run.

    A REGION IS A MOSAIC, NOT A FOV (IMA-253/IMA-249). This used to read exactly one representative
    FOV per region and stretch it over the region's whole cell, so the real 10x tissue acquisition
    showed two lone frames pretending to be two 27- and 28-FOV mosaics, and the mosaic only ever
    appeared *after* an operator run. It now composites every FOV of a region into that region's
    cell at its coordinate-derived box (``_placement.cell_boxes`` — the same geometry the operator
    path uses, so preview and result describe one layout).

    The cost is driven by the REAL FOV COUNT PER REGION, which is the only way both datasets stay
    fast: the 1536-well fixture is 1536 regions x 1 FOV, so it reads 1536 planes per channel exactly
    as before, takes the identical single-tile code path (``box=None``), and cannot get slower. The
    tissue slide is 2 regions x ~27 FOVs, so it reads 55. Work is emitted per FOV as it lands, so
    cells fill progressively and the UI never blocks on a whole mosaic.
    """

    tileReady = Signal(int, int, str, object, object)   # (ri, ci, well_id, tile, box|None)
    #: This is the raw fill, not an operator run. ``_explore.operator_busy`` reads it so a retired
    #: preview still draining cannot make the next operator run refuse itself.
    IS_PREVIEW = True

    # THE SAME PROGRESS CHANNEL ``_OperatorWorker`` USES, deliberately: one immutable
    # ``squidmip._progress.ProgressReport`` per completed unit. Julio asked for one bar covering
    # "whichever operator we're applying in bulk or in a specific window, even if it's preview", and
    # a second channel for the preview would be a second thing for the one bar to reconcile.
    #
    # WHAT IT DOES NOT REUSE IS ``unit_plan``, and that is not an oversight. ``unit_plan`` computes
    # the ENGINE's denominator -- what ``select_fovs`` would select -- and this pass does not run the
    # engine: ``_plan`` collapses a region to ONE read whenever a mosaic is not derivable or its
    # boxes would be sub-pixel specks (see ``_plan``), so ``unit_plan`` would promise 27 FOVs for a
    # region this pass reads once. The total comes from ``len(plan)``, which is the work that will
    # actually happen, and it is still known BEFORE the first read -- the property that matters.
    runProgress = Signal(object)                # ProgressReport

    streamEnded = Signal()                      # preview complete -> recomposite the whole plate
    failed = Signal(str)                         # a preview that could not finish NAMES why —
    #                                                 a bare `except: pass` left the plate frozen
    #                                                 half-grey, indistinguishable from "loading".

    def __init__(self, reader, meta, fov_index: dict, order: list, mosaic: bool = True,
                 cache=_CACHE_AUTO):
        super().__init__()
        self._reader, self._meta = reader, meta
        self._fov_index, self._order = fov_index, order
        self._channels = [c["name"] for c in meta["channels"]]
        self._dtype = np.dtype(meta["dtype"])
        self._mosaic = bool(mosaic)
        self._stop = threading.Event()
        # The persisted plate cells (_platecache). A sentinel default rather than None so a caller
        # can say "no cache" explicitly and mean it, which is what the uncached-path tests need.
        # `for_reader` returns None, logging why, for anything it cannot cache -- a reader with no
        # `_path`, an unwritable cache dir, SQUIDMIP_PLATE_CACHE=0 -- so this worker never has to
        # care whether caching is available.
        from squidmip._platecache import PlateCellCache

        self._cache = (PlateCellCache.for_reader(reader, meta, cell_px=_CELL)
                       if cache is _CACHE_AUTO else cache)
        self._pending: dict = {}      # region -> the cell being accumulated for the cache
        self.cache_hits = 0           # regions served from the cache, for the status line and tests
        self.cache_reads = 0          # regions actually read from the acquisition
        # INDETERMINATE until ``run`` knows the plan. Built here rather than left as None so a
        # consumer that reads ``progress_report`` before the thread starts (or after it exits) gets
        # a report saying "0 so far, total unknown" instead of raising on the GUI thread -- the same
        # rule ``_OperatorWorker._recorder`` follows for the same reason.
        self._progress = RunProgress(PREVIEW_LABEL, None, FOV_UNIT)

    def _plan(self) -> list:
        """``[(region, fov, box|None), ...]`` — the read list, in plate order, FOVs in stage order.

        ``box=None`` means "this FOV fills its cell": a single-FOV region, or an acquisition with no
        stage coordinates / no pixel size, where a mosaic is not derivable and guessing one would
        draw a wrong picture. That is the historical path, and it stays byte-identical for it.

        A region whose FOVs are so widely spread that each one lands in fewer than
        ``_MIN_PREVIEW_BOX_PX`` cell pixels also previews single-tile. Reading N planes to paint N
        specks is all cost and no picture — and at that scale the "mosaic" and the single tile are
        visually the same thing anyway. The operator path still mosaics it; this is a PREVIEW
        budget, not a change to the geometry.
        """
        from squidmip._placement import cell_boxes, fov_offsets_px

        positions = self._meta.get("fov_positions_um") or {}
        px = self._meta.get("pixel_size_um")
        plan: list = []
        for region in self._order:
            fovs = list(self._meta["fovs_per_region"][region])
            boxes: dict = {}
            if self._mosaic and len(fovs) > 1 and positions and px not in (None, 0):
                try:
                    boxes = cell_boxes(fov_offsets_px(positions, region, fovs, px),
                                       self._meta["frame_shape"], _CELL)
                except (KeyError, ValueError):
                    boxes = {}       # this region previews single-tile; the rest still mosaic
                if any(min(b[2], b[3]) < _MIN_PREVIEW_BOX_PX for b in boxes.values()):
                    boxes = {}
            if boxes:
                plan.extend((region, f, boxes[f]) for f in fovs if f in boxes)
            else:
                plan.append((region, fovs[0], None))
        return plan

    def stop(self):
        self._stop.set()

    def _replay_cached(self, plan: list) -> list:
        """Emit every region the cache can serve; return the plan entries still to be read.

        This is the whole reopen win, and it is one cell per WELL rather than one per FOV: a
        cached 27-FOV mosaic replays as a single tile, so the reopen does not merely skip the
        reads, it skips 26 of every 27 signal round trips too.

        The tile is emitted with its CONTENT BOX, never as a full 88x88 cell. ``add_tile`` feeds
        the running histogram whatever it is handed, and a mosaic cell is zero-padded wherever no
        FOV lands; replaying the padding would pin the 1st percentile at 0 and wash the plate out
        on every reopen while the first open looked right. See ``_platecache.CellTile``.
        """
        if self._cache is None:
            return plan
        by_region: dict = {}
        for item in plan:
            by_region.setdefault(item[0], []).append(item)
        remaining: list = []
        for region, items in by_region.items():
            hit = None if self._stop.is_set() else self._cache.get(region)
            if hit is None:
                remaining.extend(items)
                continue
            ri, ci = self._fov_index[region]["rc"]
            self.tileReady.emit(ri, ci, region, np.asarray(hit).astype(self._dtype), hit.box)
            self.cache_hits += 1
        self.cache_reads = len(by_region) - self.cache_hits
        # Said out loud, because a cache nobody can see is indistinguishable from a fast disk.
        log.info("plate preview: %d of %d wells served from the cell cache (%s)",
                 self.cache_hits, len(by_region), self._cache)
        return remaining

    def _remember(self, region: str, box, tile: np.ndarray, expected: int) -> None:
        """Accumulate one FOV into the region's cell, and publish the cell once it is whole.

        Published per REGION as it completes rather than once at the end, so a preview that is
        stopped half way (the user opens something else) still leaves the wells it finished
        cached. A region whose read RAISED never completes and is therefore never published: a
        half-read cell must not be persisted as though it were the well.
        """
        if self._cache is None:
            return
        st = self._pending.get(region)
        if st is None:
            st = self._pending[region] = {
                "cell": np.zeros((len(self._channels), _CELL, _CELL), dtype=self._dtype),
                "left": int(expected), "box": None}
        top, left = (int(box[0]), int(box[1])) if box is not None else (0, 0)
        h, w = tile.shape[1], tile.shape[2]      # by ACTUAL shape, as add_tile places it
        st["cell"][:, top:top + h, left:left + w] = tile
        st["box"] = _box_union(st["box"], (top, left, h, w))
        st["left"] -= 1
        if st["left"] <= 0:
            self._pending.pop(region, None)
            top0, left0, bh, bw = st["box"]
            self._cache.put(region, st["cell"][:, top0:top0 + bh, left0:left0 + bw], st["box"])

    @property
    def progress_report(self):
        """This pass's progress so far, for a consumer that arrives mid-pass.

        Same contract as ``_OperatorWorker.progress_report``, and safe for the same reason: the
        snapshot is built from ints the GIL makes atomic, so a report one tick stale is still a
        report where refusing to answer would be a blank bar over running work.
        """
        return self._progress.report()

    def run(self):
        # CAPTURE print() FOR THE DURATION OF THE PASS, exactly as ``_OperatorWorker.run`` does and
        # for exactly the same reason (the long version is in
        # ``squidmip._logpane.capture_stdout_to_log``). Julio: "when I run a preview it doesn't show
        # may standalone stitchers log messages on the master log. This tell me it was a partial
        # integration." The integration was partial in the wiring, not in the algorithm -- the
        # capture went onto the operator worker only, so every path that is NOT an operator run
        # still printed into a terminal nobody is watching.
        #
        # Scoped to the PASS and not to this thread, which is load-bearing here too: the reads below
        # run on a ``ThreadPoolExecutor``, so a thread-scoped switch would capture nothing said by
        # the reader (or by anything it imports) while a plane is being read.
        with capture_stdout_to_log():
            self._run_body()

    def _run_body(self):
        try:
            from collections import Counter
            from concurrent.futures import ThreadPoolExecutor
            zs = self._meta["z_levels"]
            z_mid = zs[len(zs) // 2]      # a mid-stack plane is a fair single-plane preview
            plan = self._replay_cached(self._plan())
            # THE DENOMINATOR IS THE PLAN THAT SURVIVED THE CACHE, not the plan before it. A region
            # replayed from the cell cache is not work: counting it would draw a bar that sits at
            # "1400 of 1536" from its first frame and then crawls, and feeding those instant
            # completions to ``RunProgress`` would also poison the rate -- 1400 arrivals in one
            # instant makes the ETA for the remaining 136 wildly optimistic. What is left IS the
            # work, and ``_replay_cached`` already says out loud how many wells it served.
            self._progress = RunProgress(PREVIEW_LABEL, len(plan), FOV_UNIT)
            # Say 0 of N before the first read, same as the operator run: the total is known here,
            # so the bar is determinate from its first frame rather than after the first arrival.
            self.runProgress.emit(self._progress.report())
            per_region = Counter(item[0] for item in plan)

            def load(item):
                region, fov, box = item
                h, w = (_CELL, _CELL) if box is None else (box[2], box[3])
                fit = _fit_cell if box is None else (lambda a: _fit_box(a, h, w))
                return region, box, [fit(self._reader.read(region, fov, ch, z_mid)
                                         .astype(np.float32)) for ch in self._channels]

            with ThreadPoolExecutor(max_workers=_VIEWER_WORKERS) as ex:
                for region, box, tiles in ex.map(load, plan):   # plate order preserved
                    if self._stop.is_set():
                        return
                    ri, ci = self._fov_index[region]["rc"]
                    tile = np.stack(tiles).astype(self._dtype)
                    self.tileReady.emit(ri, ci, region, tile, box)
                    self._remember(region, box, tile, per_region[region])
                    # One unit done. No lock: ``ex.map`` yields on THIS thread, so unlike
                    # ``_OperatorWorker._on_well`` (which runs on several writer threads at once)
                    # every tick here is serialised by construction. Stated rather than assumed,
                    # because ``RunProgress`` is deliberately not thread-safe on its own.
                    self._progress.tick(time.monotonic())
                    self.runProgress.emit(self._progress.report())
            if not self._stop.is_set():
                self.streamEnded.emit()   # the running window is mature now -> one clean recomposite
                # The pass finished, so this generation is complete: compact the per-well cells
                # into one memory-mapped page. AFTER streamEnded, never before -- the recomposite
                # is what the user is waiting on, and compaction is housekeeping. A stopped or
                # failed pass never reaches here, which is the point: a partial plate must not be
                # compacted into a page that claims to be the plate.
                if self._cache is not None and not self._cache.packed:
                    self._cache.pack(self._order)
        except Exception as exc:
            # Preview is best-effort, but best-effort is not SILENT. Finalise the tiles that did
            # land (so a partial mosaic still paints) and then name the failure — the old bare
            # `except: pass` stranded the plate half-grey forever, and streamEnded never fired so
            # the status line kept claiming the load was still in progress.
            if not self._stop.is_set():
                self.streamEnded.emit()
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class _ComputedPlateWorker(QThread):
    """Read a previously-written OME-Zarr plate back into the viewer (no recompute).

    Streams each well from disk: a coarse pyramid level -> the plate thumbnail, and a ~512px level ->
    the ndviewer slider (register_array). Bounded (one well in flight); reads via tensorstore off the
    GUI thread so opening a big computed plate never freezes the window. Emits per-channel tiles, so
    a reopened plate is windowed GLOBALLY by the widget's running contrast exactly like a live run —
    it used to take percentiles per well, which made a dim well and a bright well look identical and
    silently broke the one thing a plate overview is for (comparing wells at a glance)."""

    tileReady = Signal(int, int, str, object, object)   # (ri, ci, well_id, (C, h, w) native tile,
    #                                                          box=(top, left, h, w) in cell px)
    pushReady = Signal(int, object)             # (fov_idx, [per-channel ~512px plane])
    progress = Signal(int, int)
    streamEnded = Signal()                      # plate fully loaded -> recomposite globally
    finished_ok = Signal()
    failed = Signal(str)

    def __init__(self, base, wells, coarse_lvl, push_lvl, dtype, time_point: int = 0):
        self._time_point = int(time_point)   # which timepoint the plate is showing
        super().__init__()
        self._base = base                 # plate.ome.zarr path
        self._wells = wells               # [(well_id, wellpath, fov, ri, ci, flat_idx)]
        self._coarse, self._push = coarse_lvl, push_lvl
        self._dtype = dtype
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def _read(self, wellpath, fov, level, time_point: int = 0):
        # Through the shared pool, NOT a bare ts.open. This line opened a brand new store per well
        # per level, twice per well, with no reuse and no declared memory budget: 3072 fresh opens
        # on a 1536-well plate, each allocating its own private cache. The pool bounds both halves,
        # decoded bytes via one shared cache_pool and live handles via an LRU of 32. See _tsctx.
        arr = HANDLES.get(field_path(self._base, wellpath, fov, level))
        # The timepoint was hardcoded to 0 here, which is what made a 40-timepoint plate
        # indistinguishable from a 1-timepoint one, silently. Clamped rather than trusted, so a
        # stale slider position cannot index off the end of a shorter acquisition.
        t_idx = max(0, min(int(time_point), arr.shape[0] - 1))
        return np.asarray(arr[t_idx, :, 0].read().result())   # (C, y, x) at this t, z=0

    def run(self):
        try:
            n = len(self._wells)
            for i, (wid, wpath, fov, ri, ci, idx) in enumerate(self._wells, 1):
                if self._stop.is_set():
                    return
                coarse = self._read(wpath, fov, self._coarse, self._time_point)   # thumbnail (C,y,x)
                # By the field's OWN aspect ratio, not squashed into the square. A written plate's
                # field is a per-FOV projection (square: `content_box` returns the whole cell and
                # this is `_fit_cell` exactly) OR a stitched region mosaic (not square: `_fit_cell`
                # stretched it). Reopening a stitched plate has to draw the same cell the run that
                # wrote it drew, or the defect survives the round trip through disk.
                box = content_box(coarse.shape[1:], _CELL, _CELL)
                _, _, bh, bw = box
                tile = np.stack([_fit_box(plane.astype(np.float32), bh, bw) for plane in coarse])
                self.tileReady.emit(ri, ci, wid, tile.astype(self._dtype), box)
                push_src = self._read(wpath, fov, self._push, self._time_point)   # slider src (C,Y,X)
                # ...at the declared push canvas exactly (IMA-245): a pyramid level smaller than
                # _PUSH_PX used to be pushed at its own size, which the viewer silently refused.
                push = [_fit_letterboxed(push_src[c], _PUSH_PX, _PUSH_PX, self._dtype)
                        for c in range(push_src.shape[0])]
                self.pushReady.emit(idx, push)
                self.progress.emit(i, n)
            if not self._stop.is_set():
                self.streamEnded.emit()   # every well in the store -> one global-window recomposite
                self.finished_ok.emit()
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")
