"""QThread workers for the app's long-running work; none of them touches a widget."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
from qtpy.QtCore import QThread, Signal

from squidxplorer import _run_scope
from squidxplorer._dispatch import run_operator_once
from squidxplorer._engine import _default_workers
from squidxplorer._logpane import capture_stdout_to_log, get_logger
from squidxplorer._measure import (
    FAILED as _MEASURE_FAILED, STOPPED as _MEASURE_STOPPED, measure_run,
)
from squidxplorer._montage import _area_downsample
from squidxplorer._napari_view import full_res_level
from squidxplorer._operations import operator_label
from squidxplorer._plate_overview import (
    _CELL, _box_union, _fit_box, _fit_cell, _mosaic_boxes, content_box,
)
from squidxplorer._progress import FOV_UNIT, PREVIEW_LABEL, RunProgress, unit_plan
from squidxplorer._tsctx import HANDLES
from squidxplorer.contract import field_path

log = get_logger("viewer")

_VIEWER_WORKERS = min(6, _default_workers())   # cap: RAM grows linearly, throughput sublinearly

_MIN_PREVIEW_BOX_PX = 4    # smallest FOV box (of _CELL) the RAW preview will mosaic

#: default meaning "build the cache from the reader"; ``None`` means "no cache"
_CACHE_AUTO = object()

class _OperatorWorker(QThread):
    """Run an operator over the plate and persist it as a multiscale OME-Zarr plate."""

    tileReady = Signal(int, int, str, object, object)   # (ri, ci, well_id, (C,h,w) tile, box|None)
    progress = Signal(int, int)                 # (done, total) wells
    runProgress = Signal(object)                # ProgressReport, in the engine's own unit
    streamEnded = Signal()                      # every well landed -> recomposite the whole plate
    writtenReady = Signal(str)                  # path of the written plate.ome.zarr
    wellFailed = Signal(int, int)               # (ri, ci) of a well skipped on a read error
    resultReady = Signal(str, int, object)      # (region, fov, (C, Nz, Y, X) | (C, Y, X))
    failed = Signal(str)                        # whole-run failure (not a per-well skip)
    finished_ok = Signal()

    def __init__(self, operator: str, reader, meta, fov_index: dict, out_dir: str,
                 regions=None, save: bool = True, n_fovs=1, operator_kwargs=None):
        super().__init__()
        self._operator = operator
        self._reader, self._meta = reader, meta
        self._fov_index = fov_index
        self._out_dir = out_dir
        self._regions = regions          # None = whole plate; a list = subset preview (those wells only)
        self._save = save                # False = PREVIEW: compute + push to the viewer, write NOTHING
        self._n_fovs = n_fovs            # None = every FOV per well -> coordinate-placed mosaic tiles
        # carried on BOTH branches of run(): preview and save must run with the same parameters
        self._operator_kwargs = dict(operator_kwargs or {})
        # a region operator's fused mosaic IS the cell, so it gets no per-FOV sub-boxes
        from squidxplorer import is_region_operator

        self._region_op = is_region_operator(self._operator)
        self._boxes = {} if (self._region_op or n_fovs == 1) else _mosaic_boxes(meta)
        self._total = len(regions) if regions is not None else len(meta["regions"])
        _units, _unit = unit_plan(meta, regions, region_op=self._region_op, n_fovs=n_fovs)
        self._progress = RunProgress(operator_label(operator), _units, _unit)
        self._channels = [c["name"] for c in meta["channels"]]
        self._dtype = np.dtype(meta["dtype"])
        self._lock = threading.Lock()             # guards _done (on_well runs on writer threads)
        self._done = 0
        self._seen_fovs: dict[tuple, set] = {}    # (ri,ci) -> FOVs composited so far, for progress
        self._failed_regions: set = set()         # regions whose fields raised
        self._stop = threading.Event()            # set by the window to end the run cleanly
        self._said_z_dropped = False              # see _z_dropped_note: once per run, not per FOV

    @property
    def mosaic_boxes(self) -> dict:
        """``{(region, fov): (top, left, h, w)}`` this run composites into ({} = single-tile path)."""
        return self._boxes

    @property
    def landed(self) -> int:
        """Wells that actually produced pixels."""
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
        """Composite one written field into the plate thumbnail (runs on write_plate writer threads)."""
        info = self._fov_index[region]
        ri, ci, well_id = *info["rc"], info["well_id"]
        well = image[0, :, 0]  # (C, Y, X) -- the plate thumbnail's plane
        box = self._boxes.get((region, fov))
        n_c = len(self._channels)

        # downsample OUTSIDE the lock so the expensive part stays parallel
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
            if was_empty:                          # count WELLS, not fields
                self._done += 1
            done = self._done
            self._progress.tick(time.monotonic())
            report = self._progress.report()
        self.tileReady.emit(ri, ci, well_id, raw, box)
        self.progress.emit(done, self._total)
        self.runProgress.emit(report)
        self.resultReady.emit(region, fov, self._result_pixels(image, well))

    def _result_pixels(self, image, well):
        """What goes to the layer: a region operator's ``(C, Nz, Y, X)`` volume, else one plane."""
        if self._region_op:
            return image[0]                       # (C, Nz, Y, X)
        self._z_dropped_note(int(image.shape[2]))
        return well                               # (C, Y, X)

    def _z_dropped_note(self, depth: int) -> None:
        """Say once per run that a per-FOV operator's extra planes are not reaching the layer."""
        if depth <= 1 or self._said_z_dropped:
            return
        self._said_z_dropped = True
        log.info("%s: the layer shows z plane 0 of %d. A per-FOV operator's mosaic is re-fused "
                 "for display one plane at a time, and only that plane is kept; the WRITTEN "
                 "plate carries all %d. Stitch the region to see the whole volume in 3D.",
                 self._operator, depth, depth)

    def _on_error(self, region, fov, exc):
        """Skip a failed well, mark its dot failed, and keep the run alive."""
        log.warning("%s: region %s FOV %s was skipped — %s: %s",
                    self._operator, region, fov, type(exc).__name__, exc)
        with self._lock:
            self._failed_regions.add(region)
        info = self._fov_index.get(region)
        if info is not None:
            self.wellFailed.emit(*info["rc"])

    #: the in-flight run's recorder, or None between runs
    _recorder = None

    def run(self):
        target = _run_scope.describe_run_target(self._regions, total=self._total) or self._operator
        # capture print() for the run's duration (tilefusion reports with bare print);
        # scoped to the run, not this thread — the region loop prints from pool threads
        with capture_stdout_to_log(), \
                measure_run(self._operator, target, n_targets=self._total) as _run_metrics:
            _run_metrics.note(surface="gui", save=self._save)
            # first paint is reported by the window; cleared in finally so a late tile finds nothing
            self._recorder = _run_metrics
            try:
                self._run_body(_run_metrics)
            finally:
                self._recorder = None

    def report_first_paint(self, seconds: float) -> None:
        """Record that this run's first tile was drawn ``seconds`` after the user asked for it."""
        r = self._recorder
        if r is not None:
            r.first_paint(seconds)

    @property
    def progress_report(self):
        return self._progress.report()

    def _run_body(self, _run_metrics):
        # say 0 of N before any work, so the bar is determinate from its first frame
        self.runProgress.emit(self._progress.report())
        try:
            # the ONE save-vs-preview dispatch; this worker only adds Qt signals around it
            result = run_operator_once(
                self._reader, operator=self._operator, save=self._save, owed=self._total,
                out_dir=self._out_dir, regions=self._regions, n_fovs=self._n_fovs,
                # a region operator runs one well at a time: peak memory is workers x one fused mosaic
                workers=1 if self._region_op else _VIEWER_WORKERS,
                parameters=self._operator_kwargs, tiff=False,
                on_well=self._on_well, on_error=self._on_error, stop=self._stop.is_set)
            if result.stopped:
                _run_metrics.finish(_MEASURE_STOPPED, "stopped by the window")
                return  # window closing / re-opening; drop out cleanly (no final/written emit)
            self.streamEnded.emit()
            if self._save:
                # an acquisition-format save lands beside the source, not under out_dir
                self.writtenReady.emit(result.out_path
                                       or str(Path(self._out_dir) / "plate.ome.zarr"))
            _run_metrics.finish(result.outcome, result.detail)
            self.finished_ok.emit()
        except Exception as e:
            # catch so the QThread ends via `failed`, not an unhandled thread exception
            _run_metrics.finish(_MEASURE_FAILED, f"{type(e).__name__}: {e}")
            self.failed.emit(f"{type(e).__name__}: {e}")


class _MinervaWorker(QThread):
    """Export the selection to Minerva-ingestable files, then start Minerva Author.

    A launch failure never invalidates a successful export: the OME-TIFF on disk is the
    deliverable, so a failed launch still emits exported(paths) + launched(False).
    """
    progress = Signal(int, int)          # (done, total) regions exported
    exported = Signal(object)            # [(ome_path, story_path), ...]
    launched = Signal(bool)              # did a Minerva server end up answering?
    failed = Signal(str)
    finished_ok = Signal()

    def __init__(self, reader, selection, out_dir, z_operator: str, time_point: int = 0, launch: bool = True,
                 luts=None):
        super().__init__()
        self._reader = reader
        self._selection = list(selection)
        self._out_dir = out_dir
        self._z_operator = z_operator
        self._t = time_point
        self._launch = launch
        # snapshotted by the caller on the GUI thread; this thread must not touch napari layers
        self._luts = dict(luts) if luts else None
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        from squidxplorer import _minerva
        try:
            pairs = []

            def on_progress(done, total):
                self.progress.emit(done, total)

            # export region by region: a stop between regions keeps every file already written
            grouped = _minerva.group_selection(self._selection)
            for i, (region, fovs) in enumerate(grouped.items()):
                if self._stop.is_set():
                    break
                pairs.extend(
                    _minerva.export_selection(
                        self._reader, [(region, f) for f in fovs], self._out_dir,
                        time_point=self._t, z_operator=self._z_operator, luts=self._luts,
                    )
                )
                on_progress(i + 1, len(grouped))
            self.exported.emit(pairs)
            if pairs and self._launch and not self._stop.is_set():
                # should_stop: the liveness wait is long and closeEvent joins this thread
                self.launched.emit(
                    _minerva.launch_minerva(pairs[0][1], should_stop=self._stop.is_set))
            self.finished_ok.emit()
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class _MinervaRenderWorker(QThread):
    """Render already-exported pairs into viewable Minerva exhibits, off the GUI thread."""

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
        from squidxplorer import _minerva
        done = []
        try:
            for i, (ome, story) in enumerate(self._pairs):
                if self._stop.is_set():
                    break
                # exhibit dir sits beside the export: <stem>_rendered/
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
            # report what landed as well as what broke
            self.rendered.emit(done)
            self.failed.emit(f"{type(e).__name__}: {e}")


class _MosaicWorker(QThread):
    """Fuse one region's FOVs into a mosaic per channel, off the GUI thread."""

    ready = Signal(str, str, object, object, object)
    #        region, channel, LEVELS (pyramid), bbox_um|None, contrast window (lo, hi)|None
    problem = Signal(str)
    finished_count = Signal(int)

    def __init__(self, reader, meta, region, channels, parent=None, time_point=0):
        super().__init__(parent)
        self._reader, self._meta = reader, meta
        self._region = region
        self._channels = list(channels)
        #: which timepoint this mosaic is of
        self._t = int(time_point)
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def _seed_window(self, channel, levels, auto_window) -> tuple:
        """The contrast seed: an RGB component gets the FILE's own full range (identical across
        the three primaries, so additive blending reconstructs the file's exact color — a
        per-channel percentile window would tint it); everything else keeps the percentiles."""
        probe = getattr(self._reader, "is_rgb_component", None)
        if probe is not None and probe(channel):
            try:
                info = np.iinfo(np.dtype(self._meta["dtype"]))
                return (float(info.min), float(info.max))
            except (TypeError, ValueError, KeyError):
                pass
        return auto_window(levels, True)

    def run(self):
        from squidxplorer._mosaic_source import fuse_region_pyramid, mosaic_bbox_um
        # the same contrast seeding function add_mosaic calls: one contrast rule per quantity
        from squidxplorer._napari_view import _auto_window_for

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
                # a lazy multiscale pyramid of (z, y, x) levels; only the visible (level, z)
                # is ever materialised
                res = fuse_region_pyramid(self._reader, self._meta, self._region, ch,
                                          time_point=self._t)
            except Exception as exc:                # noqa: BLE001 - reported, never swallowed
                self.problem.emit(f"{self._region}/{ch}: {type(exc).__name__}: {exc}")
                continue
            if res is None:
                self.problem.emit(
                    f"{self._region}: no stage positions / pixel size — mosaic not derivable."
                )
                continue
            levels, _step, _nz = res
            # the contrast seed decodes every FOV of the region, so it runs here, never on the UI thread
            window = self._seed_window(ch, levels, _auto_window_for)
            self.ready.emit(self._region, ch, levels, bbox, window)
            n += 1
        self.finished_count.emit(n)

class _FocusWorker(QThread):
    """Rank one FOV's z planes by Tenengrad sharpness, off the GUI thread."""

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
        from squidxplorer.projection import _tenengrad

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
            # never return a default z when nothing could be read
            self.problem.emit(
                f"{self._region}:{self._fov} — not one z plane of {self._channel} could be "
                f"read, so there is no sharpest plane. ({'; '.join(failures[:3])})")
            return
        note = ("" if not failures else
                f" ({len(failures)} of {len(self._meta['z_levels'])} planes were unreadable "
                f"and were skipped)")
        self.ready.emit(int(best_z_i), note)


class _SpotWorker(QThread):
    """Run spot detection on the plane currently on screen, off the GUI thread."""

    progress = Signal(int, int)                # (stages done, stages total)
    stageChanged = Signal(str)                 # the stage's name
    ready = Signal(str, str, object, object, object, int)
    # ^ (region, channel, labels (H,W) int32, centroids (N,2) float, bbox_um|None, count)
    problem = Signal(str)                      # a named failure: "<region>/<channel>: ..."
    cancelled = Signal()
    finished_count = Signal(str, str, int)     # (region, channel, count)

    def __init__(self, region, channel, data, z_index, bbox_um, params=None, parent=None,
                 algorithm=None):
        super().__init__(parent)
        self._region, self._channel = region, channel
        self._data, self._z = data, z_index
        self._bbox_um = bbox_um
        self._params = params
        # (name, segment) — resolved by the CALLER off the registry so the button's label, its
        # params and the run agree on which algorithm this is; None falls back here.
        self._algorithm = algorithm
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        from squidxplorer._spots import SpotDetectionCancelled, detect_spots

        where = f"{self._region}/{self._channel}"
        algorithm, segment = self._algorithm or nuclei_operator()
        # the progress denominator is whatever the running algorithm reports
        reported_total = [0]

        def _stage(name, done, total):
            reported_total[0] = int(total)
            self.stageChanged.emit(name)
            self.progress.emit(int(done), int(total))

        try:
            plane = _full_res_mip(self._data)          # segment the MIP over z, not one z-plane
            log.info("%s: detecting nuclei with %s on a %s MIP", where, algorithm, plane.shape)

            res = detect_spots(
                plane, self._params, segment=segment,
                on_stage=_stage,
                should_stop=self._stop.is_set,
            )
        except SpotDetectionCancelled:
            self.cancelled.emit()
            return
        except Exception as exc:                   # noqa: BLE001 - NAMED, never swallowed
            # log AND banner: the banner is where the user looks, the log is the copyable record
            log.error("%s: spot detection failed — %s: %s", where, type(exc).__name__, exc)
            self.problem.emit(f"{where}: spot detection failed — {type(exc).__name__}: {exc}")
            return

        # close on the same denominator the run reported
        total = reported_total[0] or len(_spot_stages())
        self.progress.emit(total, total)
        self.stageChanged.emit("done")
        log.info("%s: %d nuclei detected (%s)", where, res.count, algorithm)
        self.ready.emit(self._region, self._channel, res.labels, res.centroids,
                        self._bbox_um, res.count)
        self.finished_count.emit(self._region, self._channel, res.count)


def nuclei_operator():
    """``(name, segment)`` the detect-nuclei button runs: ``cellpose`` when the REGISTRY says it
    is available, else ``spot`` (the Otsu-watershed).

    The choice reads ``operator_available`` — the record's own ``requires`` declaration, the
    same one every other run surface selects on — never a private dependency probe. The names
    are the registered operator names, so the button's label, the console line and the panel
    whose values reach the run all agree on which algorithm this is.
    """
    from squidxplorer._engine import operator_available

    if operator_available("cellpose")[0]:
        from squidxplorer._cellpose import OPERATOR_NAME, cellpose_nuclei

        return OPERATOR_NAME, cellpose_nuclei
    from squidxplorer._spots import LAYER_KEY, skimage_watershed

    return LAYER_KEY, skimage_watershed


def _spot_stages():
    """The stage list, imported lazily."""
    from squidxplorer._spots import STAGES

    return STAGES


class _FlatfieldWorker(QThread):
    """Estimate an illumination profile from a spread sample of plate tiles, off the GUI thread."""

    done = Signal(object)     # FlatfieldProfile
    problem = Signal(str)
    stage = Signal(str)

    def __init__(self, reader, meta, channel, *, max_tiles=48, use_darkfield=False, parent=None):
        super().__init__(parent)
        self._reader, self._meta, self._channel = reader, meta, channel
        self._max_tiles = int(max_tiles)
        self._use_dark = bool(use_darkfield)
        self._stop = threading.Event()

    def stop(self):
        """Cancel the reads; the BaSiC solve itself is not interruptible."""
        self._stop.set()

    def run(self):                                    # pragma: no cover - Qt thread
        try:
            from squidxplorer._flatfield import estimate_profile

            meta = self._meta
            z0 = (meta.get("z_levels") or [0])[0]
            fpr = meta.get("fovs_per_region") or {}
            pairs = [(region, int(fov))
                     for region in (meta.get("regions") or [])
                     for fov in (fpr.get(region) or [])]
            if not pairs:
                self.problem.emit("no FOVs to estimate a flat-field from.")
                return
            # spread the sample across the plate, not the first N of one well
            step = max(1, len(pairs) // self._max_tiles)
            sample = pairs[::step][: self._max_tiles]
            tiles = []
            unreadable = []
            for region, fov in sample:
                if self._stop.is_set():
                    return                            # cancelled: no profile, and nothing claimed
                try:
                    tiles.append(np.asarray(self._reader.read(region, fov, self._channel, int(z0))))
                except Exception as exc:              # noqa: BLE001 - one bad tile is not fatal…
                    unreadable.append(f"{region}/{fov}: {type(exc).__name__}: {exc}")
                    continue
                self.stage.emit(f"read {len(tiles)}/{len(sample)} tiles for {self._channel}…")
            if unreadable:
                # …but it is not invisible either
                log.warning("flat-field %s: %d of %d sample tiles were unreadable and were left "
                            "out of the estimate — first: %s",
                            self._channel, len(unreadable), len(sample), unreadable[0])
            if len(tiles) < 3:
                self.problem.emit(
                    f"flat-field estimate needs at least 3 readable tiles for {self._channel}, "
                    f"got {len(tiles)}"
                    + (f" ({len(unreadable)} unreadable, first: {unreadable[0]})."
                       if unreadable else "."))
                return
            self.stage.emit(f"estimating illumination (tilefusion BaSiC) from {len(tiles)} tiles…")
            profile = estimate_profile(np.stack(tiles), use_darkfield=self._use_dark)
            log.info("flat-field: estimated a %s profile from %d tiles (tilefusion BaSiC)",
                     self._channel, len(tiles))
            self.done.emit(profile)
        except Exception as exc:                      # noqa: BLE001 - NAMED to the log, not swallowed
            log.error("flat-field estimate failed for %s: %s", self._channel, exc)
            self.problem.emit(f"{type(exc).__name__}: {exc}")


class _VideoWorker(QThread):
    """Fuse, composite and encode a region's T (or Z) sweep to an .mp4, off the GUI thread.

    A cancelled run emits ``cancelled`` and leaves the partial .mp4 where the user pointed it.
    """

    progress = Signal(int, int)        # (frames done, frames total)
    done = Signal(str, int, float)     # (path, frame count, seconds)
    problem = Signal(str)              # a named failure, never a silent no-op
    cancelled = Signal()

    def __init__(self, reader, meta, region, out_path, *, axis, fps,
                 channels=None, windows=None, rgb_by_channel=None, z_level=0, time_point=0, parent=None):
        super().__init__(parent)
        self._reader, self._meta, self._region = reader, meta, region
        self._out_path = str(out_path)
        self._axis, self._fps = str(axis), int(fps)
        self._channels = list(channels) if channels is not None else None
        self._windows = list(windows) if windows else None
        self._rgb_by_channel = dict(rgb_by_channel or {})
        self._z, self._t = int(z_level), int(time_point)
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):                                    # pragma: no cover - Qt thread
        from squidxplorer._video import record_region

        started = time.perf_counter()
        try:
            path, n = record_region(
                self._reader, self._meta, self._region, self._out_path,
                axis=self._axis, fps=self._fps, channels=self._channels,
                windows=self._windows, rgb_by_channel=self._rgb_by_channel,
                z_level=self._z, time_point=self._t,
                on_frame=lambda d, total: self.progress.emit(int(d), int(total)),
                should_stop=self._stop.is_set,
            )
        except Exception as exc:                      # noqa: BLE001 - NAMED, never swallowed
            if self._stop.is_set():
                # a cancel empties the frame iterator; only here is it knowable which happened
                self.cancelled.emit()
                return
            log.error("movie export failed for %s: %s", self._region, exc)
            self.problem.emit(f"{type(exc).__name__}: {exc}")
            return
        if self._stop.is_set():
            self.cancelled.emit()
            return
        seconds = time.perf_counter() - started
        log.info("movie: wrote %d %s-axis frames of %s to %s in %.1fs",
                 n, self._axis, self._region, path, seconds)
        self.done.emit(path, int(n), float(seconds))


def _full_res_plane(data, z_index):
    """The full-resolution 2-D plane behind a napari layer's ``data``, whatever shape it is in."""
    # the one pyramid rule, shared with every reader of a layer's data (_napari_view.pyramid_levels)
    data = full_res_level(data)

    # trust .ndim when present (keeps a lazy dask/zarr level lazy); materialise once when absent
    ndim = getattr(data, "ndim", None)
    if ndim is None:
        data = np.asarray(data)
        ndim = data.ndim

    # a (z, y, x) stack is indexed at the z on screen; more leading axes is refused below
    if ndim == 3:
        n_z = int(data.shape[0])
        from squidxplorer._contrast import opening_z

        z = opening_z(n_z) if z_index is None else int(z_index)
        data = data[min(max(z, 0), n_z - 1)]

    plane = np.asarray(data)
    if plane.ndim != 2:
        raise ValueError(
            f"expected a 2-D plane to count on, got shape {plane.shape!r}. The layer's data is "
            "neither a pyramid level list, a (z, y, x) stack, nor a (y, x) plane."
        )
    return plane


def _full_res_mip(data):
    """The full-resolution MIP (max over z) behind a napari layer's ``data``."""
    data = full_res_level(data)
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
    """Fast RAW plate preview: one representative z-plane per channel per FOV, composited per cell."""

    tileReady = Signal(int, int, str, object, object)   # (ri, ci, well_id, tile, box|None)
    #: raw fill, not an operator run; ``_run_scope.operator_busy`` reads it
    IS_PREVIEW = True

    runProgress = Signal(object)                # ProgressReport; total is len(_plan), not unit_plan

    streamEnded = Signal()                      # preview complete -> recomposite the whole plate
    failed = Signal(str)                         # a preview that could not finish names why

    def __init__(self, reader, meta, fov_index: dict, order: list, mosaic: bool = True,
                 cache=_CACHE_AUTO, time_point: int = 0):
        super().__init__()
        self._reader, self._meta = reader, meta
        self._fov_index, self._order = fov_index, order
        self._channels = [c["name"] for c in meta["channels"]]
        self._dtype = np.dtype(meta["dtype"])
        self._mosaic = bool(mosaic)
        #: which timepoint this preview is of
        self._t = max(0, int(time_point))
        self._stop = threading.Event()
        # persisted plate cells; for_reader returns None (logging why) when caching is unavailable
        from squidxplorer._platecache import PlateCellCache

        self._cache = (PlateCellCache.for_reader(reader, meta, cell_px=_CELL,
                                                 time_point=self._t)
                       if cache is _CACHE_AUTO else cache)
        # a handed-in cache must be for this pass's timepoint; refuse loudly at construction
        if (self._cache is not None
                and getattr(self._cache, "time_point", self._t) != self._t):
            raise ValueError(
                f"_PreviewWorker(time_point={self._t}) was handed a cache for timepoint "
                f"{self._cache.time_point}: its cells would be published under the wrong frame.")
        self._pending: dict = {}      # region -> the cell being accumulated for the cache
        self.cache_hits = 0           # regions served from the cache
        self.cache_reads = 0          # regions actually read from the acquisition
        self.well_image_hits = 0      # regions served from Squid's saved mosaic_view/wells
        # indeterminate until run() knows the plan; built here so an early progress_report never raises
        self._progress = RunProgress(PREVIEW_LABEL, None, FOV_UNIT)

    def _plan(self) -> list:
        """``[(region, fov, box|None), ...]`` read list; ``box=None`` means the FOV fills its cell."""
        from squidxplorer._placement import cell_boxes, fov_offsets_px

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
        """Emit every region the cache can serve; return the plan entries still to be read."""
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
        log.info("plate preview at t=%d: %d of %d wells served from the cell cache (%s)",
                 self._t, self.cache_hits, len(by_region), self._cache)
        return remaining

    def _seed_from_well_images(self, plan: list) -> list:
        """Serve whole regions from Squid's saved ``mosaic_view/wells`` mosaics.

        One small file read per well instead of its FOV walk; returns the plan entries a
        well image could not serve (absent, corrupt, multi-z, or missing a channel; the
        reasons are logged by ``_wellimage``). Served cells are published to the cell cache
        so the next open replays them without touching the files at all.
        """
        from squidxplorer import _wellimage

        by_region: dict = {}
        for item in plan:
            by_region.setdefault(item[0], []).append(item)
        remaining: list = []
        for region, items in by_region.items():
            stack = None if self._stop.is_set() else _wellimage.load_well_stack(
                self._reader, self._meta, region, self._t)
            planes = None
            if stack is not None:
                planes = [stack.channel_plane(ch) for ch in self._channels]
                if any(p is None for p in planes):
                    planes = None      # a channel the file lacks: this well takes the FOV walk
            if planes is None:
                remaining.extend(items)
                continue
            box = content_box(planes[0].shape, _CELL, _CELL)
            _top, _left, bh, bw = box
            tile = np.stack([_fit_box(p.astype(np.float32), bh, bw)
                             for p in planes]).astype(self._dtype)
            ri, ci = self._fov_index[region]["rc"]
            self.tileReady.emit(ri, ci, region, tile, box)
            if self._cache is not None:
                self._cache.put(region, tile, box)
            self.well_image_hits += 1
        if self.well_image_hits:
            log.info("plate preview at t=%d: %d well(s) seeded from mosaic_view/wells, "
                     "skipping their FOV walk.", self._t, self.well_image_hits)
        return remaining

    def _backfill_well_images(self) -> None:
        """Leave the acquisition as a mosaic_view-saving Squid would have (best-effort).

        Runs after the preview has fully painted, still on this worker thread; a failure
        (read-only mount, anything) is logged and never fails the preview.
        """
        from squidxplorer import _wellimage

        try:
            if not _wellimage.enabled():
                return
            root = _wellimage.acquisition_root(self._reader)
            if root is None or _wellimage.has_well_images(root, self._t):
                return
            _wellimage.write_well_images(self._reader, self._meta, time_point=self._t,
                                         should_stop=self._stop.is_set)
        except Exception as exc:        # noqa: BLE001 - a backfill must not fail a preview
            log.warning("well-image backfill failed (%s: %s); the preview is unaffected.",
                        type(exc).__name__, exc)

    def _remember(self, region: str, box, tile: np.ndarray, expected: int) -> None:
        """Accumulate one FOV into the region's cell, and publish the cell once it is whole."""
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
        """This pass's progress so far, for a consumer that arrives mid-pass."""
        return self._progress.report()

    def run(self):
        # capture print() for the pass's duration; scoped to the pass, not this thread,
        # because the reads run on a ThreadPoolExecutor
        with capture_stdout_to_log():
            self._run_body()

    def _run_body(self):
        try:
            from collections import Counter
            from concurrent.futures import ThreadPoolExecutor
            zs = self._meta["z_levels"]
            z_mid = zs[len(zs) // 2]      # a mid-stack plane is a fair single-plane preview
            plan = self._seed_from_well_images(self._replay_cached(self._plan()))
            # the denominator is the plan that survived the cache: what is left IS the work
            self._progress = RunProgress(PREVIEW_LABEL, len(plan), FOV_UNIT)
            # say 0 of N before the first read
            self.runProgress.emit(self._progress.report())
            per_region = Counter(item[0] for item in plan)

            def load(item):
                region, fov, box = item
                # poll stop FIRST: every item is submitted up front, so this is what
                # turns a not-yet-started item into a no-op
                if self._stop.is_set():
                    return None
                h, w = (_CELL, _CELL) if box is None else (box[2], box[3])
                fit = _fit_cell if box is None else (lambda a: _fit_box(a, h, w))
                return region, box, [fit(self._reader.read(region, fov, ch, z_mid,
                                                            time_point=self._t)
                                         .astype(np.float32)) for ch in self._channels]

            # ex.map submits every item before the first result; the poll in load cancels the work
            with ThreadPoolExecutor(max_workers=_VIEWER_WORKERS) as ex:
                for done in ex.map(load, plan):                # plate order preserved
                    if self._stop.is_set() or done is None:
                        return
                    region, box, tiles = done
                    ri, ci = self._fov_index[region]["rc"]
                    tile = np.stack(tiles).astype(self._dtype)
                    self.tileReady.emit(ri, ci, region, tile, box)
                    self._remember(region, box, tile, per_region[region])
                    # no lock: ex.map yields on this thread, so every tick is serialised
                    self._progress.tick(time.monotonic())
                    self.runProgress.emit(self._progress.report())
            if not self._stop.is_set():
                self.streamEnded.emit()   # one clean recomposite
                # compact the finished generation AFTER streamEnded; a partial pass never gets here
                if self._cache is not None and not self._cache.packed:
                    self._cache.pack(self._order)
                # after everything painted: make the acquisition mosaic_view-complete
                self._backfill_well_images()
        except Exception as exc:
            # best-effort is not silent: finalise the tiles that landed, then name the failure
            if not self._stop.is_set():
                self.streamEnded.emit()
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class _ComputedPlateWorker(QThread):
    """Read a previously-written OME-Zarr plate back into the viewer (no recompute)."""

    tileReady = Signal(int, int, str, object, object)   # (ri, ci, well_id, (C, h, w) tile, box)
    progress = Signal(int, int)
    streamEnded = Signal()                      # plate fully loaded -> recomposite globally
    finished_ok = Signal()
    failed = Signal(str)

    def __init__(self, base, wells, coarse_lvl, dtype, time_point: int = 0):
        self._time_point = int(time_point)   # which timepoint the plate is showing
        super().__init__()
        self._base = base                 # plate.ome.zarr path
        self._wells = wells               # [(well_id, wellpath, fov, ri, ci, flat_idx)]
        self._coarse = coarse_lvl
        self._dtype = dtype
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def _read(self, wellpath, fov, level, time_point: int = 0):
        # through the shared handle pool, not a bare ts.open (see _tsctx)
        arr = HANDLES.get(field_path(self._base, wellpath, fov, level))
        # clamped so a stale slider position cannot index off the end of a shorter acquisition
        t_idx = max(0, min(int(time_point), arr.shape[0] - 1))
        return np.asarray(arr[t_idx, :, 0].read().result())   # (C, y, x) at this t, z=0

    def run(self):
        try:
            n = len(self._wells)
            for i, (wid, wpath, fov, ri, ci, _idx) in enumerate(self._wells, 1):
                if self._stop.is_set():
                    return
                coarse = self._read(wpath, fov, self._coarse, self._time_point)   # thumbnail (C,y,x)
                # by the field's own aspect ratio, not squashed into the square
                box = content_box(coarse.shape[1:], _CELL, _CELL)
                _, _, bh, bw = box
                tile = np.stack([_fit_box(plane.astype(np.float32), bh, bw) for plane in coarse])
                self.tileReady.emit(ri, ci, wid, tile.astype(self._dtype), box)
                self.progress.emit(i, n)
            if not self._stop.is_set():
                self.streamEnded.emit()   # every well in the store -> one global-window recomposite
                self.finished_ok.emit()
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")
