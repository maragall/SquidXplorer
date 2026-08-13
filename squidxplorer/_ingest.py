"""The acquisition-open pipeline, out of ``_viewer``: ingest, the raw-preview lifecycle, and the
loupe-source bookkeeping. Every function takes the ``PlateWindow`` and works on ITS state — one
bookkeeping, owned by the window, so nothing here can drift out of step with it. The window keeps
one-line forwarders (and the Qt slots), so every caller and test surface is unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from qtpy.QtWidgets import QApplication

from squidxplorer._logpane import get_logger
from squidxplorer._montage import _hex_to_rgb01
from squidxplorer._plate import PlateBuildError, build_plate
from squidxplorer._plate_overview import (
    PlateOverview, _RawLoupeSource, _fov_of_well, _mosaic_boxes, resolve_plate_root,
)
from squidxplorer._plate_shape import PlateShapeError
from squidxplorer._workers import _PreviewWorker

log = get_logger("ingest")


# -- open an acquisition (no processing yet — that's the Process menu) --------------------------
def ingest(win, path: str) -> None:
    from squidxplorer import open_reader

    p, is_plate = resolve_plate_root(path)
    if is_plate:
        win._readout.setText("this is already a written plate — drop a raw Squid acquisition")
        return
    # stop any in-flight run/preview/export and clear prior state before opening a new
    # acquisition. _stop_minerva matters as much as the other two: a Minerva worker left
    # running holds the OLD reader and would keep exporting (and launching) against an
    # acquisition the window no longer shows.
    win._stop_worker()
    win._stop_preview()
    win._stop_minerva()
    win._reader = win._meta = None
    win._fov_index = {}
    win._selected_regions = []   # wells picked on the plate (IMA-221); scopes an operator run
    win._current_well = None
    win._current_fov = 0
    win._enable_operators(False)
    if win._overview is not None:
        win._release_loupe_sources()   # BOTH read threads, before dropping their owner
        win._overview.setParent(None)
        win._overview.deleteLater()
        win._overview = None
    win._readout.setText("scanning acquisition …")
    QApplication.processEvents()
    try:
        reader = open_reader(str(p))
        meta = reader.metadata
    except Exception as e:   # not a Squid acquisition / unreadable -> report, don't crash the app
        win._readout.setText(f"not a readable Squid acquisition: {e}")
        win._drop.show()
        return
    # Resolve the layout format ONCE: an explicit override wins, then the declared field, then
    # inference from the well ids (IMA-219 — two real acquisitions carry no format at all).
    # Never fatal: an un-inferable plate keeps the declared value and falls through the guard.
    # Resolve the sample holder ONCE (IMA-214). build_plate handles wells AND slides: a slide
    # carrier is a Plate whose cells are the freeform region ids, so a glass-slide/tissue
    # acquisition reaches this widget by the same path a 384wp does. It also reconciles a
    # declared format against the MEASURED stage pitch, so a mis-declared plate cannot lay out
    # at the wrong scale.
    try:
        plate = build_plate(meta, override=win._plate_format_override)
    except (PlateShapeError, PlateBuildError) as e:
        win._readout.setText(f"cannot lay out this acquisition: {e}")
        win._drop.show()
        return
    win._plate = plate
    win._plate_format = plate.format_name
    win._reader, win._meta = reader, meta
    win._acq_name = Path(p).name
    win._acq_path = Path(p)
    win._processed_plate = None
    win._viewer_manager.set_dataset(reader, meta)   # every spawned window shares this reader
    rows, cols, wells, order = plate.viewer_grid()
    for idx, region in enumerate(order):
        win._fov_index[region] = {"idx": idx, "well_id": region, "rc": plate.cell_index(region)}

    win._order = order                          # well order = the detail's FOV-slider order
    # A freeform holder places its cells by GEOMETRY (IMA-253): the plate hands over one
    # rectangle per region, in grid units, and the overview draws exactly those. A well plate
    # returns None here and keeps the uniform grid it has always had.
    cl = plate.cell_layout() if hasattr(plate, "cell_layout") else None
    layout = ({plate.cell_index(cid): rect for cid, rect in cl.items()} if cl else None)
    win._overview = PlateOverview(rows, cols, wells, layout=layout)
    # Carrier art behind the cells (IMA-220). Hand over the PLATE, not its name: `plate` is what
    # build_plate RESOLVED (measured pitch beat the 2x2's mis-declared "384 well plate"), so the
    # background can only ever be drawn at the same scale the grid is laid out at.
    win._overview.set_carrier(plate)
    # DEEP ZOOM: arm the tile overlay for this acquisition. Fail-quiet by contract — an
    # acquisition with no usable stage positions keeps the montage and nothing else changes.
    if win._overview.set_tile_source(reader, meta):
        g = win._overview._ladder.geometry
        log.info("deep zoom armed: %d rungs, %.3f-%.1f um/px, %d tiles at fit",
                 len(g), g.levels[0].scale_um_per_px, g.levels[-1].scale_um_per_px,
                 g.worst_case_tiles)
    else:
        log.info("deep zoom not armed (no usable stage positions) — montage only")
    win._selected_regions = []                  # a new acquisition starts with nothing picked
    win._overview.hovered.connect(win._on_hover)
    win._overview.wellActivated.connect(win.activate_well)
    win._overview.selectionChanged.connect(win._on_selection_changed)
    win._overview.marqueeSelected.connect(win._on_marquee_selected)
    # The loupe's source is chosen by which layer the plate SHOWS, so it follows the plate
    # rather than being re-pointed by hand at each of the six places the layer moves.
    win._overview.activeLayerChanged.connect(lambda _k: win._update_loupe_source())
    win._plate_mode = "raw"                     # a freshly-opened plate shows raw previews
    win._plate_title.setText(f"{win._acq_name}   ·   raw")   # bottom-left plate-pane title
    win._op_stack.reset()                       # fresh layer stack (base only)
    win._active_op_key = None
    if getattr(win, "_raw_btn", None):
        win._raw_btn.hide()                     # raw view on open -> nothing to return from
    win._refresh_layers_tab()
    win._drop.hide()
    win._left_l.addWidget(win._overview, 1)   # fills the pane and self-fits — no scrollbars
    win._declare_channel_axis(meta["channels"], meta["dtype"])

    # Hand the plate's region order to the SINGLE OWNER. Announcing it is what puts the red
    # ROI frame on region 0 — one move, not several calls that could each be forgotten on some
    # path.
    #
    # Cleared first so the announce always happens: re-opening an acquisition whose region ids
    # match the previous one would otherwise be a no-op move and every surface reading the
    # cursor would keep pointing at the OLD plate's region.
    win._cursor.set_order([])
    win._cursor.set_order(order)

    win._enable_operators(True)

    # The loupe works from the moment the folder opens — the raw layer's real pixels are the
    # acquisition's own TIFFs, the same planes the preview below is about to downsample. No
    # operator run is required to look closely at a well.
    win._loupe_sources = {"raw": _RawLoupeSource(
        reader, meta, lambda w: _fov_of_well(w, meta.get("fovs_per_region")))}
    win._update_loupe_source()

    # The mosaic geometry is known the moment the acquisition opens — it is pure arithmetic on
    # coordinates.csv — so hand it to the plate NOW rather than waiting for an operator run
    # (IMA-249: it was only ever set from run_operator, which is why the plate looked like a
    # grid of lone frames until something was run). The preview below composites into exactly
    # these boxes.
    win._overview.set_mosaic_boxes(_mosaic_boxes(meta))

    # Size the timepoint bar to what was just ingested. set_count hides it at n_t == 1 and
    # clamps the position, so re-ingesting a SHORTER acquisition cannot leave the bar pointing
    # past the end. It does not fire the callback: an ingest is not a user gesture.
    win._time_point_bar.set_count(int(meta.get("n_t", 1) or 1))
    # set_count CLAMPS the position, so read it back rather than assuming it survived: a
    # re-ingest onto a shorter acquisition moves the bar, and the loupe must move with it.
    win._overview.set_time_point(win.time_point)

    # fast RAW preview: fill the plate with downsampled thumbnails immediately (grey dots),
    # in the SAME row-major order the operator will later process them in.
    win._start_preview(reader, meta, order)
    # top-left = STATUS (what's happening / what's shown); the plate name is the pane title.
    # NOT "live". This is a POST-ACQUISITION tool: nothing here is streaming off a scope --
    # the acquisition is finished and on disk, and calling it live invited exactly the wrong
    # mental model of what the operators below are doing.
    # Multi-FOV policy (IMA-187): an operator run processes EVERY FOV and composites them into
    # the well's cell by stage coordinate. The raw preview above is still one FOV per well (it
    # reads a single plane per well precisely to stay fast), so say which one you're looking at.
    multi = sum(1 for r in order if len(meta["fovs_per_region"][r]) > 1)
    note = (f" · {multi} multi-FOV region(s), previewing as mosaics" if multi else "")
    win._readout.setText(
        f"{len(win._fov_index)} wells loaded · double-click a well, or Shift-drag to open "
        f"several{note}")


# -- the raw-preview lifecycle ------------------------------------------------------------------
def start_preview(win, reader, meta, order, *, time_point: int):
    """Start the raw preview over *order*, fully wired. THE only place a preview is built.

    Extracted because there were three byte-identical five-line copies of this (first ingest,
    the tab re-scope, and the return-to-raw resume), and the progress wiring below had to land
    on all three or the bar would appear on some entry paths and not others. One constructor,
    one set of connections, no third chance to disagree.

    ``time_point`` arrives from the window's forwarder, which is the ONE place the plate's bar
    reaches the pixels: every entry path must preview the frame the bar is showing. The worker
    carries the same t into its cell cache, so a revisited timepoint is a cache HIT and not a
    re-read (`_platecache.PlateCellCache`).
    """
    win._preview = _PreviewWorker(reader, meta, win._fov_index, order, time_point=time_point)
    win._preview.tileReady.connect(win._on_preview_tile)
    win._preview.streamEnded.connect(lambda: win._recomposite("raw"))
    win._preview.failed.connect(win._on_preview_failed)
    # The preview reports on the SAME channel an operator run does, so the one bar covers it
    # ("even if it's preview"). Published straight through: the plate window is only a relay
    # here, because the preview has no status line of its own to feed.
    win._preview.runProgress.connect(win._publish_progress)
    # QThread.finished, not streamEnded: at streamEnded the thread is still running, so
    # _clear_progress_if_idle would see it and decline. This also covers the failed and the
    # stopped preview, which never reach streamEnded at all.
    win._preview.finished.connect(win._clear_progress_if_idle)
    win._preview.start()
    return win._preview


def stop_preview(win) -> None:
    win._retire(win._preview)
    win._preview = None


# -- the loupe-source bookkeeping (IMA-208): layer key -> _LoupeSource backing its pixels -------
def release_loupe_sources(win) -> None:
    """Drop every source AND join the read thread that serves them.

    The one call every "the plate is being replaced" path must make. Assigning
    ``win._loupe_sources = {}`` (which _open_computed did) only forgets the sources: the
    _LoupeWorker QThread lives on the OVERVIEW, so the old overview walked off with a running
    thread and its ~35 MB plane cache on every plate open — confirmed still isRunning() after
    the overview was replaced. Only PlateOverview.shutdown() stops and joins it.

    It goes through ``shutdown()`` rather than ``set_loupe_source(None)`` because the overview
    owns TWO threads and the loupe worker is only one of them: this function's own sentence
    above ("the one call every 'the plate is being replaced' path must make") was true of the
    loupe and false of the tile fetcher, which outlived two of the three replacement paths."""
    if win._overview is not None:
        win._overview.shutdown()
    win._loupe_sources = {}


def set_loupe_source(win, layer_key, source) -> None:
    win._loupe_sources[layer_key] = source
    update_loupe_source(win)


def drop_loupe_source(win, layer_key) -> None:
    win._loupe_sources.pop(layer_key, None)
    update_loupe_source(win)


def update_loupe_source(win) -> None:
    """Point the plate at the source for whatever layer is on screen right now."""
    if win._overview is None:
        return
    active = getattr(win._overview, "_active", "raw")
    source = win._loupe_sources.get(active)
    if source is win._overview._loupe_src:
        return                                   # unchanged: don't churn the worker thread
    colors = None
    if win._meta and win._meta.get("channels"):
        colors = np.stack([_hex_to_rgb01(c["display_color"]) for c in win._meta["channels"]])
    win._overview.set_loupe_source(source, colors)
