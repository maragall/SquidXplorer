"""The mosaic load/playback pipeline of a RegionViewer: one load at a time, frames gated.

Extracted from ``_region_viewer`` (2026-08-14). This cluster owns two rules and they move
INTACT (the reload-vs-different-region rule is settled in ``docs/rendering-contract.md``):

* **Generation dropping.** Every load bumps ``win._load_gen`` and every result carries the
  generation it was made for; a plane or completion arriving for a superseded load is dropped
  at the door (:func:`is_current_load`). ``load_mosaic`` must never ``wait()`` on the worker it
  supersedes — ``stop()`` only sets an Event — and the test that pins that inspects THIS
  module's source.
* **The frame gate.** :func:`frame_done` opens the playback gate only once the mosaic is on
  screen, so the region slider and the timepoint bar request the next frame no faster than
  frames actually land.

Functions over the window (the ``_ingest`` precedent), with the state attributes staying ON the
window (``_load_gen``, ``_worker``, ``_shown_region``, ``_result_region``, ``_retired_workers``):
tests read ``win._load_gen`` by name, and ``tests/test_viewer.py`` drives ``_on_plane`` unbound
over a duck shell, both of which a controller object holding the state would break.
``RegionViewer`` keeps thin delegates for the same reason.

The ``_MosaicWorker`` NAME is resolved through ``squidxplorer._viewer`` at call time: that
module attribute is the seam tests monkeypatch with stand-ins.
"""

from __future__ import annotations

from typing import Optional

from squidxplorer import _measure
from squidxplorer._worker_lifecycle import launch as _launch_worker

_RAW_OP = "raw"


def load_mosaic(win, region: Optional[str]) -> None:
    """Fuse one region's FOVs into this window's napari pane, one layer per channel."""
    pane = win._pane
    if pane is None or not getattr(pane, "ok", False):
        return
    if win._reader is None or win._meta is None or not region:
        return
    from squidxplorer._viewer import _MosaicWorker

    win._load_gen = int(getattr(win, "_load_gen", 0)) + 1
    gen = win._load_gen
    prior = win._worker
    if prior is not None and prior.isRunning():
        prior.stop()
        retire_worker(win, prior)

    if win._result_region is not None and win._result_region != str(region):
        win._drop_result_layers(f"this window moved from {win._result_region} to {region}")
    if win._shown_region != str(region):
        pane.mosaic.remove_op(_RAW_OP)
    channels = [c["name"] for c in win._meta["channels"]]
    w = _MosaicWorker(win._reader, win._meta, region, channels, parent=win,
                      time_point=win.time_point)
    _launch_worker(
        win, w, slot="_worker",
        on_problem=win._say,
        on_finished=lambda: worker_ended(win, w),
        signals={
            "ready": lambda r, ch, levels, bbox, window:
                win._on_plane(r, ch, levels, bbox, window, gen=gen),
            "finished_count": lambda n: win._on_done(region, n, gen=gen),
        })


def worker_ended(win, worker) -> None:
    """A load's thread has ended. Drop every reference to it, ours and Qt's."""
    if win._worker is worker:
        win._worker = None
    forget_worker(win, worker)
    try:
        worker.deleteLater()
    except RuntimeError:
        pass


def retire_worker(win, worker) -> None:
    """Let a superseded worker die on its own time, without dropping it on the floor."""
    retired = getattr(win, "_retired_workers", None)
    if retired is None:
        retired = win._retired_workers = []
    retired.append(worker)


def forget_worker(win, worker) -> None:
    retired = getattr(win, "_retired_workers", None)
    if retired is not None and worker in retired:
        retired.remove(worker)


def is_current_load(win, gen: int) -> bool:
    """Whether *gen* is the load this window is still waiting for."""
    return int(gen) == int(getattr(win, "_load_gen", 0))


def on_plane(win, region: str, channel: str, levels, bbox_um, window=None,
             gen: Optional[int] = None) -> None:
    pane = win._pane
    if pane is None or not getattr(pane, "ok", False):
        return
    if gen is not None and not is_current_load(win, gen):
        return
    if win._cursor is not None and win._cursor.region != region:
        return
    from squidxplorer._napari_pane import _colormap_for
    from squidxplorer._region_viewer import _crop_levels_to_bbox

    add_levels, add_bbox = levels, bbox_um
    add_window = window
    if win._roi_bbox is not None and bbox_um is not None:
        cropped = _crop_levels_to_bbox(levels, bbox_um, win._roi_bbox)
        if cropped is not None:
            add_levels, add_bbox = cropped
            add_window = None
        else:
            win._say("ROI does not overlap this region — showing the whole region.")

    pane.mosaic.add_mosaic(
        _RAW_OP, channel, add_levels,
        contrast_limits=add_window,
        colormap=_colormap_for(channel),
        multiscale=True,
        bbox_um=add_bbox,
        z_scale_um=(win._meta or {}).get("dz_um"),
    )
    if win.open_clock is not None:
        win.open_clock.first_layer()


def on_done(win, region: str, n: int, gen: Optional[int] = None) -> None:
    pane = win._pane
    if pane is None or not getattr(pane, "ok", False):
        return
    if gen is not None and not is_current_load(win, gen):
        return
    if n == 0:
        pane.say(f"{region}: no mosaic could be built (see the message above).")
        try:
            pane.mosaic.remove_op(_RAW_OP)
        except Exception:                        # noqa: BLE001 - already gone is fine
            pass
        win._shown_region = None
        if win.open_clock is not None:
            win.open_clock.finish(_measure.FAILED, f"{region}: no mosaic could be built")
        frame_done(win)
        return
    pane.say("")
    first_look = win._shown_region != str(region)
    try:
        if win._result_region is None:
            pane.mosaic.show_op(_RAW_OP)
        if first_look:
            # A FOVs VIEW'S OPENING CAMERA IS ITS CURRENT FIELD, not its region. Framing it
            # before the mosaic landed would be undone by this very `reset_view`, and framing
            # it on EVERY load would fight the user's pan on every timepoint step -- which is
            # exactly why the whole branch is behind `first_look` already.
            if getattr(win, "_fov_mode", False) and win._fov_slider is not None:
                win._draw_fov_boxes()
                fov = win._fov_slider.fov
                if fov is not None:
                    win._on_fov_changed(win._fov_slider.index, int(fov))
                else:
                    pane.mosaic.model.reset_view()
            else:
                pane.mosaic.model.reset_view()
    except Exception:                            # noqa: BLE001 - view framing is cosmetic
        pass
    win._shown_region = str(region)
    win._apply_settings_once()
    if win.open_clock is not None:
        win.open_clock.finish()
    frame_done(win)


def frame_done(win) -> None:
    """Open the playback gate: this mosaic is on screen, the next frame may be requested."""
    if win._slider is not None:
        win._slider.frame_done()
    bar = getattr(win, "_time_point_bar", None)
    if bar is not None:
        bar.frame_done()
