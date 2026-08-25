"""Launching the Gallery View and the native-3D popout FROM the plate window.

Extracted from ``PlateWindow`` (2026-08-14): neither is plate machinery — they are launchers
of other products that happen to be aimed by the plate's selection and cursor. Following the
``_ingest`` precedent, these are functions over the window's own state (one bookkeeping, no
second copy of it); ``PlateWindow`` keeps thin delegates because tests and the control wiring
reach ``gallery_scope`` / ``_open_gallery_view`` / ``_open_native_3d`` by name on the window.
"""

from __future__ import annotations

from squidxplorer._gallery import GalleryScope
from squidxplorer._logpane import get_logger

log = get_logger("viewer")


def gallery_scope(win):
    """The scope a gallery would open on RIGHT NOW: the plate selection, else the whole thing.

    The selection plumbing is the one that already exists — ``selected_region_fovs()``, which is
    fed by ``PlateOverview.selected_wells()`` through ``_on_selection_changed`` and by the
    shift-drag marquee. A gallery therefore inherits the marquee, Cmd/Ctrl-A, and shift-click
    refinement for free, and there is no second selection mechanism to keep in step. The pairs
    it returns are ``(region, fov)``, which is exactly the ``{region: [fov, ...]}`` mapping
    ``run_plate(regions=...)`` takes — so a cropped well stays cropped all the way through.

    Returns ``None`` (never an empty gallery) when no acquisition is open.
    """
    if win._meta is None:
        return None
    sel = win.selected_region_fovs()
    if sel:
        return GalleryScope.from_region_fovs(win._meta, sel, time_point=win.time_point)
    return GalleryScope.whole(win._meta, time_point=win.time_point)


def open_gallery_view(win) -> None:
    """Tile the selected Regions side by side, one row each, one column per channel.

    The port of hongquanli/gallery-view's Region view ("Add Region view: stitched per-region
    MIPs", #7), adapted rather than imported for the same reason ``_napari3d`` adapts its 3-D
    recipe: gallery-view pins napari <0.6 and we run 0.6.6. See :mod:`squidxplorer._gallery` for
    which of its decisions were taken and which two were diverged from.

    SUBSET-NATIVE, and that is the whole design rather than an option on it: the scope is
    :func:`gallery_scope`, i.e. the plate selection when there is one and the whole acquisition
    when there is not. One code path, two scopes.

    ONE gallery at a time. A second click RESCOPES and raises the open one instead of stacking
    a second window on the first, because "gallery of the current selection" is a question with
    one answer, and two galleries side by side is what the gallery itself is for.
    """
    scope = gallery_scope(win)
    if scope is None:
        win._readout.setText("Open an acquisition before opening the Gallery View.")
        return
    if scope.is_empty():
        win._readout.setText(
            "Gallery View: this acquisition has no regions with FOVs to tile.")
        return

    from squidxplorer._gallery_window import GalleryWindow

    title = win._acq_name or "acquisition"
    gallery = getattr(win, "_gallery", None)
    if gallery is not None and gallery.isVisible():
        gallery.rescope(scope, title=title)
    else:
        gallery = GalleryWindow(win._reader, win._meta, scope, title=title, parent=None)
        win._gallery = gallery
        gallery.resize(min(1400, 220 + 180 * max(1, len(scope.channels))), 900)
        gallery.show()
    gallery.raise_()
    gallery.activateWindow()
    msg = f"Gallery View: {scope.describe(win._meta)}"
    win._readout.setText(msg)
    win.log.info("%s", msg)


def open_native_3d(win) -> None:
    """Popout napari 3D on the current region's centre FOV at native resolution (gallery-view
    recipe). Fails to the LOG by name, never silently.

    It carries NO contrast or colormap. It used to harvest both off ``win._mosaic_pane``'s
    layers, which have never existed: the pane was pinned to None on 2026-07-23 and the harvest
    could only ever produce two empty dicts. ``open_native_3d`` defaults both to None and
    resolves the acquisition's own ``display_color`` and an autoscale, which is what this call
    has actually been doing all along. A window's on-screen LUTs reach 3D through
    ``RegionViewer``, which has the layers.
    """
    if win._reader is None or win._meta is None:
        win._readout.setText("No acquisition open - drop one before opening the 3D view.")
        return
    region = getattr(win, "_mosaic_region", None) or win._cursor.region
    if region is None:
        win._readout.setText("No region is open to render in 3D.")
        return
    try:
        from squidxplorer._napari3d import open_native_3d as _popout

        # HELD, not fire-and-forget. A later region can raise the dataset's contrast ceiling
        # (the 14-bit set reads 3437 at C3 and 16380 at E7), and a popout nobody kept a ref to
        # cannot be told -- it would sit on a slider that stops short of its own pixels.
        win._plate_native3d = _popout(win._reader, win._meta, region)
        log.info("opened native napari 3D popout for region %s", region)
    except Exception as exc:                     # noqa: BLE001 - NAMED, to the log and readout
        log.error("native 3D view failed for region %s: %s", region, exc)
        win._readout.setText(f"3D native view failed: {exc}")
