"""Gallery View: the scope, the crop, the contrast, the thread, and the window.

Complements test_viewer.py's "not implemented" pin; this covers what Gallery View actually
builds: real FOV subsetting, contrast over covered pixels only, incremental cell delivery,
and that nothing decodes on the Qt thread.
"""
from __future__ import annotations

import os
import sys
import threading

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # headless Qt; must precede the PyQt import

import numpy as np                                                       # noqa: E402
import pytest                                                            # noqa: E402

pytest.importorskip("qtpy")
# Same guard as test_viewer.py, for the same segfault: with PySide already in the process
# (napari / pytest-qt autoload it), importing PyQt5 GUI widgets on top crashes.
if "PySide6" in sys.modules or "PySide2" in sys.modules:
    pytest.skip(
        "PySide already loaded (napari/pytest-qt) — Qt binding conflict; run with "
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 to run the PyQt5 GUI tests.",
        allow_module_level=True,
    )

from qtpy.QtWidgets import QApplication                                   # noqa: E402

from squidxplorer import _gallery as G                                        # noqa: E402
from squidxplorer.reader import open_reader                                   # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    """This module's QApplication, not pytest-qt's — the chunked suite disables plugin
    autoload, so pytest-qt's fixture of the same name does not exist there."""
    app = QApplication.instance() or QApplication([])
    app.setProperty("_squidxplorer_test", True)   # main() won't call exec_/exit under test
    return app


# --- helpers --------------------------------------------------------------------------------


class _RecordingReader:
    """Wraps a real reader and records which thread each read happened on (not a fake reader,
    so this still exercises real acquisition geometry)."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self._path = getattr(inner, "_path", None)
        self.read_threads: list = []
        self.reads = 0
        self._lock = threading.Lock()

    @property
    def metadata(self):
        return self._inner.metadata

    def read(self, region, fov, channel, z_level, time_point=0):
        with self._lock:
            self.read_threads.append(threading.get_ident())
            self.reads += 1
        return self._inner.read(region, fov, channel, z_level, time_point)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _meta(root):
    reader = open_reader(root)
    return reader, reader.metadata


def _drain(qapp, win, timeout_s: float = 60.0):
    """Spin the event loop until the gallery's worker finishes, then flush its budgeted
    repaints until settled."""
    import time

    deadline = time.time() + timeout_s
    while win._worker is not None and win._worker.isRunning() and time.time() < deadline:
        qapp.processEvents()
    for _ in range(10):
        qapp.processEvents()
    guard = 0
    while win._dirty and guard < 10_000:
        win._flush_repaints()
        guard += 1
    return win


# --- the scope ------------------------------------------------------------------------------


def test_the_whole_acquisition_is_a_scope_and_so_is_a_selection(squid_dataset):
    """One code path, two scopes — the subset requirement is the design, not a mode."""
    root, _arrays = squid_dataset
    _reader, meta = _meta(root)

    whole = G.GalleryScope.whole(meta)
    assert whole.regions == ("B2", "B3")
    assert whole.fovs_of("B2") == (0, 1)
    assert whole.from_selection is False
    assert whole.crops(meta) == (), "the whole acquisition cannot be a crop of itself"

    sel = G.GalleryScope.from_region_fovs(meta, [("B3", 1)])
    assert sel.regions == ("B3",)
    assert sel.fovs_of("B3") == (1,)
    assert sel.from_selection is True
    assert sel.crops(meta) == ("B3",), "1 of 2 FOVs is a crop and must be named as one"


def test_a_stale_selection_drops_the_wells_that_are_gone_rather_than_refusing(squid_dataset):
    """A selection outlives a re-ingest: ``run_plate`` drops an unknown region, so does this."""
    root, _arrays = squid_dataset
    _reader, meta = _meta(root)
    sel = G.GalleryScope.from_region_fovs(meta, [("B2", 0), ("Z99", 0), ("B3", 47)])
    assert sel.regions == ("B2",)
    assert sel.fovs_of("B2") == (0,)


def test_the_scope_is_in_plate_order_whatever_order_the_selection_arrives_in(squid_dataset):
    """Rows read down the plate; a marquee reports cells in drag order, not plate order."""
    root, _arrays = squid_dataset
    _reader, meta = _meta(root)
    sel = G.GalleryScope.from_region_fovs(meta, [("B3", 0), ("B2", 1), ("B3", 1), ("B2", 0)])
    assert sel.regions == ("B2", "B3")


def test_cells_are_queued_region_major_so_a_whole_row_lands_first(squid_dataset):
    """A user comparing two wells wants one whole well early, not every well's first channel."""
    root, _arrays = squid_dataset
    _reader, meta = _meta(root)
    cells = G.GalleryScope.whole(meta).cells()
    assert [r for r, _c in cells[:2]] == ["B2", "B2"]
    assert len(cells) == 2 * 2


def test_the_cell_cap_truncates_whole_regions_and_never_half_a_row():
    """Half a region's channels is a row with holes, and a hole reads as a failed read."""
    scope = G.GalleryScope(
        regions=tuple(f"A{i}" for i in range(10)),
        fovs=tuple((f"A{i}", (0,)) for i in range(10)),
        channels=("c1", "c2", "c3"),
    )
    capped, dropped = scope.capped(max_cells=10)
    assert dropped == 7
    assert capped.regions == ("A0", "A1", "A2")
    assert capped.cell_count == 9 <= 10
    assert all(len(capped.fovs_of(r)) == 1 for r in capped.regions)
    # Under the cap it is the same object's content, untouched.
    same, none_dropped = scope.capped(max_cells=1000)
    assert none_dropped == 0 and same.regions == scope.regions


def test_a_scope_names_itself_including_the_crop(squid_dataset):
    """The status line has to distinguish a marquee'd corner from the whole well."""
    root, _arrays = squid_dataset
    _reader, meta = _meta(root)
    line = G.GalleryScope.from_region_fovs(meta, [("B2", 0)]).describe(meta)
    assert "1 region(s)" in line and "1 FOV(s)" in line
    assert "selection" in line and "cropped to a FOV subset" in line
    assert "whole acquisition" in G.GalleryScope.whole(meta).describe(meta)


def test_an_unknown_projection_is_refused_at_construction():
    """A typo'd projection must not degrade to "some z" — it names the axis the cell is reduced on."""
    with pytest.raises(ValueError, match="projection must be one of"):
        G.GalleryScope(regions=("A1",), fovs=(("A1", (0,)),), channels=("c",), projection="max")


# --- the crop, on real pixels ------------------------------------------------------------------


def test_a_fov_subset_fuses_a_smaller_cell_than_the_whole_region(squid_dataset):
    """Same FOV mapping ``run_plate`` takes; the crop comes for free from
    ``_placement.fov_offsets_px``, which normalises whatever FOV set it is handed."""
    root, _arrays = squid_dataset
    reader, meta = _meta(root)
    channel = meta["channels"][0]
    channel = G.channel_field(channel, "name", channel)

    whole = G.fuse_gallery_cell(reader, meta, "B2", (0, 1), channel)
    one = G.fuse_gallery_cell(reader, meta, "B2", (0,), channel)

    assert whole.full_shape[1] > one.full_shape[1], (
        f"a 1-FOV scope fused to {one.full_shape}, the same width as the 2-FOV region "
        f"{whole.full_shape} — the subset was not honoured")
    assert one.full_shape[0] == whole.full_shape[0]      # same row: height is unchanged
    assert one.n_fovs == 1 and whole.n_fovs == 2


def test_a_cell_reports_the_decimation_it_was_fused_at(squid_dataset):
    """``step`` is how the cell relates to the acquisition, so the caption can say "1/23 of
    11462x9587"."""
    root, _arrays = squid_dataset
    reader, meta = _meta(root)
    ch = G.channel_field(meta["channels"][0], "name")
    cell = G.fuse_gallery_cell(reader, meta, "B2", (0, 1), ch, target_px=4)
    assert cell.step >= 2
    assert cell.shape[1] <= cell.full_shape[1]


def test_a_geometry_that_is_not_derivable_returns_none_rather_than_a_guess(squid_dataset):
    """The same "do not guess" signal ``fuse_region_mosaic`` and ``mosaic_bbox_um`` use."""
    root, _arrays = squid_dataset
    reader, meta = _meta(root)
    blind = dict(meta)
    blind["fov_positions_um"] = {}
    ch = G.channel_field(meta["channels"][0], "name")
    assert G.fuse_gallery_cell(reader, blind, "B2", (0, 1), ch) is None
    assert G.plan_cell(blind, "B2", (0, 1)) is None


def test_a_cell_whose_every_fov_is_unreadable_raises_instead_of_going_black(squid_dataset):
    """A black cell would read a failure as empty tissue; one bad FOV is a hole, all of them
    is not a picture."""
    root, _arrays = squid_dataset
    reader, meta = _meta(root)
    with pytest.raises(ValueError, match="not one of the"):
        G.fuse_gallery_cell(reader, meta, "B2", (0, 1), "no_such_channel")


def test_a_ragged_z_is_cropped_to_the_common_shape_rather_than_losing_the_whole_cell(
        squid_dataset, monkeypatch):
    """``fuse_region_pyramid`` raises on a ragged z, correctly. Here z is being reduced, so
    crop to the common shape instead of losing the whole cell to one bad plane."""
    root, _arrays = squid_dataset
    reader, meta = _meta(root)
    ch = G.channel_field(meta["channels"][0], "name")
    real_read = reader.read

    def ragged(region, fov, channel, z_level, time_point=0):
        plane = real_read(region, fov, channel, z_level, time_point)
        return plane[:3, :3] if int(z_level) == 1 else plane      # z=1 is one pixel short both ways

    monkeypatch.setattr(reader, "read", ragged)
    cell = G.fuse_gallery_cell(reader, meta, "B2", (0, 1), ch, projection="mip")
    assert cell is not None and cell.image.size
    assert not cell.unreadable, "a ragged plane was counted as an unreadable FOV"


def test_one_unreadable_fov_leaves_a_counted_hole_in_its_own_place(squid_dataset, monkeypatch):
    """Dropping the failed FOV instead would shift its neighbours and misregister the region,
    not merely leave it incomplete."""
    root, _arrays = squid_dataset
    reader, meta = _meta(root)
    ch = G.channel_field(meta["channels"][0], "name")
    whole = G.fuse_gallery_cell(reader, meta, "B2", (0, 1), ch)

    real_read = reader.read

    def flaky(region, fov, channel, z_level, time_point=0):
        if int(fov) == 1:
            raise OSError("simulated bad plane")
        return real_read(region, fov, channel, z_level, time_point)

    monkeypatch.setattr(reader, "read", flaky)
    holed = G.fuse_gallery_cell(reader, meta, "B2", (0, 1), ch)
    assert holed.unreadable == (1,)
    assert holed.has_holes
    assert holed.shape == whole.shape, "the canvas moved when a FOV failed"
    assert not holed.covered.all(), "a failed FOV must be uncovered, not silently painted"


# --- contrast --------------------------------------------------------------------------------


def test_the_window_is_taken_over_the_covered_pixels_only():
    """Black gaps between FOVs otherwise drag the low end to zero and the whole region washes out."""
    rng = np.random.default_rng(0)
    signal = (rng.normal(9000, 400, size=(200, 200))).astype(np.uint16)
    image = np.zeros((200, 400), dtype=np.uint16)
    image[:, :200] = signal
    covered = np.zeros((200, 400), dtype=bool)
    covered[:, :200] = True

    honest = G.cell_window(image, covered)
    naive = G.cell_window(image, None)
    assert honest is not None and naive is not None
    assert honest[0] > naive[0], (
        f"counting the black gap moved the low end from {honest[0]:.0f} to {naive[0]:.0f}; "
        "the covered mask is not being used")


def test_a_flat_channel_gets_no_window_at_all_rather_than_a_fabricated_one():
    """``auto_contrast`` refuses for a blank channel — a narrow window would render its noise
    at full intensity, i.e. as signal — and the gallery must carry the refusal, not undo it."""
    flat = np.full((64, 64), 700, dtype=np.uint16)
    assert G.cell_window(flat, np.ones_like(flat, dtype=bool)) is None
    assert G.cell_window(np.zeros((0, 0), dtype=np.uint16)) is None


def test_shared_windows_is_the_union_and_omits_channels_that_refused():
    """The default, diverging from gallery-view: per-cell contrast would make a dim well and
    a bright well look the same."""
    def cell(region, channel, window):
        return G.GalleryCell(region, channel, np.zeros((2, 2), np.uint16),
                             np.ones((2, 2), bool), window, 1.0, (2, 2), 1)

    got = G.shared_windows([
        cell("A1", "g", (100.0, 900.0)),
        cell("A2", "g", (50.0, 400.0)),
        cell("A1", "r", None),
    ])
    assert got == {"g": (50.0, 900.0)}
    assert "r" not in got, "a channel with no window must be absent, never fabricated"


# --- the cache -------------------------------------------------------------------------------


def test_a_gallery_cell_keys_into_the_shared_plane_cache_and_a_second_fuse_reads_nothing(
        squid_dataset):
    """Reuse the shared plane cache (``_budget.cache_budget()``) rather than a gallery-private one."""
    from squidxplorer._mosaic_source import plane_cache, source_token

    root, _arrays = squid_dataset
    inner = open_reader(root)
    reader = _RecordingReader(inner)
    meta = reader.metadata
    ch = G.channel_field(meta["channels"][0], "name")
    cache, token = plane_cache(), source_token(reader)

    first = G.fuse_gallery_cell(reader, meta, "B2", (0, 1), ch, cache=cache, token=token)
    cold = reader.reads
    assert cold > 0
    second = G.fuse_gallery_cell(reader, meta, "B2", (0, 1), ch, cache=cache, token=token)
    assert reader.reads == cold, "a warm gallery cell decoded planes again"
    np.testing.assert_array_equal(first.image, second.image)

    # A DIFFERENT FOV subset is a different picture and must not be served the cached one.
    subset = G.fuse_gallery_cell(reader, meta, "B2", (0,), ch, cache=cache, token=token)
    assert reader.reads > cold
    assert subset.full_shape != first.full_shape


def test_a_cell_with_a_hole_in_it_is_not_cached(squid_dataset, monkeypatch):
    """A degraded read is a read to retry, not a picture to keep: caching it would also drop
    the hole silently on a later hit, since the cache stores pixels only."""
    from squidxplorer._mosaic_source import plane_cache, source_token

    root, _arrays = squid_dataset
    inner = open_reader(root)
    reader = _RecordingReader(inner)
    meta = reader.metadata
    ch = G.channel_field(meta["channels"][0], "name")
    cache, token = plane_cache(), source_token(reader)
    real_read = inner.read

    def flaky(region, fov, channel, z_level, time_point=0):
        if int(fov) == 1:
            raise OSError("simulated bad plane")
        return real_read(region, fov, channel, z_level, time_point)

    monkeypatch.setattr(inner, "read", flaky)
    holed = G.fuse_gallery_cell(reader, meta, "B2", (0, 1), ch, cache=cache, token=token)
    assert holed.unreadable == (1,)
    after_first = reader.reads

    # Second fuse must go back to the reader rather than being served the degraded cell.
    again = G.fuse_gallery_cell(reader, meta, "B2", (0, 1), ch, cache=cache, token=token)
    assert reader.reads > after_first, "a cell with a hole in it was cached"
    assert again.unreadable == (1,), "the retry lost the hole count"


def test_a_reader_with_no_identity_runs_uncached_rather_than_risking_another_acquisitions_pixels(
        squid_dataset):
    """``source_token`` refuses a reader with no ``_path``; the gallery must degrade, not crash."""
    root, _arrays = squid_dataset
    reader, meta = _meta(root)
    ch = G.channel_field(meta["channels"][0], "name")
    cell = G.fuse_gallery_cell(reader, meta, "B2", (0, 1), ch, cache=None, token=None)
    assert cell is not None and cell.image.size


# --- the window ------------------------------------------------------------------------------


def test_the_gallery_never_reads_a_plane_on_the_qt_thread(qapp, squid_dataset):
    """``sample_plane`` costs a measured ~493 ms per region on the UI thread; a gallery of N
    regions would pay that N times, so the reader is instrumented rather than trusted by inspection."""
    from squidxplorer._gallery_window import GalleryWindow

    root, _arrays = squid_dataset
    reader = _RecordingReader(open_reader(root))
    meta = reader.metadata
    main = threading.get_ident()

    win = GalleryWindow(reader, meta, G.GalleryScope.whole(meta), title="t")
    try:
        _drain(qapp, win)
        assert reader.reads > 0, "the gallery read nothing at all — the assertion below is vacuous"
        on_ui = [t for t in reader.read_threads if t == main]
        assert not on_ui, (
            f"{len(on_ui)} of {reader.reads} plane reads happened on the Qt thread; the gallery "
            "must decode only in GalleryWorker")
    finally:
        win.close()
        qapp.processEvents()


def test_cells_are_emitted_one_at_a_time_as_they_are_fused(qapp, squid_dataset):
    """Driven against ``GalleryWorker`` directly, connected before ``start()`` — connecting
    after ``GalleryWindow.__init__`` starts it would race the first cell."""
    from squidxplorer._gallery_window import GalleryWorker

    root, _arrays = squid_dataset
    reader, meta = _meta(root)
    scope = G.GalleryScope.whole(meta)

    worker = GalleryWorker(reader, meta, scope)
    seen: list = []
    progress: list = []
    worker.cellReady.connect(lambda c: seen.append((c.region, c.channel)))
    worker.progress.connect(lambda d, t: progress.append((d, t)))
    worker.start()
    try:
        deadline = __import__("time").time() + 60
        while worker.isRunning() and __import__("time").time() < deadline:
            qapp.processEvents()
        for _ in range(10):
            qapp.processEvents()

        assert len(seen) == scope.cell_count, (
            f"{len(seen)} cellReady signals for {scope.cell_count} cells — cells are being "
            "batched instead of arriving as they are fused")
        assert len(set(seen)) == len(seen), "a cell was emitted twice"
        assert seen == scope.cells(), "cells did not arrive region-major, so a row completes late"
        # Progress is reported per cell too, not once at the end.
        assert len(progress) == scope.cell_count
        assert progress[-1] == (scope.cell_count, scope.cell_count)
    finally:
        worker.stop()
        worker.wait(5000)


def test_every_cell_of_the_scope_ends_up_painted(qapp, squid_dataset):
    """The window half of the same property: nothing is left as an empty grey square."""
    from squidxplorer._gallery_window import GalleryWindow

    root, _arrays = squid_dataset
    reader, meta = _meta(root)
    scope = G.GalleryScope.whole(meta)

    win = GalleryWindow(reader, meta, scope, title="t")
    try:
        _drain(qapp, win)
        painted = [k for k, lab in win._labels.items()
                   if lab.pixmap() is not None and not lab.pixmap().isNull()]
        assert len(painted) == scope.cell_count, (
            f"{len(painted)} of {scope.cell_count} cells carry a pixmap")
    finally:
        win.close()
        qapp.processEvents()


def test_the_grid_is_regions_down_and_channels_across(qapp, squid_dataset):
    """A channel must stay in ONE column across regions, or the two wells being compared are
    not side by side."""
    from squidxplorer._gallery_window import GalleryWindow

    root, _arrays = squid_dataset
    reader, meta = _meta(root)
    scope = G.GalleryScope.whole(meta)
    win = GalleryWindow(reader, meta, scope, title="t")
    try:
        _drain(qapp, win)
        positions = {}
        for (region, channel), label in win._labels.items():
            idx = win._grid.indexOf(label)
            r, c, _rs, _cs = win._grid.getItemPosition(idx)
            positions[(region, channel)] = (r, c)
        for channel in scope.channels:
            cols = {positions[(r, channel)][1] for r in scope.regions}
            assert len(cols) == 1, f"channel {channel} is not in one column: {cols}"
        for region in scope.regions:
            rows = {positions[(region, c)][0] for c in scope.channels}
            assert len(rows) == 1, f"region {region} is not on one row: {rows}"
    finally:
        win.close()
        qapp.processEvents()


def test_a_rescope_leaves_no_orphan_widgets_painting_over_the_new_grid(qapp, squid_dataset):
    """Removing a widget from a QGridLayout does not reparent it, and ``deleteLater`` only
    SCHEDULES — so a rescoped gallery drew the previous layout on top of the new one."""
    from squidxplorer._gallery_window import GalleryWindow

    root, _arrays = squid_dataset
    reader, meta = _meta(root)
    win = GalleryWindow(reader, meta, G.GalleryScope.whole(meta), title="t")
    try:
        _drain(qapp, win)
        live = set(win._labels.values()) | set(win._headers)

        win.rescope(G.GalleryScope.from_region_fovs(meta, [("B3", 0)]))
        _drain(qapp, win)

        wanted = set(win._labels.values()) | set(win._headers)
        strays = [c for c in win._grid_host.children()
                  if c.isWidgetType() and c.isVisible() and c not in wanted]
        assert not strays, (
            f"{len(strays)} widget(s) from the previous scope are still children of the grid host "
            f"and still visible: {[getattr(s, 'text', lambda: '?')() for s in strays[:4]]}")
        assert not (live & wanted), "rescope reused widgets it was supposed to rebuild"
        assert set(k[0] for k in win._labels) == {"B3"}, "the rescoped grid kept the old regions"
        assert len(win._labels) == len(win._scope.channels)
    finally:
        win.close()
        qapp.processEvents()


def test_a_column_header_never_overruns_its_column(qapp, squid_dataset):
    """The header is fixed to the cell width and labelled by excitation wavelength rather than
    the full channel name, which used to run several headers together into one line."""
    from squidxplorer._gallery_window import GalleryWindow

    root, _arrays = squid_dataset
    reader, meta = _meta(root)
    win = GalleryWindow(reader, meta, G.GalleryScope.whole(meta), title="t")
    try:
        assert win._headers, "no column headers were built"
        for head in win._headers:
            assert head.width() <= win._cell_px, (
                f"header {head.text()!r} is {head.width()} px wide in a {win._cell_px} px column")
            assert head.toolTip(), "the short header dropped the full channel name entirely"

        # The fixture's channels are 488 nm and 638 nm; headings are the wavelength, not the
        # 22-character filename channel.
        headings = {h.text() for h in win._headers}
        assert any(t.endswith(" nm") for t in headings), headings
        assert not any("Fluorescence" in t for t in headings), headings

        # A size change moves the columns; the headers must move with them or they overrun again.
        win._size.setCurrentIndex(2)              # Large
        qapp.processEvents()
        for head in win._headers:
            assert head.width() == win._cell_px == 320
    finally:
        win.close()
        qapp.processEvents()


def test_a_repaint_is_budgeted_so_a_big_gallery_stutters_no_more_than_a_small_one(
        qapp, squid_dataset):
    """Measured 159-190 ms of frozen Qt thread on a 1536-well plate: not decoding (that's on
    the worker) but repainting, since each widened shared window invalidates every cell of
    that channel. Work per tick is now bounded by a budget, not by scope size."""
    from squidxplorer import _gallery_window as GW

    root, _arrays = squid_dataset
    reader, meta = _meta(root)
    win = GW.GalleryWindow(reader, meta, G.GalleryScope.whole(meta), title="t")
    try:
        _drain(qapp, win)
        painted: list = []
        original = win._paint
        win._paint = lambda cell: (painted.append(cell), original(cell))[1]

        win._queue_repaint(win._cells.keys())
        assert len(win._dirty) == len(win._cells)

        win._flush_repaints()
        assert len(painted) <= GW.REPAINT_BUDGET, (
            f"one flush painted {len(painted)} cells, over the {GW.REPAINT_BUDGET} budget — "
            "the stall scales with the gallery again")

        # And it must not stop half-done: the timer re-arms while work remains.
        if win._dirty:
            assert win._repaint_timer.isActive(), "a partial flush left cells dirty and disarmed"
        guard = 0
        while win._dirty and guard < 1000:
            win._flush_repaints()
            guard += 1
        assert not win._dirty, "the budgeted flush never converged"
    finally:
        win._paint = original
        win.close()
        qapp.processEvents()


def test_compositing_at_display_resolution_never_starves_the_label(qapp, squid_dataset):
    """The invariant behind the 2.5x paint speed-up: the strided view must stay >= 1 source
    pixel per drawn pixel, or cells go visibly soft with nothing else to notice."""
    from squidxplorer._gallery_window import GalleryWindow

    root, _arrays = squid_dataset
    reader, meta = _meta(root)
    win = GalleryWindow(reader, meta, G.GalleryScope.whole(meta), title="t")
    try:
        _drain(qapp, win)
        assert win._cells, "nothing was fused, so the invariant below is vacuous"
        for size_index, expected_px in ((0, 80), (1, 160), (2, 320)):
            win._size.setCurrentIndex(size_index)
            qapp.processEvents()
            assert win._cell_px == expected_px
            for cell in win._cells.values():
                k = win._draw_stride(cell)
                assert k >= 1
                h, w = cell.shape
                if k > 1:
                    assert len(range(0, h, k)) >= win._cell_px, (
                        f"stride {k} leaves {len(range(0, h, k))} rows for a "
                        f"{win._cell_px} px label — the cell would be upscaled")
                    assert len(range(0, w, k)) >= win._cell_px
    finally:
        win.close()
        qapp.processEvents()


def test_a_size_change_re_renders_from_ram_and_reads_nothing(qapp, squid_dataset):
    """Display settings re-render the arrays already in RAM, never re-read."""
    from squidxplorer._gallery_window import GalleryWindow

    root, _arrays = squid_dataset
    reader = _RecordingReader(open_reader(root))
    meta = reader.metadata
    win = GalleryWindow(reader, meta, G.GalleryScope.whole(meta), title="t")
    try:
        _drain(qapp, win)
        after_build = reader.reads
        win._size.setCurrentIndex(2)              # Large
        win._contrast.setCurrentIndex(1)          # per cell
        qapp.processEvents()
        assert reader.reads == after_build, "a display change re-read the acquisition"
        assert win._cell_px == 320
        assert win._contrast_mode == "cell"
    finally:
        win.close()
        qapp.processEvents()


def test_a_subset_gallery_shows_the_crop_on_the_row_label(qapp, squid_dataset):
    """Only the label can tell a marquee'd corner apart from the whole well, so it carries the
    FOV count in scope against the region's own."""
    from squidxplorer._gallery_window import GalleryWindow

    root, _arrays = squid_dataset
    reader, meta = _meta(root)
    scope = G.GalleryScope.from_region_fovs(meta, [("B2", 0)])
    win = GalleryWindow(reader, meta, scope, title="t")
    try:
        assert win._region_label("B2") == "B2\n1/2 FOV"
        assert G.GalleryScope.whole(meta).cell_count > scope.cell_count
    finally:
        win.close()
        qapp.processEvents()


def test_more_than_one_timepoint_gets_a_control_and_changing_it_changes_the_pixels(
        qapp, multi_time_point_dataset):
    """The bar is hidden at n_t == 1 (a spin pinned to 0..0 implies an axis that doesn't exist)
    and shown with real range otherwise."""
    from squidxplorer._gallery_window import GalleryWindow

    root, _planes = multi_time_point_dataset
    reader, meta = _meta(root)
    assert meta["n_t"] == 3
    scope = G.GalleryScope.whole(meta)

    win = GalleryWindow(reader, meta, scope, title="t")
    try:
        assert win._t.isVisibleTo(win) and win._t.maximum() == 2
        _drain(qapp, win)
        first = {k: c.image.copy() for k, c in win._cells.items()}
        assert first, "nothing was fused at t=0"

        win._t.setValue(2)                        # -> restart() on the new timepoint
        _drain(qapp, win)
        assert win._scope.time_point == 2
        later = {k: c.image for k, c in win._cells.items()}
        assert set(later) == set(first)
        assert any(not np.array_equal(later[k], first[k]) for k in first), (
            "every cell is identical at t=0 and t=2 — the timepoint is not reaching the reader")
    finally:
        win.close()
        qapp.processEvents()


def test_a_single_timepoint_acquisition_shows_no_timepoint_control(qapp, squid_dataset):
    from squidxplorer._gallery_window import GalleryWindow

    root, _arrays = squid_dataset
    reader, meta = _meta(root)
    win = GalleryWindow(reader, meta, G.GalleryScope.whole(meta), title="t")
    try:
        assert not win._t.isVisibleTo(win)
    finally:
        win.close()
        qapp.processEvents()


def test_the_gallery_reports_its_own_first_paint(qapp, squid_dataset):
    """First paint (CONTEXT.md's term) is distinct from the total; the status line adds the total."""
    from squidxplorer._gallery_window import GalleryWindow

    root, _arrays = squid_dataset
    reader, meta = _meta(root)
    win = GalleryWindow(reader, meta, G.GalleryScope.whole(meta), title="t")
    try:
        assert win.first_paint_ms() is None or win.first_paint_ms() >= 0
        _drain(qapp, win)
        fp = win.first_paint_ms()
        assert fp is not None and fp >= 0
        assert "first paint" in win._status.text()
    finally:
        win.close()
        qapp.processEvents()


# --- the command ------------------------------------------------------------------------------


def test_the_view_menu_opens_a_gallery_on_the_selection_and_rescopes_on_a_second_click(
        qapp, squid_dataset):
    """Scope comes from ``selected_region_fovs()``, the same plumbing the marquee and
    shift-click already feed."""
    import squidxplorer._viewer as V
    from tests.conftest import shutdown_plate_window

    root, _arrays = squid_dataset
    win = V.PlateWindow(None)
    try:
        win.ingest(str(root))
        qapp.processEvents()

        # No selection -> the WHOLE acquisition.
        assert win.gallery_scope().regions == ("B2", "B3")
        assert win.gallery_scope().from_selection is False

        win._gallery_act.trigger()
        gallery = win._gallery
        assert gallery is not None and gallery.isVisible()
        assert gallery._scope.regions == ("B2", "B3")

        # A selection -> that subset, in the SAME window.
        win._selected_regions = ["B3"]
        scope = win.gallery_scope()
        assert scope.from_selection is True and scope.regions == ("B3",)
        win._gallery_act.trigger()
        assert win._gallery is gallery, "a second Gallery View click opened a second window"
        assert gallery._scope.regions == ("B3",)
        assert "selection" in win._readout.text()
    finally:
        shutdown_plate_window(qapp, win)


def test_closing_the_plate_closes_the_gallery_and_joins_its_thread(qapp, squid_dataset):
    """A gallery left open would draw into a closed plate's reader with a QThread still alive."""
    import squidxplorer._viewer as V
    from tests.conftest import shutdown_plate_window

    root, _arrays = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    qapp.processEvents()
    win._gallery_act.trigger()
    gallery = win._gallery
    assert gallery is not None
    shutdown_plate_window(qapp, win)
    assert win._gallery is None
    assert not gallery.isVisible()
    assert gallery._worker is None or not gallery._worker.isRunning()
