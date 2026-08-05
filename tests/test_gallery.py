"""Gallery View, for real: the scope, the crop, the contrast, the thread, and the window.

Replaces the half of `test_viewer.py::test_gallery_view_is_a_view_menu_command_and_not_an_operator`
that pinned the stub's "not implemented" status line. That test still owns "it is a View-menu
command and not an operator"; everything about what it BUILDS is here.

The four things worth pinning, and why each one is a test rather than a comment:

1. **The subset is real.** `stitch_plate(regions={region: [fov, ...]})` was already verified to
   crop; the gallery reuses the same mapping and the same `_placement` helpers, and
   `test_a_fov_subset_fuses_a_smaller_cell_than_the_whole_region` measures that it comes out
   smaller. A gallery that quietly fused the whole well would still look like a picture.
2. **Nothing decodes on the Qt thread.** `_contrast.py:157` costs a measured 493 ms per region
   when a caller materialises pixels on the UI thread to pick contrast limits, and a gallery is N
   of those. `test_the_gallery_never_reads_a_plane_on_the_qt_thread` instruments the reader with
   the thread ident of every `read` and fails on the main one, so the rule cannot be broken by a
   later refactor that "just" fuses one cell inline.
3. **Cells arrive one at a time.** The owner asked for exactly that ("populate each channel as soon
   as it is ready"), so it is behaviour, not an implementation detail.
4. **Contrast is over the covered pixels.** gallery-view's own lesson. A region with holes windows
   differently from the same region with its holes counted as black, and the difference is the
   whole cell washing out.
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

from squidmip import _gallery as G                                        # noqa: E402
from squidmip.reader import open_reader                                   # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    """This module's QApplication, by the same convention every other GUI test module here uses.

    NOT pytest-qt's fixture of the same name: `tools/run_suite_chunked.py` runs with
    `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` (it has to, or the PyQt5 tests silently skip on a PySide
    that napari dragged in), so pytest-qt is not loaded and its `qapp` does not exist. A test that
    only ever ran under a bare `pytest` invocation passes locally and errors at SETUP in the
    suite — which is exactly how this module first went red.
    """
    app = QApplication.instance() or QApplication([])
    app.setProperty("_squidmip_test", True)   # main() won't call exec_/exit under test
    return app


# --- helpers --------------------------------------------------------------------------------


class _RecordingReader:
    """A reader that forwards to a real one and records WHICH THREAD each read happened on.

    Deliberately a wrapper over the real reader rather than a fake: a fake would let the gallery
    pass this suite while failing on a real acquisition's geometry, which is the failure shape
    this repo keeps finding.
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self._path = getattr(inner, "_path", None)
        self.read_threads: list = []
        self.reads = 0
        self._lock = threading.Lock()

    @property
    def metadata(self):
        return self._inner.metadata

    def read(self, region, fov, channel, z, t=0):
        with self._lock:
            self.read_threads.append(threading.get_ident())
            self.reads += 1
        return self._inner.read(region, fov, channel, z, t)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _meta(root):
    reader = open_reader(root)
    return reader, reader.metadata


def _drain(qapp, win, timeout_s: float = 60.0):
    """Spin the event loop until the gallery's worker is finished, then flush its repaints."""
    import time

    deadline = time.time() + timeout_s
    while win._worker is not None and win._worker.isRunning() and time.time() < deadline:
        qapp.processEvents()
    for _ in range(10):
        qapp.processEvents()
    # Repaints are BUDGETED (see GalleryWindow._flush_repaints), so one flush is not the end of
    # them. Drain to a settled gallery, which is the state the assertions are about.
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
    """A selection outlives a re-ingest. ``stitch_plate`` drops an unknown region; so does this.

    Refusing the whole gallery over one stale well would be worse than showing the wells that are
    still there — and silently RENAMING it would be worse still, which is why the drop is total.
    """
    root, _arrays = squid_dataset
    _reader, meta = _meta(root)
    sel = G.GalleryScope.from_region_fovs(meta, [("B2", 0), ("Z99", 0), ("B3", 47)])
    assert sel.regions == ("B2",)
    assert sel.fovs_of("B2") == (0,)


def test_the_scope_is_in_plate_order_whatever_order_the_selection_arrives_in(squid_dataset):
    """Rows read down the plate. A marquee reports cells in drag order, which is not plate order."""
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
    """Half a region's channels is a row with holes, and a hole reads as a failed read.

    The cap is not gallery-view's — it has none — and it is here because this product opens
    1536-well plates, where a whole-acquisition gallery is 6144 cells.
    """
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
    """The status line has to distinguish a marquee'd corner from the whole well: both are pictures."""
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
    """THE subset requirement, measured. The FOV mapping is the same one ``stitch_plate`` takes.

    The two FOVs of the fixture sit side by side (+0.5 mm in x, same row), so one of them is half
    the mosaic. The crop comes for free from ``_placement.fov_offsets_px``, which normalises the
    top-left FOV of whatever set it is handed to (0, 0) — there is no cropping code to get wrong.
    """
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
    """``step`` is how the cell relates to the acquisition. Without it a cell is a picture of
    unknown scale, and the caption cannot say "1/23 of 11462x9587"."""
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
    """A black cell would report a read failure as empty tissue — the silent failure this
    codebase has six confirmed instances of. One bad FOV is a hole; all of them is not a picture."""
    root, _arrays = squid_dataset
    reader, meta = _meta(root)
    with pytest.raises(ValueError, match="not one of the"):
        G.fuse_gallery_cell(reader, meta, "B2", (0, 1), "no_such_channel")


def test_a_ragged_z_is_cropped_to_the_common_shape_rather_than_losing_the_whole_cell(
        squid_dataset, monkeypatch):
    """A MIP collapses the z axis away, so there is nothing left to misregister.

    ``fuse_region_pyramid`` RAISES on a ragged z, and is right to: a pyramid level whose z planes
    disagree misaligns the stack napari puts a slider on. Here z is being reduced, so the honest
    thing is the opposite — crop to the common shape and keep the FOVs that are fine, rather than
    lose the cell to one bad plane. Both rules are stated where they apply.
    """
    root, _arrays = squid_dataset
    reader, meta = _meta(root)
    ch = G.channel_field(meta["channels"][0], "name")
    real_read = reader.read

    def ragged(region, fov, channel, z, t=0):
        plane = real_read(region, fov, channel, z, t)
        return plane[:3, :3] if int(z) == 1 else plane      # z=1 is one pixel short both ways

    monkeypatch.setattr(reader, "read", ragged)
    cell = G.fuse_gallery_cell(reader, meta, "B2", (0, 1), ch, projection="mip")
    assert cell is not None and cell.image.size
    assert not cell.unreadable, "a ragged plane was counted as an unreadable FOV"


def test_one_unreadable_fov_leaves_a_counted_hole_in_its_own_place(squid_dataset, monkeypatch):
    """gallery-view builds the canvas from EVERY coordinate for exactly this reason: dropping the
    FOV instead would shift its neighbours and the region would be misregistered, not incomplete."""
    root, _arrays = squid_dataset
    reader, meta = _meta(root)
    ch = G.channel_field(meta["channels"][0], "name")
    whole = G.fuse_gallery_cell(reader, meta, "B2", (0, 1), ch)

    real_read = reader.read

    def flaky(region, fov, channel, z, t=0):
        if int(fov) == 1:
            raise OSError("simulated bad plane")
        return real_read(region, fov, channel, z, t)

    monkeypatch.setattr(reader, "read", flaky)
    holed = G.fuse_gallery_cell(reader, meta, "B2", (0, 1), ch)
    assert holed.unreadable == (1,)
    assert holed.has_holes
    assert holed.shape == whole.shape, "the canvas moved when a FOV failed"
    assert not holed.covered.all(), "a failed FOV must be uncovered, not silently painted"


# --- contrast --------------------------------------------------------------------------------


def test_the_window_is_taken_over_the_covered_pixels_only():
    """gallery-view's lesson, kept: black gaps between FOVs otherwise drag the low end to zero
    and the whole region washes out."""
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
    """``auto_contrast`` refuses for a blank channel because a narrow window renders its noise at
    full intensity, i.e. it reads as SIGNAL. The gallery must carry the refusal, not undo it."""
    flat = np.full((64, 64), 700, dtype=np.uint16)
    assert G.cell_window(flat, np.ones_like(flat, dtype=bool)) is None
    assert G.cell_window(np.zeros((0, 0), dtype=np.uint16)) is None


def test_shared_windows_is_the_union_and_omits_channels_that_refused():
    """The default, and the divergence from gallery-view: per-cell contrast makes a dim well and a
    bright well look the same, which is the one question a gallery exists to answer."""
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
    """Reuse, not a new cache: ``_budget.cache_budget()`` already decided how much this machine
    can spend on preview pixels, and a gallery-private cache would spend it twice."""
    from squidmip._mosaic_source import plane_cache, source_token

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
    """A degraded read is a read to RETRY, not a picture to keep.

    Two failures avoided by one rule. The transient that caused the hole (a disk hiccup, a file
    mid-write) would otherwise become this session's permanent picture of that region. And the
    cache stores pixels only, so a hit would rebuild the cell with ``unreadable=()`` — the hole
    still on screen, the caption no longer saying so, which is worse than the hole.
    """
    from squidmip._mosaic_source import plane_cache, source_token

    root, _arrays = squid_dataset
    inner = open_reader(root)
    reader = _RecordingReader(inner)
    meta = reader.metadata
    ch = G.channel_field(meta["channels"][0], "name")
    cache, token = plane_cache(), source_token(reader)
    real_read = inner.read

    def flaky(region, fov, channel, z, t=0):
        if int(fov) == 1:
            raise OSError("simulated bad plane")
        return real_read(region, fov, channel, z, t)

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
    """THE performance contract, and the reason it is a test.

    ``_contrast.py``'s ``sample_plane`` costs a measured 493 ms per region when a caller
    materialises pixels on the UI thread to pick contrast limits. A gallery of N regions would pay
    that N times, as N freezes. Instrumenting the reader is the only way to keep that true through
    a later refactor that "just" fuses one cell inline to fix a repaint.
    """
    from squidmip._gallery_window import GalleryWindow

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
    """The owner's words: "populate each channel as soon as it is ready".

    Driven against ``GalleryWorker`` DIRECTLY rather than by connecting to a window's worker after
    the fact. That is not stylistic: ``GalleryWindow.__init__`` starts its worker, so a test that
    connects afterwards races the first cell and fails intermittently — observed once here on a
    fast fixture before this was restructured. A flaky test is worse than no test, so the subject
    is the object whose contract this is, connected BEFORE ``start()``.
    """
    from squidmip._gallery_window import GalleryWorker

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
    from squidmip._gallery_window import GalleryWindow

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
    """gallery-view's table, not a reflowing grid: a channel has to stay in ONE column across
    regions or the two wells the user is comparing are not side by side."""
    from squidmip._gallery_window import GalleryWindow

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
    """Taking a widget out of a QGridLayout does not reparent it, and `deleteLater` only SCHEDULES.

    Between those two facts a rescoped gallery drew the previous layout on top of the new one:
    observed on the 10x tissue set as "manual0 / 28 FOV / manual1 / 28 FOV" superimposed in one row
    label, plus a leftover dark cell over the top-left corner. It looked like a paint bug and was
    a lifetime bug. `tests/test_no_orphan_windows.py` is the same class one container up: a widget
    that is no longer wanted must stop being a CHILD, not merely stop being in a layout.
    """
    from squidmip._gallery_window import GalleryWindow

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
    """Observed on the 10x tissue set: four "Fluorescence_405_nm_Ex"-length headers ran together
    into one illegible line and overprinted a row label.

    Two fixes, both pinned here. The header is FIXED to the cell width so it cannot overrun
    whatever it is called, and it is labelled by excitation wavelength — which is what actually
    distinguishes the columns, where "Fluorescence_" is what they all share. The full name stays
    reachable as the tooltip, so this narrows the label and not the datum.
    """
    from squidmip._gallery_window import GalleryWindow

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
    """MEASURED on the 1536-well plate before this was budgeted: 159-190 ms of frozen Qt thread.

    Not a decode — the reads were all on the worker, exactly as intended. It was the REPAINT: the
    shared window widens as cells land, each widening invalidates every cell of that channel, and
    64 cells x ~2-3 ms of composite + QImage + scale in one slot is the stall. The property that
    has to hold is that a gallery of 4 regions and a gallery of 64 feel the same, so the work per
    tick is bounded by a budget rather than by how many regions are in scope.
    """
    from squidmip import _gallery_window as GW

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


def test_a_size_change_re_renders_from_ram_and_reads_nothing(qapp, squid_dataset):
    """gallery-view's rule: display settings re-render the arrays already in RAM, never re-read."""
    from squidmip._gallery_window import GalleryWindow

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
    """Both a marquee'd corner and the whole well render as a picture. Only the label can tell
    them apart, so the label carries the FOV count in scope against the region's own."""
    from squidmip._gallery_window import GalleryWindow

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
    """The 5-D case. The bar is HIDDEN at n_t == 1 — a spin box pinned to 0..0 implies an axis the
    acquisition does not have — and shown, with real range, when there is a t axis to move on."""
    from squidmip._gallery_window import GalleryWindow

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
        assert win._scope.t == 2
        later = {k: c.image for k, c in win._cells.items()}
        assert set(later) == set(first)
        assert any(not np.array_equal(later[k], first[k]) for k in first), (
            "every cell is identical at t=0 and t=2 — the timepoint is not reaching the reader")
    finally:
        win.close()
        qapp.processEvents()


def test_a_single_timepoint_acquisition_shows_no_timepoint_control(qapp, squid_dataset):
    from squidmip._gallery_window import GalleryWindow

    root, _arrays = squid_dataset
    reader, meta = _meta(root)
    win = GalleryWindow(reader, meta, G.GalleryScope.whole(meta), title="t")
    try:
        assert not win._t.isVisibleTo(win)
    finally:
        win.close()
        qapp.processEvents()


def test_the_gallery_reports_its_own_first_paint(qapp, squid_dataset):
    """CONTEXT.md's **first paint**: asking for the window to its first cell on screen, taken where
    the drawing happens. Distinct from the total, and the total is what the status line adds."""
    from squidmip._gallery_window import GalleryWindow

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
    """The subset plumbing that already existed, used rather than duplicated: the scope comes from
    ``selected_region_fovs()``, which is what the marquee and shift-click already feed."""
    import squidmip._viewer as V
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
    """A gallery left open would draw into a closed plate's reader, and its QThread would be alive
    at teardown — which is the one thing ``PlateWindow.closeEvent`` exists to prevent."""
    import squidmip._viewer as V
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
