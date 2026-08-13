"""Shared test fixtures.

`squid_dataset` builds a tiny, real-shaped Squid individual-TIFF acquisition on disk
(2 regions x 2 fov x 2 z x 2 channels, 4x4 uint16 frames) with a legacy-schema
coordinates.csv and a pre-v1.0 (camera_settings-nested color) acquisition_channels.yaml,
plus the acquisition parameters.json scalars. Returns (root_path, {(region,fov,z,ch): array}).

`multi_time_point_dataset` is the same writer's layout with `Nt=3`: the ONLY multi-timepoint
acquisition anywhere in this suite. See the block comment above it for the layout and for why a
corpus that was 100% single-timepoint could not catch a whole class of bug.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import numpy as np
import pytest
import tifffile

#: The process's QApplication, held for the whole session so no fixture teardown can free it.
#:
#: PyQt gives the QApplication to PYTHON to own. Every GUI test module reaches it as
#: ``QApplication.instance() or QApplication([])`` inside a module-scoped fixture, which made a
#: PYTEST FIXTURE CACHE the only reference in the process. At the last teardown of such a module
#: pytest ran ``item.funcargs = None`` (``_pytest/runner.py:147``) and deleted the QApplication
#: while widgets, the shared QStyle and posted events still pointed at it, segfaulting the run and
#: taking the summary line with it, so a run that crashed and a run that passed looked identical.
#:
#: ``squidmip._viewer.qt_app()`` fixes this for anything that builds a PlateWindow, but modules
#: like ``tests/test_layer_tree.py`` never build one, so nothing pinned the app for them: measured
#: 2026-07-28, ``pytest tests/test_layer_tree.py`` returned 139 on 3 of 3 runs, and 0 on 3 of 3
#: with a reference held. Holding it here covers every GUI module at once, which is the right
#: layer because the ownership problem is created by the fixture cache, not by the library.
#:
#: Deliberately a module global rather than a session fixture: a fixture is still something pytest
#: releases, which is the exact hazard.
_QT_APP = None

#: The session's plate cell cache directory, created in ``pytest_configure`` and removed at the
#: end. See there for why the suite must never write into the real one.
_CACHE_DIR = None


def pytest_sessionfinish(session, exitstatus):
    """Tear Qt down while Qt is still alive, so the process does not abort at interpreter exit.

    A PlateWindow is never freed. Its widgets connect dozens of ``self``-capturing lambdas, and
    PyQt keeps each lambda alive in a slot proxy parented to the sender, so the closure's ``self``
    closes a window -> child -> proxy -> lambda -> window cycle whose links live in C++ where
    Python's cyclic collector cannot see them. ``gc.collect()`` frees nothing; measured, twelve
    build-and-close cycles leave 88 top-level widgets alive.

    Left alone they are destroyed during interpreter finalisation, against a Qt that is itself
    half torn down, and the process aborts with 134 or 139 AFTER pytest has printed a green
    summary. That is not cosmetic: the commit gate reads the exit status, so a perfectly green
    suite could not be committed, and "tests failed" was reported when none had.

    So we destroy them HERE, at a controlled point where the QApplication is still fully alive.
    Deleting the widgets is the real fix; the leak itself (the lambda cycles) is a production
    defect that outlives this hook and is recorded in TODOS.md.
    """
    if _CACHE_DIR:
        import shutil

        shutil.rmtree(_CACHE_DIR, ignore_errors=True)
    try:
        from qtpy.QtWidgets import QApplication
    except ImportError:
        return
    app = QApplication.instance()
    if app is None:
        return
    for w in list(app.topLevelWidgets()):
        try:
            w.close()
            w.deleteLater()
        except RuntimeError:
            pass            # already destroyed on the C++ side; nothing owed
    app.processEvents()     # run the deferred deletions while the app can still service them
    import gc
    gc.collect()


def pytest_configure(config):
    """Pin the QApplication before any test runs. No-op where PyQt5 is not installed.

    Also redirects the plate cell cache (``squidmip._platecache``) into a temporary directory for
    the whole session. Two reasons, and both are about the suite being honest rather than tidy:
    a test that opens a fixture acquisition would otherwise write real cells into the developer's
    ``user_cache_dir``, and a cell left there by one run would be read by the next, so a caching
    bug could pass on a warm machine and fail on a cold one. The directory is deliberately NOT
    cleaned up per test: within a session the cache is supposed to persist, and a test that wants
    a cold cache says so with its own ``root=``.
    """
    # THE CYCLIC COLLECTOR IS THE HAZARD, so it does not run mid-suite.
    #
    # A PlateWindow is never freed by refcount: its widgets connect self-capturing lambdas, PyQt
    # keeps each in a slot proxy parented to the sender, and the window -> child -> proxy -> lambda
    # -> window cycle lives in C++ where refcounting cannot see it. Only the cyclic collector frees
    # those, and it fires WHENEVER IT LIKES -- inside an unrelated allocation, with C++ frames on
    # the stack. That is not a leak symptom, it is an abort.
    #
    # Measured 2026-08-05: `test_loupe_follows_cursor_and_coalesces_to_newest` aborted inside
    # `Garbage-collecting` (while a yaml error object was being built) once 35 GUI tests preceded
    # it, killing the chunk and hiding ~50 tests behind an INCOMPLETE. The pair alone passes and 34
    # predecessors pass, so nothing is wrong with the loupe -- it is where the collector happened
    # to fire. Collecting MORE made it worse, not better: an autouse gc.collect() after every test
    # moved the abort earlier, from test 36 to test 23. That is the confirmation that the timing of
    # collection, not the amount of garbage, is what decides whether the run survives.
    #
    # So: no automatic collection. `pytest_sessionfinish` still collects explicitly, after it has
    # destroyed the top-level widgets, which is the one point where Qt is fully alive and no C++
    # frame is on the stack. Unreachable cycles simply accumulate until then, and
    # tools/run_suite_chunked.py already bounds that by running ~100 tests per process.
    #
    # This does NOT fix the cycles. They are a real product defect (TODOS.md) and they outlive this
    # hook; it stops the suite from crashing on someone else's schedule while they are fixed.
    gc.disable()

    global _QT_APP, _CACHE_DIR
    import sys
    import tempfile

    from squidmip import _platecache

    if not os.environ.get(_platecache.ENV_DIR):
        _CACHE_DIR = tempfile.mkdtemp(prefix="squidmip-test-cache-")
        os.environ[_platecache.ENV_DIR] = _CACHE_DIR
    # Do NOT force PyQt5 in. The GUI modules skip themselves when PySide is already loaded
    # (pytest-qt autoload pulls it in, and two Qt bindings in one process break GL rendering);
    # importing PyQt5 here first would defeat that guard and run those modules under a mixed
    # binding, which is a different bug, not a fix. Under PYTEST_DISABLE_PLUGIN_AUTOLOAD=1, the
    # way the commit gate runs, PySide is absent and this pins as intended.
    if any(m.startswith("PySide") for m in sys.modules):
        return
    try:
        from qtpy.QtWidgets import QApplication
    except ImportError:
        return                      # headless CI without the [gui] extra: nothing to pin
    _QT_APP = QApplication.instance() or QApplication([])
    _keep_test_windows_off_the_foreground()


def _keep_test_windows_off_the_foreground() -> None:
    """Let test windows OPEN without dragging the user to the Space they open on. macOS only.

    The windows are wanted -- this is not ``QT_QPA_PLATFORM=offscreen``, which would delete them.
    What is not wanted is the side effect: macOS follows an APPLICATION when it activates, and Qt
    activates the process the moment it shows a window. Over a full run that is dozens of
    activations, so anyone working on another Space gets yanked back at intervals. The Mission
    Control preference that governs it ("when switching to an application, switch to a Space with
    open windows") is global and worth keeping on for real apps, so the fix belongs on THIS
    process, not in System Settings.

    Two levers, both needed, because they cover different moments:

    ``NSApplicationActivationPolicyAccessory``
        Stops the pytest process from being a foreground app at all: no Dock tile, and it is
        never made frontmost on its own. Windows still render and still appear on the Space
        they are created on.
    ``WA_ShowWithoutActivating``
        Per window, set on every top-level widget just before it is shown. The activation policy
        governs the process; this governs the individual ``show()``, which is what would
        otherwise make the window key and pull focus.

    Best-effort by construction: every failure path leaves the old behaviour rather than breaking
    a test run over a display nicety. Set ``SQUIDMIP_TEST_FOREGROUND=1`` to opt back into windows
    that take focus (useful when driving a test by hand and wanting it frontmost).
    """
    import sys

    if sys.platform != "darwin" or os.environ.get("SQUIDMIP_TEST_FOREGROUND"):
        return

    try:                                        # process-level: never become the active app
        from AppKit import NSApp, NSApplicationActivationPolicyAccessory

        app = NSApp()
        if app is not None:
            app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    except Exception:
        pass                                    # no pyobjc, or a Qt that owns NSApp differently

    try:                                        # window-level: show without becoming key
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QWidget

        if getattr(QWidget, "_squidmip_no_activate", False):
            return                              # already patched (pytest_configure ran twice)
        original_set_visible = QWidget.setVisible

        def set_visible(self, visible):
            # Every show() and showNormal() funnels through setVisible, so patching the one
            # entry point covers napari's windows too -- and those are built deep inside napari,
            # where there is no seam to pass a flag through.
            if visible and self.isWindow():
                self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
            return original_set_visible(self, visible)

        QWidget.setVisible = set_visible
        QWidget._squidmip_no_activate = True
    except Exception:
        pass


# --- process-global operator registries: snapshot + restore around every test ----------------
#
# ``add_projector`` / ``add_segmenter`` / ``add_region_operator`` write into module-level dicts that
# live for the whole pytest process. A test that registers one and does not remove it leaks into
# every test that runs after it, and the failure lands somewhere else entirely:
# ``tests/test_decon.py::test_decon_op_factory_produces_a_plane_op_and_is_registrable`` registered
# ``decon_test_factory`` with no teardown, which made
# ``tests/test_operator_integration.py::test_available_projectors_exact_list`` pass on its own and
# fail in the full suite -- an order-dependent failure that reads as a bug in the second file.
#
# AUTOUSE is safe here because this only SAVES and RESTORES: it registers nothing, removes nothing
# during a test, and every registration in this suite happens inside a test function (nothing is
# registered at module import or in a session/module fixture, which this would otherwise undo).
# Being autouse is the point -- it covers registry-mutating tests nobody has written yet.
# TWO, not four: `add_projector` and `add_region_operator` both write `_engine._OPERATORS`, and the
# `_stitch._REGION_OPERATORS` / `_stitch._REGION_REQUIRES` pair they used to write instead is gone.
_REGISTRIES = (
    ("squidmip._engine", "_OPERATORS"),
    ("squidmip._spots", "_SEGMENTERS"),
)


@pytest.fixture(autouse=True)
def _cold_plate_cell_cache(tmp_path, monkeypatch):
    """Every test starts with a COLD plate cell cache (``squidmip._platecache``).

    Two failures this prevents, and the second is the one that bites. A test that opens a fixture
    acquisition would otherwise write real cells into the developer's ``user_cache_dir``; and,
    worse, ``real_dataset`` and ``sim_1536wp`` live at FIXED paths, so a cell written by one test
    would be read by every later test that opens the same acquisition. The second test would then
    see one replayed tile per well where the producer emits one per FOV -- an order-dependent
    failure landing in a file that did nothing wrong, which is exactly the shape of the registry
    leak documented above.

    A test that wants a WARM cache builds one explicitly with ``root=``; that is what
    ``tests/test_platecache.py`` does, and it is the honest way to test persistence.
    """
    from squidmip import _platecache

    monkeypatch.setenv(_platecache.ENV_DIR, str(tmp_path / "plate-cells"))
    _platecache.clear_memory_tier()
    yield
    _platecache.clear_memory_tier()


@pytest.fixture(autouse=True)
def _cold_result_cache():
    """Every test starts and ends with an EMPTY ``squidmip._recipe.RESULTS``.

    ``RESULTS`` is process-wide by design -- that is what makes a result computed in one window
    available to a window opened later -- and its key is ``(region scope, acquisition, chain)``.
    Every fixture in this file uses the SAME region ids (``B2``, ``B3``, ``A1``), so an entry left
    behind by one test is a candidate replay for the next one that opens a window on that region.
    The acquisition version keeps two DIFFERENT acquisitions apart, but ``squid_dataset`` is a
    tmp_path fixture and two tests that happen to get the same tmp_path prefix are not worth
    reasoning about: the same argument as ``_cold_plate_cell_cache`` above, and the same fix.
    """
    from squidmip import _recipe

    _recipe.RESULTS.clear()
    yield
    _recipe.RESULTS.clear()


@pytest.fixture(autouse=True)
def _cold_dataset_depth():
    """Every test starts on an UNMEASURED contrast ceiling (``squidmip._bitdepth``).

    The ceiling is process-wide for the same reason ``_LUT_CLIPBOARD`` is -- one app, one open
    acquisition -- and in production ``ViewerManager.set_dataset`` is what clears it. A test that
    fuses a 12-bit fixture leaves the module saying 4095, and the next test to assert a contrast
    window of 9000 (``test_view_settings``) would then see it clamped by a dataset it never
    loaded. Cleared on the way out too, so the failing test is the one that measured, not the one
    after it.
    """
    from squidmip import _bitdepth

    _bitdepth.new_dataset(None)
    yield
    _bitdepth.new_dataset(None)


@pytest.fixture(autouse=True)
def _restore_operator_registries():
    import importlib

    saved = []
    for module_name, attr in _REGISTRIES:
        try:
            registry = getattr(importlib.import_module(module_name), attr)
        except (ImportError, AttributeError):
            # NAMED, not silent: a renamed registry must be visible, not quietly unguarded.
            raise AssertionError(
                f"{module_name}.{attr} no longer exists; tests/conftest.py restores it between "
                "tests and must be updated to the new name, or a leaked registration will again "
                "fail a different test file than the one that caused it."
            ) from None
        saved.append((registry, dict(registry)))
    yield
    for registry, snapshot in saved:
        if registry != snapshot:
            registry.clear()
            registry.update(snapshot)


REGIONS = ["B2", "B3"]
FOVS = [0, 1]
NZ = 2
# One channel present in the YAML (color via nested camera_settings), one ABSENT from the
# YAML (exercises the CHANNEL_COLORS_MAP wavelength fallback). Both contain '_' and '-'.
CH_IN_YAML = "Fluorescence_638_nm_-_Penta"
CH_NOT_IN_YAML = "Fluorescence_561_nm_-_Penta"
CHANNELS = [CH_IN_YAML, CH_NOT_IN_YAML]

_YAML = """\
version: 1
objective: 20x
channels:
- name: Fluorescence 638 nm - Penta
  camera_settings:
    '1':
      display_color: '#FF0000'
      exposure_time_ms: 50.0
"""

# Legacy flat sidecar (fallback source). Note magnification/sensor -> recomputed px 0.188,
# deliberately DIFFERENT from acquisition.yaml's stored 0.325 so tests prove which is used.
#
# `objective.NA` is here because every real acquisition writes it and NOTHING ELSE DOES: Squid's
# acquisition.yaml objective block is name/magnification/pixel_size_um/camera_binning, with no
# aperture (see `_acquisition.load_objective_na`). 0.8 is the real 20x's own value, transcribed
# from ~/Downloads/20x_scan_2025-09-05_17-57-50, which is the objective this fixture claims.
# Without it the fixture is not a faithful Squid acquisition and decon must refuse on it.
_PARAMS = {
    "Nz": NZ,
    "Nt": 1,
    "dz(um)": 1.5,
    "objective": {"magnification": 20.0, "NA": 0.8},
    "sensor_pixel_size_um": 3.76,
}

# Authoritative rich metadata. pixel_size_um is stored (binning-aware), not recomputed.
_ACQ_YAML = """\
objective:
  pixel_size_um: 0.325
  magnification: 20.0
  sensor_pixel_size_um: 3.76
sample:
  wellplate_format: 1536 well plate
z_stack:
  nz: 2
  delta_z_mm: 0.0015
time_series:
  nt: 1
"""


def _pixel_value(r_i, fov, z, c_i):
    # deterministic, unique per plane so exact-read comparisons are meaningful
    return r_i * 1000 + fov * 100 + z * 10 + c_i


def _write_timepoint(folder: Path, arrays: dict, tag: int = 0):
    folder.mkdir(parents=True, exist_ok=True)
    for r_i, region in enumerate(REGIONS):
        for fov in FOVS:
            for z in range(NZ):
                for c_i, ch in enumerate(CHANNELS):
                    base = _pixel_value(r_i, fov, z, c_i) + tag * 5000
                    arr = (np.arange(16, dtype=np.uint16).reshape(4, 4) + base).astype(np.uint16)
                    tifffile.imwrite(folder / f"{region}_{fov}_{z}_{ch}.tiff", arr)
                    arrays[(region, fov, z, ch)] = arr


# Real Squid coordinates.csv schema (verified against synthetic_2x2_wellplate): region + x/y in
# mm, a z column that is present but EMPTY, and NO fov column. FOV identity is row order within
# a region. Rows are repeated once per z-level here (NZ=2) because that is what a multi-z
# acquisition writes — the reader must de-duplicate on (region, x, y) before counting, or every
# real z-stack would trip the row-count cross-check.
_FOV_MM = {0: (10.0, 20.0), 1: (10.5, 20.0)}   # fov 1 is +0.5 mm in x => same row, next column


def _coordinates_csv() -> str:
    lines = ["region,x (mm),y (mm),z (mm)"]
    for region in REGIONS:
        for _z in range(NZ):                    # one row per z-level, same stage position
            for fov in FOVS:
                x, y = _FOV_MM[fov]
                lines.append(f"{region},{x},{y},")
    return "\n".join(lines) + "\n"


@pytest.fixture
def squid_dataset(tmp_path):
    root = tmp_path / "acq"
    arrays: dict = {}
    _write_timepoint(root / "0", arrays, tag=0)
    (root / "acquisition_channels.yaml").write_text(_YAML)
    (root / "acquisition.yaml").write_text(_ACQ_YAML)
    (root / "acquisition parameters.json").write_text(json.dumps(_PARAMS))
    (root / "coordinates.csv").write_text(_coordinates_csv())
    return root, arrays


@pytest.fixture
def pyramid_dataset(tmp_path):
    """A Squid acquisition whose fields are big enough to produce a REAL pyramid.

    `squid_dataset` uses 4x4 frames, and `_output._PYRAMID_MIN_YX` (256) collapses anything
    that small to level 0 alone — so it cannot exercise pyramid level selection at all. This
    one writes 640px fields (two levels: 640 -> 320) and stays deliberately minimal otherwise
    (1 well, 1 fov, 1 channel, 2 z) so the extra pixels don't cost real time.

    Returns (root, region, frame_size).
    """
    root = tmp_path / "acq_pyr"
    size, region, ch = 640, "B2", CH_IN_YAML
    folder = root / "0"
    folder.mkdir(parents=True, exist_ok=True)
    for z in range(2):
        # A gradient plus a per-z offset: downsampled levels stay distinguishable, and a crop
        # from a known position has a predictable value (so a loupe read can be checked).
        yy, xx = np.mgrid[0:size, 0:size]
        arr = ((yy + xx).astype(np.uint16) + z * 7)
        tifffile.imwrite(folder / f"{region}_0_{z}_{ch}.tiff", arr)
    (root / "acquisition_channels.yaml").write_text(_YAML)
    (root / "acquisition.yaml").write_text(_ACQ_YAML)
    (root / "acquisition parameters.json").write_text(json.dumps(_PARAMS))
    return root, region, size


# --- the multi-timepoint acquisition -----------------------------------------------------------
#
# WHY IT EXISTS. Every other fixture in this file is Nt=1, and so is every real acquisition on
# this machine: the 10x laser-AF tissue set, the 20x scan and sim_1536wp all record ``Nt: 1`` in
# ``acquisition parameters.json`` and ``nt: 1`` in ``acquisition.yaml``. The entire test corpus is
# single-timepoint, so the time axis has never been exercised by anything, which is precisely why
# nothing catches the bug that ``tests/test_time_point.py`` documents.
#
# LAYOUT FIDELITY, MINIMAL CONTENT. The shape below is Squid's writer, read from its source and
# cross-checked against a real acquisition on disk; the content is as small as it can be and still
# mean something (1 region, 1 fov, 2 z, 2 channels, 4x4 uint16, 3 timepoints).
#
#   {root}/
#     acquisition.yaml                 time_series.nt        (multi_point_controller.py:882 ->
#                                                             _save_acquisition_yaml, :93)
#     acquisition parameters.json      Nt, dt(s)
#     acquisition_channels.yaml
#     coordinates.csv                  PLANNED positions, written BEFORE the run, columns
#                                      region,x (mm),y (mm),z (mm)
#                                      (multi_point_controller.py:735-744)
#     0/ 1/ 2/                         ONE FOLDER PER TIMEPOINT, named
#                                      f"{time_point:0{FILE_ID_PADDING}}"
#                                      (multi_point_worker.py:744). FILE_ID_PADDING is 0 in
#                                      control/_def.py:720, so the names carry NO padding at all.
#       {region}_{fov}_{z_level}_{channel}.tiff   (multi_point_worker.py:1108)
#       coordinates.csv                EXECUTED positions, one row per (fov, z_level), written
#                                      when the timepoint finishes, columns
#                                      region,fov,z_level,x (mm),y (mm),z (um),time
#                                      (multi_point_worker.py:802-805 build it, :757 writes it)
#       .done                          empty marker (multi_point_worker.py:785 ->
#                                      control/utils.py:193)
#
# THE TRAP, and the reason the two files are written separately here. There are TWO files named
# coordinates.csv with DIFFERENT columns and different meanings: the root one is the PLAN (where
# the scope intended to go), each timepoint one is the RECORD (where the stage actually was, with
# a wall-clock stamp per plane). ``reader.py`` calls them "schema (a)" and "schema (b)" and reads
# only the ROOT file, which hides that they are not two dialects of one thing. A fixture that
# wrote one and called it both would encode that confusion into the test corpus.
#
# DISTINGUISHABLE FRAMES. Every plane is a CONSTANT fill, ``time_point * 100 + z * 10 + channel``,
# so a test can say which timepoint it is holding by reading one pixel. A fixture whose timepoints
# looked alike would pass while the bug was still live, which is the exact failure mode being
# fixed here.
TIME_SERIES_REGION = "A1"
TIME_SERIES_FOV = 0
TIME_SERIES_NZ = 2
N_TIME_POINTS = 3

#: The same two channels as `squid_dataset`, in the order the READER resolves them
#: (``reader.metadata`` builds its channel list from ``sorted(channels)``). The fixture's pixel
#: values encode a channel INDEX, so they have to be indexed in the reader's order or every
#: assertion silently compares the wrong channel.
TIME_SERIES_CHANNELS = sorted(CHANNELS)

_TIME_SERIES_XY_MM = (12.0, 24.0)

#: Executed positions: Squid's per-timepoint schema, with the fov column and a wall-clock stamp.
_EXECUTED_COORDS_HEADER = "region,fov,z_level,x (mm),y (mm),z (um),time"
#: Planned positions: Squid's root schema, no fov column, no time.
_PLANNED_COORDS_HEADER = "region,x (mm),y (mm),z (mm)"

_TIME_SERIES_PARAMS = {
    "Nz": TIME_SERIES_NZ,
    "Nt": N_TIME_POINTS,
    "dt(s)": 60.0,
    "dz(um)": 1.5,
    "objective": {"magnification": 20.0},
    "sensor_pixel_size_um": 3.76,
}

_TIME_SERIES_ACQ_YAML = f"""\
objective:
  pixel_size_um: 0.325
  magnification: 20.0
  sensor_pixel_size_um: 3.76
sample:
  wellplate_format: 1536 well plate
z_stack:
  nz: {TIME_SERIES_NZ}
  delta_z_mm: 0.0015
time_series:
  nt: {N_TIME_POINTS}
  delta_t_s: 60.0
"""


def time_series_pixel_value(time_point: int, z_level: int, channel_index: int) -> int:
    """The constant every plane of the multi-timepoint fixture is filled with.

    Carries the timepoint in its hundreds digit on purpose: a consumer that silently serves t=0
    for every t produces values that differ from the truth by a visible multiple of 100.
    """
    return time_point * 100 + z_level * 10 + channel_index


def _executed_coordinates_csv() -> str:
    """One timepoint's coordinates.csv: what the stage DID, one row per (fov, z_level)."""
    x_mm, y_mm = _TIME_SERIES_XY_MM
    lines = [_EXECUTED_COORDS_HEADER]
    for z_level in range(TIME_SERIES_NZ):
        lines.append(
            f"{TIME_SERIES_REGION},{TIME_SERIES_FOV},{z_level},{x_mm},{y_mm},"
            f"{3930.0 + z_level * 1.5},2026-07-29_10-00-0{z_level}.000000"
        )
    return "\n".join(lines) + "\n"


def _planned_coordinates_csv() -> str:
    """The root coordinates.csv: what the scan PLANNED, one row per FOV, no z repeats."""
    x_mm, y_mm = _TIME_SERIES_XY_MM
    return f"{_PLANNED_COORDS_HEADER}\n{TIME_SERIES_REGION},{x_mm},{y_mm},\n"


@pytest.fixture
def multi_time_point_dataset(tmp_path):
    """A Squid individual-TIFF acquisition with ``Nt=3``, in Squid's own on-disk layout.

    Returns ``(root, planes)`` where ``planes`` is ``{(time_point, z_level, channel): array}``:
    region and fov are singletons, so keying on them would only add noise to every assertion.
    """
    root = tmp_path / "acq_time_series"
    planes: dict = {}
    for time_point in range(N_TIME_POINTS):
        # FILE_ID_PADDING is 0, so this is str(time_point) with nothing added. Written the long
        # way so the fixture still tracks Squid if a deployment ever raises the padding.
        folder = root / f"{time_point:0{0}}"
        folder.mkdir(parents=True, exist_ok=True)
        for z_level in range(TIME_SERIES_NZ):
            for channel_index, channel in enumerate(TIME_SERIES_CHANNELS):
                value = time_series_pixel_value(time_point, z_level, channel_index)
                arr = np.full((4, 4), value, dtype=np.uint16)
                name = f"{TIME_SERIES_REGION}_{TIME_SERIES_FOV}_{z_level}_{channel}.tiff"
                tifffile.imwrite(folder / name, arr)
                planes[(time_point, z_level, channel)] = arr
        (folder / "coordinates.csv").write_text(_executed_coordinates_csv())
        (folder / ".done").write_text("")
    (root / "acquisition_channels.yaml").write_text(_YAML)
    (root / "acquisition.yaml").write_text(_TIME_SERIES_ACQ_YAML)
    (root / "acquisition parameters.json").write_text(json.dumps(_TIME_SERIES_PARAMS))
    (root / "coordinates.csv").write_text(_planned_coordinates_csv())
    return root, planes


# --- IMA-254: one fixture per Squid output writer --------------------------------------------
#
# The builders live in tests/writer_fixtures.py and are imported INSIDE each fixture, not at
# module scope: writer_fixtures imports this module's shape constants, so a top-level import here
# would be circular. Deferring it also keeps collection cheap for the many tests that need none
# of them.
#
# Every one of these produces the SAME logical acquisition (2 regions x 2 FOVs x 2 z x 2 channels
# of 4x4 uint16) through a DIFFERENT writer, which is what lets the coverage suite assert
# identical metadata and identical pixels across all six with no per-writer special-casing.

def _writer_fixture(tmp_path, builder, name):
    from tests import writer_fixtures

    root = builder(tmp_path / name)
    return root, writer_fixtures.expected_arrays()


@pytest.fixture
def multipage_dataset(tmp_path):
    """MULTI_PAGE_TIFF: ``0/{region}_{fov:04}_stack.tiff``, positions inline, no coordinates.csv."""
    from tests import writer_fixtures

    return _writer_fixture(tmp_path, writer_fixtures.build_multi_page_tiff, "acq_multipage")


@pytest.fixture
def ome_tiff_dataset(tmp_path):
    """SaveOMETiffJob: ``ome_tiff/{region}_{fov:04}.ome.tiff``, 5-D TZCYX."""
    from tests import writer_fixtures

    return _writer_fixture(tmp_path, writer_fixtures.build_ome_tiff, "acq_ome")


@pytest.fixture
def zarr_hcs_dataset(tmp_path):
    """SaveZarrJob HCS: ``plate.ome.zarr/{row}/{col}/{fov}/0``, 5-D TCZYX."""
    from tests import writer_fixtures

    return _writer_fixture(tmp_path, writer_fixtures.build_zarr_hcs, "acq_zarr_hcs")


@pytest.fixture
def zarr_per_fov_dataset(tmp_path):
    """SaveZarrJob non-HCS default: ``zarr/{region}/fov_{n}.ome.zarr/0``, 5-D TCZYX."""
    from tests import writer_fixtures

    return _writer_fixture(tmp_path, writer_fixtures.build_zarr_per_fov, "acq_zarr_fov")


@pytest.fixture
def zarr_6d_dataset(tmp_path):
    """SaveZarrJob non-HCS 6D: ``zarr/{region}/acquisition.zarr``, 6-D FTCZYX (non-standard)."""
    from tests import writer_fixtures

    return _writer_fixture(tmp_path, writer_fixtures.build_zarr_6d, "acq_zarr_6d")


@pytest.fixture
def real_dataset():
    """The real 10x laser-AF tissue acquisition; else skip (used by integration tests).

    Repointed from the old `hongquan` z-stack, which was deleted. This is the acquisition the
    product is actually demoed on, and it is the harder case: a GLASS SLIDE with freeform regions
    (manual0 27 FOVs / manual1 28), Nz=10, 4 channels, 0.752 um/px, OME-TIFF on disk. Real pixels,
    real overlap (~209 px, ~10%), real per-channel focus disagreement.
    """
    path = Path("/Users/julioamaragall/Downloads/"
                "test_10x_laser_af_z_stack_2025-10-28_13-40-43.939945 yy")
    if not path.is_dir():
        pytest.skip(f"real tissue acquisition not present at {path}")
    return path


# --- sim_1536wp: the plate-scale fixture, and a guard that can tell HOLLOW from ABSENT ---------

SIM_1536WP = Path("/Users/julioamaragall/CEPHLA/Data/sim_1536wp")

#: Every plane of sim_1536wp is a symlink into this folder. Nothing in this repo generates it.
#: It USED to be ``~/Downloads/synthetic_2x2_wellplate``; that folder was deleted and every link
#: went hollow. Both 1536 fixtures on this machine (this one and ``~/Downloads/
#: synthetic_1536_wellplate``) were recomposed onto the real 20x scan below, channel for channel,
#: so a link ending ``_Fluorescence_561_nm_Ex.tiff`` still resolves to 561 nm pixels and the LUTs
#: and contrast stay meaningful. The scan is REAL DATA and is read-only.
SIM_1536WP_SOURCE = Path("/Users/julioamaragall/Downloads/20x_scan_2025-09-05_17-57-50")


def sim_1536wp_problem():
    """Why ``sim_1536wp`` is unusable, as a sentence naming the fix, or None when it is fine.

    The old guard was ``if not SIM_1536WP.is_dir(): skip``, which only asks whether the FOLDER is
    there. sim_1536wp's ``0/`` holds 6144 SYMLINKS into ``~/Downloads/synthetic_2x2_wellplate``,
    and that folder was deleted, so the directory check passed, every link dangled, and
    ``open_reader`` correctly refused with "contains no {region}_{fov}_{z}_{channel}.tiff": nine
    integration tests failing on a MISSING DATASET while reading as a reader bug. A fixture that is
    present but hollow has to be diagnosed as such, here, once, rather than in nine tracebacks.
    """
    if not SIM_1536WP.is_dir():
        return f"sim_1536wp not present at {SIM_1536WP}"
    planes = SIM_1536WP / "0"
    if not planes.is_dir():
        return f"sim_1536wp is present at {SIM_1536WP} but has no timepoint folder {planes}"
    entries = sorted(planes.glob("*.tif*"))[:8]        # cheap: a hollow fixture is hollow throughout
    if not entries:
        return f"sim_1536wp is present at {SIM_1536WP} but {planes} holds no TIFF entries"
    # `exists()` FOLLOWS symlinks, so a dangling link reads as absent, which is the point.
    dead = [p for p in entries if not p.exists()]
    if dead:
        target = None
        for p in dead:
            if p.is_symlink():
                target = Path(os.readlink(p))
                break
        return (
            f"sim_1536wp at {SIM_1536WP} is HOLLOW: its planes are symlinks whose target is gone "
            f"({target if target is not None else 'unreadable'}; e.g. {dead[0].name}). "
            f"To restore it, put the source acquisition back at {SIM_1536WP_SOURCE}: every plane "
            "here is a symlink onto that scan's planes, so restoring the source revives every "
            "link with no regeneration step. If the links point somewhere else entirely, repoint "
            "them at that scan MATCHING THE CHANNEL IN EACH LINK'S OWN NAME, or the plate opens "
            "with 405 nm pixels under a 561 nm LUT. This repo contains no generator for "
            f"{SIM_1536WP_SOURCE.name}."
        )
    return None


@pytest.fixture
def sim_1536wp():
    """The 1536-well plate-scale acquisition, or a skip that says exactly what is wrong.

    THE ONLY skip this suite is allowed to add: the data is genuinely not on this machine, and no
    code change can conjure it. It is deliberately loud about which of "absent" and "hollow" it is.
    """
    problem = sim_1536wp_problem()
    if problem:
        pytest.skip(problem)
    return SIM_1536WP


# --- the decentralized WINDOW (squidmip/_region_viewer.RegionViewer) under offscreen Qt ---------
#
# Since the decentralization (2026-07-23) the root PlateWindow has no central viewer: `_detail` is
# permanently None and viewing happens in independent `RegionViewer` windows, each with its OWN
# napari pane, region cursor + slider, and autofocus. Those windows are where the region slider and
# the focus/z-slider paths went, so the tests that used to drive them through `_detail` have to
# drive them through a real window instead.
#
# The obstacle is napari, not the window: `make_pane` calls `gl_available()`, which is false under
# `QT_QPA_PLATFORM=offscreen` (what the whole suite runs under), so `_build` takes its "napari
# viewer unavailable" branch and never constructs the slider at all. `napari_pane_stub` replaces the
# ONE seam that needs a GL context with a recording stand-in, so everything downstream of it -- the
# real RegionViewer, the real RegionCursor/RegionSlider, the real `_MosaicWorker`, the real
# `_FocusWorker` -- is the production code. That is a real limitation and it is stated rather than
# hidden: napari's own rendering is not exercised here, and never was under offscreen.
def _stub_pane_classes():
    """Build the stub classes lazily: conftest must import without the [gui] extra."""
    from qtpy.QtWidgets import QWidget

    def _as_napari_data(data, multiscale):
        """What napari would hand BACK for *data*, which is not always what went in.

        A multiscale layer's ``.data`` is ``napari.layers._multiscale_data.MultiScaleData`` -- a
        Sequence of levels that is neither a list nor a tuple, and that reports level 0's ndim,
        shape and dtype as its own while ``np.asarray`` of it yields the COARSEST level.

        This stub used to hand back the plain list it was given, and that lie is why three
        production sites drifted into ``isinstance(data, (list, tuple))`` pyramid checks that are
        False for every real pyramid: ``_workers._full_res_mip`` (AttributeError: 'MultiScaleData'
        object has no attribute 'max' -- Julio, running cellpose on the 10x set),
        ``MosaicLayers._swap_layer_scale`` (a silent no-op) and
        ``RegionViewer._render_roi_volume`` (silently the coarsest level). A stub that answers
        differently from the production object cannot catch any of them.
        """
        if not multiscale or not isinstance(data, (list, tuple)):
            return data
        try:
            from napari.layers._multiscale_data import MultiScaleData

            return MultiScaleData(data)
        except Exception:                    # noqa: BLE001 - napari is a [gui] extra
            return data

    def _as_napari_placement(data, kw):
        """``(scale, translate)`` napari would report, which is NOT what the caller passed.

        ``MosaicLayers.add_mosaic`` takes ``bbox_um`` and places the layer with
        ``_napari_view.placement_for(ndim, bbox_um, shape[-2:], z_scale_um)`` -- bbox / shape --
        so the layer's own ``scale`` is the pitch of the pixels IT holds, whatever decimation
        produced them (measured on the 10x set: 1.504 um/px for a mosaic fused at step 2, against
        an acquisition ``pixel_size_um`` of 0.752). This stub reported ``scale=None`` for every
        mosaic, which is a layer no production code path ever sees, and it is why a reader of the
        pitch could take ``pixel_size_um`` instead and stay green.
        """
        scale, translate = kw.get("scale"), kw.get("translate")
        bbox = kw.get("bbox_um")
        if scale is not None or bbox is None:
            return scale, translate
        try:
            from squidmip._napari_view import placement_for

            level0 = data[0] if isinstance(data, (list, tuple)) else data
            return placement_for(int(level0.ndim), bbox, level0.shape[-2:], kw.get("z_scale_um"))
        except Exception:                        # noqa: BLE001 - unplaceable: report it as such
            return scale, translate

    def _first_stub_level(data):
        """Level 0 of whatever `add_mosaic` was handed -- multiscale list, MultiScaleData, or array.

        The dtype question is asked of LEVEL 0 for the same reason production asks it there: every
        rung of a fused pyramid shares one dtype, but only level 0 is guaranteed to exist.
        """
        try:
            return data[0] if isinstance(data, (list, tuple)) or hasattr(data, "__len__") and not (
                hasattr(data, "dtype")) else data
        except Exception:                        # noqa: BLE001 - unindexable: it IS the level
            return data

    class StubLayer:
        """A napari image layer as RegionViewer reads it back."""

        def __init__(self, data, kw):
            self.data = _as_napari_data(data, kw.get("multiscale"))
            self.scale, self.translate = _as_napari_placement(data, kw)
            self.contrast_limits = None
            #: The slider's TRAVEL, which is not its value. `MosaicLayers.add_mosaic` sets this on
            #: every real layer and `_widen_range` reads it back; a stub without it made both a
            #: silent no-op, because `_widen_range` treats a layer with no range as "not an
            #: intensity layer" and returns False. So the stub reported that widening had reached
            #: nothing, which is indistinguishable from the production code not widening at all.
            from squidmip import _bitdepth
            self.contrast_limits_range = _bitdepth.range_for(
                getattr(_first_stub_level(self.data), "dtype", None))
            self.colormap = kw.get("colormap")
            #: A result delivered to a window that did not ask for the run arrives DARK. That is
            #: a property of `deliver_result`, so the stub has to carry it or no test can see it.
            self.visible = bool(kw.get("visible", True))

    class StubMosaic:
        """The `MosaicPane.mosaic` surface RegionViewer drives, recording what it was handed."""

        def __init__(self):
            self.model = None                 # napari Viewer; None -> `_napari_viewer` uses _viewer
            self.added = []                   # (op, channel, levels, kwargs) per add_mosaic
            self.contrast_subscribers = []     # what _bind_window_contrast subscribed
            self.visibility_subscribers = []
            self.colormap_subscribers = []
            self.op_subscribers = []           # which PROCESSING LAYER this window is showing
            self.removed = []
            self.shown = []
            self._layers = {}

        def ops(self):
            """The processing layers this pane holds, in insertion order — `MosaicLayers.ops`.

            It was MISSING, and its absence was invisible in exactly the way this file's other
            stub notes describe. ``RegionViewer._window_operators`` calls it inside a bare
            ``except Exception: return []``, so against this stub every window reported holding NO
            operators — and `⚙ controls`'s whole plural-tab behaviour could only be tested by
            replacing ``pane.mosaic`` with a hand-written object that had one. The stub agreed
            with the tests and neither of them touched the production accessor.
            """
            seen = []
            for op, _channel in self._layers:
                if op not in seen:
                    seen.append(op)
            return seen

        def channels(self, op):
            """`MosaicLayers.channels` — the channels of ONE processing layer, in insertion order.

            Missing, like `ops` was, and missing in the same invisible way: it is read at
            `_region_viewer._volume_source` immediately after `visible_op`, inside the same bare
            `except Exception` that answers "raw" for anything that raises.
            """
            out = []
            for layer_op, channel in self._layers:
                if layer_op == op and channel not in out:
                    out.append(channel)
            return out

        def visible_op(self):
            """WHICH processing layer is on screen — `MosaicLayers.visible_op`.

            It was MISSING, and `RegionViewer._volume_source` calls it inside
            ``try: ... except Exception: return None, _RAW_OP, None``. So against this stub every
            window answered RAW, unconditionally, and the whole operator branch below that call
            -- `_reduces_z`'s refusal of a z-reducer's single plane, and the per-operator channel
            list -- was unreachable from any window test. `tests/test_roi_pitch.py`'s
            "no operator is displayed, so this is the reader" then asserted an answer the stub
            could not have given any other way.
            """
            for (op, _channel), layer in self._layers.items():
                if getattr(layer, "visible", True):
                    return op
            return None

        @staticmethod
        def _reduces_z(op):
            """The DECLARATION, exactly as `MosaicLayers._reduces_z` reads it — never the name."""
            from squidmip._engine import Z_REDUCER, operator_consumes
            from squidmip._operations import operator_name

            try:
                return operator_consumes(operator_name(str(op))) == Z_REDUCER
            except Exception:                        # noqa: BLE001 - not a registered operator
                return False

        def set_channel_visible(self, channel, visible):
            """Show/hide one channel across the VISIBLE processing layer only.

            Missing, and `RegionViewer._apply_view_settings` calls it inside
            ``except Exception: pass  # a missing channel is skipped``, so restoring a saved
            channel visibility was a total no-op under test while `_apply_luts` -- the call on the
            line above it, over members this stub does have -- was honestly covered.
            """
            current = self.visible_op()
            if current is None:
                return
            for (op, ch), layer in self._layers.items():
                if op == current and ch == channel:
                    layer.visible = bool(visible)

        def channel_visible(self, channel):
            peers = [ly for (_op, ch), ly in self._layers.items() if ch == channel]
            if not peers:
                return None
            return any(bool(getattr(p, "visible", False)) for p in peers)

        def channel_rgb(self, channel):
            for (_op, ch), layer in self._layers.items():
                if ch == channel:
                    return getattr(layer, "colormap", None)
            return None

        def layers_for(self, op, channel):
            layer = self._layers.get((op, channel))
            return [layer] if layer is not None else []

        def remove_op(self, op):
            self.removed.append(op)
            for key in [k for k in self._layers if k[0] == op]:
                del self._layers[key]

        # The three sinks the plate subscribes to when it starts following a window
        # (_bind_window_contrast). Without these the stub could only prove the plate TOLERATES a
        # window; with them a test can fire a gesture and watch the plate react, which is the
        # behaviour Julio asked for: "there shouldn't be any controls for the plate view, it just
        # reacts to toggles and contrast adjustments in napari."
        def on_user_contrast(self, cb):
            self.contrast_subscribers.append(cb)

        def on_user_visibility(self, cb):
            self.visibility_subscribers.append(cb)

        def on_user_colormap(self, cb):
            self.colormap_subscribers.append(cb)

        def on_user_op(self, cb):
            self.op_subscribers.append(cb)

        def gesture_op(self, op, on):
            """Pretend the user ticked a PROCESSING LAYER in this window's layer tree."""
            for cb in list(self.op_subscribers):
                cb(op, on)

        def gesture_contrast(self, channel, lo, hi):
            """Pretend the user dragged contrast in napari."""
            for cb in list(self.contrast_subscribers):
                cb(channel, lo, hi)

        def gesture_visibility(self, channel, on):
            """Pretend the user clicked an eye icon."""
            for cb in list(self.visibility_subscribers):
                cb(channel, on)

        def add_mosaic(self, op, channel, levels, **kw):
            self.added.append((op, channel, levels, kw))
            layer = StubLayer(levels, kw)
            self._layers[(op, channel)] = layer
            return layer

        def add_result(self, kind, op, channel, data, **kw):
            """The sink `RegionViewer.deliver_result` actually calls for an OPERATOR result.

            It was missing, and its absence was invisible: `deliver_result` wraps the call in
            ``except Exception`` so an operator layer that cannot be added is named rather than
            lost, and against a stub with no `add_result` that turned every delivery into a
            reported no-op. A stub that answers a narrower surface than the production object
            silently changes what the code under test does.
            """
            self.added.append((op, channel, data, dict(kw, kind=kind)))
            layer = StubLayer(data, kw)
            self._layers[(op, channel)] = layer
            return layer

        def match_contrast_to(self, op):
            """The "Match raw contrast" action: *op*'s window onto the channel's other layers.

            The real one is `MosaicLayers.match_contrast_to`, pinned against a real napari
            ViewerModel in test_napari_view.py. This stub exists so a WINDOW test can assert the
            chip's handler reaches a mosaic at all, on the layer values rather than on the call.
            """
            matched = 0
            for (layer_op, channel), layer in list(self._layers.items()):
                source = self._layers.get((op, channel))
                if source is None or layer_op == op:
                    continue
                layer.contrast_limits = source.contrast_limits
                matched += 1
            return matched

        def show_op(self, op):
            self.shown.append(op)

        def find(self, op, channel):
            return self._layers.get((op, channel))

        def widen_contrast_range(self, lo, hi):
            """`MosaicLayers.widen_contrast_range` -- open every slider, never narrow one.

            Called by `RegionViewer._on_depth_changed` when a later region proves the dataset
            holds bigger numbers than the region this window opened on. Without it here the
            handler's own ``except Exception`` would swallow the AttributeError and every test
            would agree that already-open windows do not widen.
            """
            moved = 0
            for layer in self._layers.values():
                r = getattr(layer, "contrast_limits_range", None)
                if r is None:
                    continue
                new = (min(float(r[0]), float(lo)), max(float(r[1]), float(hi)))
                if new != (float(r[0]), float(r[1])):
                    layer.contrast_limits_range = new
                    moved += 1
            return moved

    class StubDims:
        """napari `Dims`, only the parts a window's autofocus drives."""

        def __init__(self, nsteps=(2, 8, 8)):
            self.nsteps = tuple(nsteps)
            self.ndim = len(self.nsteps)
            self.ndisplay = 2
            self.current_step = tuple(0 for _ in self.nsteps)

    class StubViewer:
        def __init__(self):
            self.dims = StubDims()
            self.layers = []

    class StubPane(QWidget):
        ok = True

        def __init__(self):
            super().__init__()
            self.mosaic = StubMosaic()
            self._viewer = StubViewer()
            self.detect_channel = None
            self.detect_button = None
            self.said = []
            self.shutdowns = 0

        def say(self, text):
            self.said.append(text)

        def shutdown(self):
            """COUNTS, rather than no-ops. The real ``MosaicPane.shutdown`` is what closes the
            napari Viewer and drops it from napari's instance registry, and it went uncalled for
            long enough to leak a GL context per closed window. A stub that silently accepted the
            call could not tell the difference between "disposed" and "never disposed" -- and
            ``dispose`` wraps every teardown in ``except Exception``, so a MISSING method here
            would be swallowed and read as success."""
            self.shutdowns += 1
            self._viewer = None

    return StubPane


def shutdown_plate_window(app, win):
    """Close a PlateWindow AND every window it spawned, then let Qt actually delete them.

    `RegionViewer` sets `WA_DeleteOnClose`, so `close()` only SCHEDULES the deletion; the widget,
    its napari QtDims and its worker threads stay alive until the event loop runs. A test that
    closes only the plate leaves its windows behind, and enough of them accumulated in one process
    segfault a later test -- an order-dependent crash that takes pytest's summary with it, which is
    exactly the failure mode that hid this suite's real failures for weeks. Draining here is the
    difference between "the window was closed" and "the window is gone".
    """
    manager = getattr(win, "_viewer_manager", None)
    if manager is not None:
        manager.close_all()
    win.close()
    for _ in range(20):
        app.processEvents()
    # Collect HERE, at a controlled point with the app still alive, rather than letting the cycle
    # collector fire in the middle of an unrelated later test: a Qt wrapper whose C++ half Qt has
    # already destroyed segfaults on collection, and a segfault takes pytest's summary with it.
    gc.collect()
    app.processEvents()


@pytest.fixture
def napari_pane_stub(monkeypatch):
    """Make `RegionViewer` buildable headlessly. Returns the list of panes handed out.

    Every window opened while this fixture is active gets its own recording pane, in open order.
    """
    import squidmip._napari_pane as napari_pane

    stub_pane_cls = _stub_pane_classes()
    panes = []

    def _make_pane(*_args, **_kw):
        pane = stub_pane_cls()
        panes.append(pane)
        return pane, "napari", ""

    monkeypatch.setattr(napari_pane, "make_pane", _make_pane)
    return panes


# ---------------------------------------------------------------------------------------------
# THE TWO SCENES ONE RULE IS ASKED OF: a flat mosaic and a bricked volume
# ---------------------------------------------------------------------------------------------
#
# Julio: "why is the layering of 2d and 3d different in the same place?"
#
# Every rule about layers -- what the tree shows, what a checkbox reaches, what one contrast
# window covers -- has to hold for both, so the tests that pin those rules are parametrized over
# these two builders. They live HERE rather than in one of the two test files that use them
# (`test_viewer_3d.py`, `test_layer_tree.py`) so that "what a 3D scene is" has one definition: a
# second copy is how the two modes drifted apart in the first place.
#
# Both build into a real `napari.components.ViewerModel`, which is Qt-free, so a scene is real
# napari layers with real evented properties rather than a stub that agrees by construction.

def _scene_stack(seed=0, shape=(4, 16, 16)):
    import numpy as np

    return np.random.default_rng(seed).integers(0, 4000, shape, dtype=np.uint16)


def build_flat_scene(mosaic, op="raw", channels=("488", "561")):
    """THE 2D SCENE: one mosaic layer per (op, channel), as a region window builds it."""
    for i, ch in enumerate(channels):
        mosaic.add_mosaic(op, ch, _scene_stack(i, (4, 16, 16)), bbox_um=(0.0, 0.0, 16.0, 16.0))
    return mosaic


def build_volume_scene(mosaic, op="raw", channels=("488", "561"), bricks=3):
    """THE 3D SCENE: a bricked volume of the same op and channels, in the same viewer.

    Goes through the real `BrickedVolume._add_layer`, so what is exercised is the production path
    -- the brick kwargs, `pin_max_compositing` and `MosaicLayers.adopt` -- and not a re-statement
    of it. The loader thread is the one thing stubbed: starting a real QThread with no
    QApplication aborts the interpreter, and reading pixels is not what these rules are about.

    SEVERAL bricks by default. A rule that only holds for a one-brick volume is the bug wearing a
    disguise: one brick and one mosaic layer are the same shape, and the shape is the problem.
    """
    from squidmip._brick_view import BrickedVolume

    vol = BrickedVolume(
        mosaic, reader=None, meta={}, region="A1", window_px=(0, 8, 0, 8),
        channels=list(channels), scale=(1.5, 0.75, 0.75), origin_um=(0.0, 0.0, 0.0),
        limit=2048, budget_bytes=1 << 30, op=op,
    )
    vol._loader.start = lambda *a, **k: None
    vol._loader.stop = lambda *a, **k: None
    vol._loader.wait = lambda *a, **k: True
    for ch_i, ch in enumerate(channels):
        for b in range(bricks):
            vol._add_layer((ch, (0, b)), ch, _scene_stack(10 + ch_i * 10 + b, (4, 8, 8)),
                           (1.5, 0.75, 0.75), (0.0, 0.0, float(b) * 6.0))
    return vol
