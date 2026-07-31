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
_REGISTRIES = (
    ("squidmip._engine", "_PROJECTORS"),
    ("squidmip._spots", "_SEGMENTERS"),
    ("squidmip._stitch", "_REGION_OPERATORS"),
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
_PARAMS = {
    "Nz": NZ,
    "Nt": 1,
    "dz(um)": 1.5,
    "objective": {"magnification": 20.0},
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
SIM_1536WP_SOURCE = Path("/Users/julioamaragall/Downloads/synthetic_2x2_wellplate")


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
            f"To restore it, put the source acquisition back at {SIM_1536WP_SOURCE}: the 1536 "
            "wells are symlinks onto that one 2x2 plate's four planes, so restoring the source "
            "revives every link with no regeneration step. This repo contains no generator for "
            f"{SIM_1536WP_SOURCE.name}; tools/gates.py, tools/walkthrough.py, tools/acceptance.py "
            "and tools/odon_benchmark.py all read it from that path and none of them create it."
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

    class StubLayer:
        """A napari image layer as RegionViewer reads it back: `.data` is the level list."""

        def __init__(self, data, kw):
            self.data = data
            self.scale = kw.get("scale")
            self.translate = kw.get("translate")
            self.contrast_limits = None
            self.colormap = kw.get("colormap")

    class StubMosaic:
        """The `MosaicPane.mosaic` surface RegionViewer drives, recording what it was handed."""

        def __init__(self):
            self.model = None                 # napari Viewer; None -> `_napari_viewer` uses _viewer
            self.added = []                   # (op, channel, levels, kwargs) per add_mosaic
            self.contrast_subscribers = []     # what _bind_window_contrast subscribed
            self.visibility_subscribers = []
            self.colormap_subscribers = []
            self.removed = []
            self.shown = []
            self._layers = {}

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
            self._layers[(op, channel)] = StubLayer(levels, kw)

        def show_op(self, op):
            self.shown.append(op)

        def find(self, op, channel):
            return self._layers.get((op, channel))

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

        def say(self, text):
            self.said.append(text)

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
