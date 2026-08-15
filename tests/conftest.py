"""Shared test fixtures: tiny on-disk Squid acquisitions, Qt lifecycle pins, and the
headless region-window stubs."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import numpy as np
import pytest
import tifffile

# The suite renders OFFSCREEN by default: real widget tests on the native platform open actual
# windows, and macOS yanks focus to each one — the person at the keyboard loses their typing
# focus for the length of the run. setdefault, not assignment, so a visual debug can still ask
# for real windows with QT_QPA_PLATFORM=cocoa (or xcb). Must be set before QApplication exists.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

#: The process's QApplication, pinned for the whole session so no fixture teardown can free it.
_QT_APP = None

#: The session's plate cell cache directory, created in ``pytest_configure`` and removed at the end.
_CACHE_DIR = None


def pytest_sessionfinish(session, exitstatus):
    """Tear Qt down while Qt is still alive, so the process does not abort at interpreter exit."""
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
    """Pin the QApplication and redirect the plate cell cache before any test runs."""
    # No automatic GC mid-suite: the cyclic collector can fire with C++ frames on the stack and
    # abort the process. pytest_sessionfinish collects explicitly at a safe point.
    gc.disable()

    global _QT_APP, _CACHE_DIR
    import sys
    import tempfile

    from squidxplorer import _platecache

    if not os.environ.get(_platecache.ENV_DIR):
        _CACHE_DIR = tempfile.mkdtemp(prefix="squidxplorer-test-cache-")
        os.environ[_platecache.ENV_DIR] = _CACHE_DIR
    # Do NOT force PyQt5 in: the GUI modules skip themselves when PySide is already loaded,
    # and two Qt bindings in one process break GL rendering.
    if any(m.startswith("PySide") for m in sys.modules):
        return
    try:
        from qtpy.QtWidgets import QApplication
    except ImportError:
        return                      # headless CI without the [gui] extra: nothing to pin
    _QT_APP = QApplication.instance() or QApplication([])
    _keep_test_windows_off_the_foreground()


def _keep_test_windows_off_the_foreground() -> None:
    """Let test windows open without pulling the user to their Space. macOS only, best-effort."""
    import sys

    if sys.platform != "darwin" or os.environ.get("SQUIDXPLORER_TEST_FOREGROUND"):
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

        if getattr(QWidget, "_squidxplorer_no_activate", False):
            return                              # already patched (pytest_configure ran twice)
        original_set_visible = QWidget.setVisible

        def set_visible(self, visible):
            # Every show() funnels through setVisible, so this covers napari's windows too.
            if visible and self.isWindow():
                self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
            return original_set_visible(self, visible)

        QWidget.setVisible = set_visible
        QWidget._squidxplorer_no_activate = True
    except Exception:
        pass


# The process-global operator registry: snapshot + restore around every test, so a leaked
# registration cannot fail an unrelated test file.
_REGISTRIES = (
    ("squidxplorer._engine", "_OPERATORS"),
)


@pytest.fixture(autouse=True)
def _cold_plate_cell_cache(tmp_path, monkeypatch):
    """Every test starts with a COLD plate cell cache (``squidxplorer._platecache``)."""
    from squidxplorer import _platecache

    monkeypatch.setenv(_platecache.ENV_DIR, str(tmp_path / "plate-cells"))
    _platecache.clear_memory_tier()
    yield
    _platecache.clear_memory_tier()


@pytest.fixture(autouse=True)
def _cold_result_cache():
    """Every test starts and ends with an EMPTY ``squidxplorer._recipe.RESULTS``."""
    from squidxplorer import _recipe

    _recipe.RESULTS.clear()
    yield
    _recipe.RESULTS.clear()


@pytest.fixture(autouse=True)
def _restore_operator_registries():
    import importlib

    saved = []
    for module_name, attr in _REGISTRIES:
        try:
            registry = getattr(importlib.import_module(module_name), attr)
        except (ImportError, AttributeError):
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
# One channel present in the YAML, one absent (exercises the wavelength fallback).
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

# Legacy flat sidecar. Its recomputed px (0.188) deliberately differs from acquisition.yaml's
# stored 0.325 so tests prove which is used. NA is what every real acquisition writes.
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


# Real Squid coordinates.csv schema: region + x/y in mm, empty z column, no fov column.
# Rows repeat once per z-level, as a multi-z acquisition writes them.
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
    """An acquisition with 640px fields — big enough for a real two-level pyramid; returns (root, region, size)."""
    root = tmp_path / "acq_pyr"
    size, region, ch = 640, "B2", CH_IN_YAML
    folder = root / "0"
    folder.mkdir(parents=True, exist_ok=True)
    for z in range(2):
        # A gradient plus a per-z offset keeps downsampled levels distinguishable.
        yy, xx = np.mgrid[0:size, 0:size]
        arr = ((yy + xx).astype(np.uint16) + z * 7)
        tifffile.imwrite(folder / f"{region}_0_{z}_{ch}.tiff", arr)
    (root / "acquisition_channels.yaml").write_text(_YAML)
    (root / "acquisition.yaml").write_text(_ACQ_YAML)
    (root / "acquisition parameters.json").write_text(json.dumps(_PARAMS))
    return root, region, size


# The multi-timepoint acquisition: the ONLY Nt>1 fixture in the suite, written in Squid's own
# on-disk layout (one folder per timepoint, executed coordinates.csv + .done marker per folder).
TIME_SERIES_REGION = "A1"
TIME_SERIES_FOV = 0
TIME_SERIES_NZ = 2
N_TIME_POINTS = 3

#: In the reader's sorted order — the fixture's pixel values encode a channel INDEX.
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
    """The constant every plane of the multi-timepoint fixture is filled with."""
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
    """A Squid individual-TIFF acquisition with ``Nt=3``; returns ``(root, planes)``."""
    root = tmp_path / "acq_time_series"
    planes: dict = {}
    for time_point in range(N_TIME_POINTS):
        # FILE_ID_PADDING is 0 in Squid, so this is str(time_point); written the long way so the
        # fixture still tracks Squid if a deployment ever raises the padding.
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


# One fixture per Squid output writer. Builders are imported inside each fixture because
# writer_fixtures imports this module's shape constants (top-level import would be circular).

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
    """The real 10x laser-AF tissue acquisition; else skip (used by integration tests)."""
    path = Path("/Users/julioamaragall/Downloads/"
                "test_10x_laser_af_z_stack_2025-10-28_13-40-43.939945 yy")
    if not path.is_dir():
        pytest.skip(f"real tissue acquisition not present at {path}")
    return path


SIM_1536WP = Path("/Users/julioamaragall/CEPHLA/Data/sim_1536wp")

#: Every plane of sim_1536wp is a symlink into this real, read-only 20x scan.
SIM_1536WP_SOURCE = Path("/Users/julioamaragall/Downloads/20x_scan_2025-09-05_17-57-50")


def sim_1536wp_problem():
    """Why ``sim_1536wp`` is unusable (absent or hollow), or None when it is fine."""
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
    """The 1536-well plate-scale acquisition, or a skip that says exactly what is wrong."""
    problem = sim_1536wp_problem()
    if problem:
        pytest.skip(problem)
    return SIM_1536WP


# Headless RegionViewer support: `napari_pane_stub` replaces the one seam that needs a GL context
# (make_pane) with a recording stand-in; everything downstream is production code.
def _stub_pane_classes():
    """The headless pane class: squidxplorer's OWN ModelPane — a real Qt-free ``ViewerModel``
    with a real ``MosaicLayers`` over it. The hand-synced StubMosaic/StubLayer reimplementation
    is deleted: it had to mirror every MosaicLayers change by hand, and its last drift let four
    sites go wrong with the suite green. Tests now cross the same interface production crosses.
    """
    from squidxplorer._napari_pane import model_pane_class

    return model_pane_class()


def shutdown_plate_window(app, win):
    """Close a PlateWindow AND every window it spawned, then let Qt actually delete them."""
    manager = getattr(win, "_viewer_manager", None)
    if manager is not None:
        manager.close_all()
    win.close()
    for _ in range(20):
        app.processEvents()
    # Collect here, with the app still alive, so the cycle collector cannot fire mid-test on
    # wrappers whose C++ half Qt has already destroyed.
    gc.collect()
    app.processEvents()


@pytest.fixture
def napari_pane_stub(monkeypatch):
    """Make `RegionViewer` buildable headlessly. Returns the list of panes handed out."""
    import squidxplorer._napari_pane as napari_pane

    stub_pane_cls = _stub_pane_classes()
    panes = []

    def _make_pane(*_args, **_kw):
        pane = stub_pane_cls()
        panes.append(pane)
        return pane, "napari", ""

    monkeypatch.setattr(napari_pane, "make_pane", _make_pane)
    return panes


# The two scenes every layer rule is asked of: a flat mosaic and a bricked volume. Both build
# into a real Qt-free `napari.components.ViewerModel`, so the layers and events are real.

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

    Uses the real `BrickedVolume._add_layer` path; only the loader thread is stubbed (a real
    QThread with no QApplication aborts the interpreter). Several bricks by default on purpose.
    """
    from squidxplorer._brick_view import BrickedVolume

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
