"""3D volume rendering of a z-stack (IMA-255) — the SquidMIP half of the seam.

ndviewer_light owns the renderer; this repo owns exactly one thing: telling it the
PHYSICAL voxel size when a raw z-stack is pushed. Without that the volume renders
isotropic, which on the tissue set (dz 1.5um, pixel 0.752um) is 2x squashed in z.

Two guards live here:

* the raw push carries pixel_size_um and dz_um, in micrometres, from the acquisition
  metadata — asserted as NUMBERS off a real fixture, not as "the call happened";
* the INSTALLED ndviewer_light actually accepts those parameters. A stale installed copy
  that silently lacked ``register_array`` once cost this project a day of black-canvas
  debugging; the same failure mode here would quietly restore isotropic rendering, which
  looks plausible and is wrong. So it is checked against the live install, by signature.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # headless Qt; must precede PyQt import

import sys  # noqa: E402

import pytest  # noqa: E402

pytest.importorskip("qtpy")
if "PySide6" in sys.modules or "PySide2" in sys.modules:
    pytest.skip(
        "PySide already loaded (napari/pytest-qt) — Qt binding conflict; run with "
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 to run the PyQt5 GUI tests.",
        allow_module_level=True,
    )

from squidmip import _viewer as V  # noqa: E402

from .conftest import shutdown_plate_window  # noqa: E402
from .test_viewer import _drain_until, qapp, stub_detail  # noqa: E402,F401  (fixtures)

# WHERE THE VOXEL SIZE NOW GOES (decentralization, 2026-07-23).
#
# `PlateWindow._detail` is permanently None and `_make_detail_viewer` has no production call sites,
# so `start_acquisition(..., pixel_size_um, dz_um)`, the seam the pushes below used to be asserted
# on, is not on any path the app takes. The declaration itself did not disappear: a region window
# hands napari `z_scale_um=meta["dz_um"]` with every mosaic (`_region_viewer.py:874`), and the 3D
# volume pushes hand it `scale=(dz, px, px)` (`_region_viewer.py:1038` and `:1078`).
#
# So the three tests below are re-pointed, not deleted: same question ("does the acquisition's real
# voxel size reach the renderer, as NUMBERS"), asked of the object that now answers it.


def _open_window(win, regions):
    w = win._viewer_manager.open(list(regions))
    assert w is not None, "no window was opened"
    return w


def _wait_for_layers(qapp, pane, timeout=30):
    assert _drain_until(qapp, lambda: bool(pane.mosaic.added), timeout=timeout), (
        "no mosaic ever reached the window's viewer")
    return pane.mosaic.added


# The ndviewer_light seam checks lived here: they asked the INSTALLED ndviewer_light whether
# start_acquisition still took pixel_size_um and dz_um, so a silent upstream signature change
# could not make 3D volumes render isotropic.
#
# Deleted with the fallback itself on 2026-07-30. They are worth a note rather than a silent
# removal, because they are what FOUND the Qt6 blocker: `pytest.importorskip("ndviewer_light.core")`
# pulled PyQt5 into a QT_API=pyqt6 process at module scope, vispy then refused to load PyQt6
# beside it, and the file aborted in teardown. A test that imports a second Qt binding is not a
# cheap test. The seam it guarded no longer exists: napari is the only renderer, and the voxel
# scale it is handed is asserted directly below.


class TestRawPushCarriesVoxelSize:
    """The raw z-stack push is the only one that declares a real n_z, so the only one
    where a volume means anything — and the only one that must carry the voxel size."""

    def test_a_window_declares_the_acquisitions_voxel_depth_to_napari(
        self, qapp, stub_detail, napari_pane_stub, squid_dataset  # noqa: F811
    ):
        """Every raw mosaic a window adds carries the acquisition's dz, in micrometres.

        Without it napari scales z as 1, and on the tissue set (dz 1.5um, pixel 0.752um) the volume
        renders 2x squashed, a picture that looks entirely plausible and is wrong.

        MUTATION: drop `z_scale_um` from the `add_mosaic` call -> None -> red.
        """
        root, _ = squid_dataset
        win = V.PlateWindow(None)
        win.ingest(str(root))
        w = _open_window(win, win._order)
        added = _wait_for_layers(qapp, napari_pane_stub[-1])

        meta = win._meta
        # The NUMBER, not just "something was passed": a None here is exactly the
        # silent degradation this test exists to catch.
        for op, channel, _levels, kw in added:
            assert kw.get("z_scale_um") == meta["dz_um"], (
                f"{op}/{channel} was added with z_scale_um={kw.get('z_scale_um')!r}")
        assert meta["dz_um"] is not None and meta["dz_um"] > 0
        assert meta["pixel_size_um"] is not None and meta["pixel_size_um"] > 0
        shutdown_plate_window(qapp, win)

    def test_the_3d_volume_push_carries_the_full_voxel_scale(
        self, qapp, stub_detail, napari_pane_stub, squid_dataset, monkeypatch  # noqa: F811
    ):
        """dz_um / pixel_size_um is the z stretch the renderer applies, and it must be
        computable from the pushed values alone, and finite and positive.

        The 3D push declares all three axes at once as `scale=(dz, px, px)`, so the aspect is
        recoverable as scale[0] / scale[1].

        MUTATION: pass `scale=(1.0, px, px)` (or omit it) -> aspect 1 -> red.
        """
        import squidmip._napari3d as napari3d

        root, _ = squid_dataset
        win = V.PlateWindow(None)
        win.ingest(str(root))
        w = _open_window(win, win._order)
        pane = napari_pane_stub[-1]
        _wait_for_layers(qapp, pane)

        pushes = []
        monkeypatch.setattr(
            napari3d, "open_native_3d_volume",
            lambda volumes, **kw: pushes.append((volumes, kw)) or object())

        w._render_roi_volume(pane.mosaic, {}, {})

        assert pushes, f"nothing was pushed in 3D: {pane.said}"
        _volumes, kw = pushes[-1]
        scale = kw["scale"]
        assert len(scale) == 3, scale
        assert scale[0] == win._meta["dz_um"]
        assert scale[1] == scale[2] == win._meta["pixel_size_um"]
        aspect = scale[0] / scale[1]
        assert aspect > 0
        assert aspect == pytest.approx(win._meta["dz_um"] / win._meta["pixel_size_um"])
        shutdown_plate_window(qapp, win)

    def test_the_raw_mosaic_declares_the_full_z_stack(
        self, qapp, stub_detail, napari_pane_stub, squid_dataset  # noqa: F811
    ):
        """A volume needs more than one plane; the raw mosaic must not collapse z.

        This is what makes the voxel size mean anything at all: a (1, y, x) layer has no z to
        stretch, so a correct `z_scale_um` on a flattened stack is still an isotropic picture.

        MUTATION: fuse a single z plane (or a MIP) into the window -> leading dim 1 -> red.
        """
        root, _ = squid_dataset
        win = V.PlateWindow(None)
        win.ingest(str(root))
        w = _open_window(win, ["B3"])
        added = _wait_for_layers(qapp, napari_pane_stub[-1])

        n_z = win._meta["n_z"]
        assert n_z > 1, "fixture needs a real z-stack or this asserts nothing"
        for op, channel, levels, _kw in added:
            level0 = levels[0] if isinstance(levels, (list, tuple)) else levels
            assert level0.ndim == 3, f"{op}/{channel} is not a (z, y, x) volume: {level0.shape}"
            assert level0.shape[0] == n_z, (
                f"{op}/{channel} declared {level0.shape[0]} planes, not {n_z}")
        shutdown_plate_window(qapp, win)
