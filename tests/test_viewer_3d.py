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

pytest.importorskip("qtpy.QtWidgets")
if "PySide6" in sys.modules or "PySide2" in sys.modules:
    pytest.skip(
        "PySide already loaded (napari/pytest-qt) — Qt binding conflict; run with "
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 to run the PyQt6 GUI tests.",
        allow_module_level=True,
    )

from squidmip import _viewer as V  # noqa: E402

from .test_viewer import qapp, stub_detail  # noqa: E402,F401  (fixtures)


# ndviewer_light.core imports PyQt5 at MODULE scope. This process is PyQt6 (squidmip pins it),
# and two Qt majors in one process abort the interpreter — so the installed seam is inspected in
# a child process that is allowed to be PyQt5. The assertion is unchanged: it is still the LIVE
# installed signature, not a stub. If ndviewer_light ever moves to qtpy this helper collapses to
# a plain import.
def _installed_start_acquisition_params():
    """Parameter names of the INSTALLED ndviewer_light LightweightViewer.start_acquisition."""
    import json
    import subprocess

    src = (
        "import inspect, json, os\n"
        "os.environ['QT_API'] = 'pyqt5'\n"
        "os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')\n"
        "from ndviewer_light.core import LightweightViewer\n"
        "print(json.dumps(list("
        "inspect.signature(LightweightViewer.start_acquisition).parameters)))"
    )
    env = dict(os.environ)
    env["QT_API"] = "pyqt5"
    proc = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True,
                          timeout=300, env=env)
    if proc.returncode != 0:
        pytest.fail(
            "could not inspect the installed ndviewer_light.core seam "
            f"(rc={proc.returncode}):\n{proc.stderr[-2000:]}"
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])


class TestInstalledViewerAcceptsVoxelSize:
    """The seam contract, checked against whatever ndviewer_light is actually installed."""

    def test_start_acquisition_takes_um_keywords(self):
        params = _installed_start_acquisition_params()

        missing = {"pixel_size_um", "dz_um"} - set(params)
        assert not missing, (
            f"installed ndviewer_light.start_acquisition is missing {sorted(missing)} — "
            "3D volumes will render isotropic (2x wrong in z on a 1.5um/0.752um stack). "
            "Upgrade ndviewer_light; check `pip show -f ndviewer-light` for the LIVE path."
        )

    def test_the_um_naming_invariant_holds(self):
        """Everything in this project is micrometres and every key ends in _um."""
        params = _installed_start_acquisition_params()

        physical = [p for p in params if "size" in p or p.startswith("dz")]
        assert physical, "expected physical-size parameters on the seam"
        assert all(p.endswith("_um") for p in physical), physical


class TestRawPushCarriesVoxelSize:
    """The raw z-stack push is the only one that declares a real n_z, so the only one
    where a volume means anything — and the only one that must carry the voxel size."""

    def test_ingest_declares_the_acquisitions_voxel_size(
        self, qapp, stub_detail, squid_dataset  # noqa: F811
    ):
        root, _ = squid_dataset
        win = V.PlateWindow(None)
        win.ingest(str(root))

        assert win._detail.voxel_um, "no start_acquisition recorded"
        pixel_size_um, dz_um = win._detail.voxel_um[0]
        meta = win._meta

        # The NUMBERS, not just "something was passed" — a None here is exactly the
        # silent degradation this test exists to catch.
        assert pixel_size_um == meta["pixel_size_um"]
        assert dz_um == meta["dz_um"]
        assert pixel_size_um is not None and pixel_size_um > 0
        assert dz_um is not None and dz_um > 0
        win.close()

    def test_voxel_aspect_is_recoverable_from_what_was_pushed(
        self, qapp, stub_detail, squid_dataset  # noqa: F811
    ):
        """dz_um / pixel_size_um is the z stretch the renderer applies. It must be
        computable from the pushed values alone, and must be finite and positive."""
        root, _ = squid_dataset
        win = V.PlateWindow(None)
        win.ingest(str(root))

        pixel_size_um, dz_um = win._detail.voxel_um[0]
        aspect = dz_um / pixel_size_um
        assert aspect > 0
        assert aspect == pytest.approx(win._meta["dz_um"] / win._meta["pixel_size_um"])
        win.close()

    def test_raw_push_declares_the_full_z_stack(
        self, qapp, stub_detail, squid_dataset  # noqa: F811
    ):
        """A volume needs more than one plane; the raw push must not collapse z.

        The processed/mosaic pushes deliberately declare n_z=1 (they are projections),
        which is why they are not given a voxel size.
        """
        root, _ = squid_dataset
        win = V.PlateWindow(None)
        win.ingest(str(root))
        win._detail.registered.clear()
        win.activate_well("B3", 0)

        z_levels = {r[2] for r in win._detail.registered}
        assert len(z_levels) == win._meta["n_z"] > 1
        win.close()
