"""Building the root window must not put a window on the user's desktop that nobody asked for."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys  # noqa: E402

import pytest  # noqa: E402

pytest.importorskip("qtpy")
if "PySide6" in sys.modules or "PySide2" in sys.modules:
    pytest.skip(
        "PySide already loaded (napari/pytest-qt) — Qt binding conflict; run with "
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 to run the PyQt5 GUI tests.",
        allow_module_level=True,
    )

from qtpy.QtWidgets import QApplication  # noqa: E402

from squidxplorer import _viewer as V  # noqa: E402

from .conftest import shutdown_plate_window  # noqa: E402
from .test_viewer import qapp  # noqa: E402,F401  (fixtures)


def _visible_top_levels():
    """Every top-level widget Qt currently considers visible, as readable descriptions."""
    out = []
    for w in QApplication.topLevelWidgets():
        try:
            if not w.isVisible():
                continue
        except RuntimeError:
            continue
        label = getattr(w, "text", None)
        out.append(f"{type(w).__name__}(title={w.windowTitle()!r}, "
                   f"text={label() if callable(label) else ''!r}, "
                   f"{w.width()}x{w.height()})")
    return out


def test_building_and_ingesting_shows_no_window_the_caller_did_not_open(
        qapp, squid_dataset):
    root, _ = squid_dataset
    before = set(_visible_top_levels())

    win = V.PlateWindow(None)
    win.ingest(str(root))

    assert len(win._meta["z_levels"]) > 1, (
        "fixture invalid: the orphan only un-hid itself on a MULTI-z acquisition, so a "
        "single-plane fixture would pass this test without testing anything")
    strays = sorted(set(_visible_top_levels()) - before)
    assert strays == [], f"the root put {len(strays)} window(s) on screen by itself: {strays}"
    shutdown_plate_window(qapp, win)


def _ancestry(widget):
    """The chain of parents from *widget* upward, as type names — for a readable failure."""
    chain, w = [], widget
    while w is not None:
        chain.append(type(w).__name__)
        w = w.parentWidget()
    return " -> ".join(chain)


def test_a_published_qc_result_is_really_inside_the_window_and_really_visible(
        qapp, squid_dataset):
    from squidxplorer._op_panels import DeconQCResultView

    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.resize(900, 900)
    win.show()
    win.ingest(str(root))
    qapp.processEvents()

    view = DeconQCResultView("B2/0/c0")
    win.publish_qc_result(view, "Decon QC · B2/0/c0")
    qapp.processEvents()

    ancestors = []
    w = view
    while w is not None:
        ancestors.append(w)
        w = w.parentWidget()
    assert win in ancestors, (
        "the QC result is not inside the plate window at all — it is in a tab bar in a pane "
        f"nothing parented. Its chain is: {_ancestry(view)}")
    assert win.centralWidget() in ancestors, (
        f"the QC result hangs off the window but outside its central widget: {_ancestry(view)}")

    assert view.isVisible(), (
        "the QC result is parented but not on screen — it is in a bar nothing shows")
    assert view.width() > 0 and view.height() > 0, "the QC result has no geometry"

    assert win._left_tabs.indexOf(view) >= 0, (
        "the QC result did not land in the operator tab bar")

    shutdown_plate_window(qapp, win)


def test_the_dead_reference_plane_chain_is_not_on_the_plate_window(qapp,
                                                                  squid_dataset):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))

    for gone in ("_focus_btn", "_sync_focus_button", "_focus_reference_plane",
                 "_on_focus_problem", "_on_reference_plane", "_set_z_index"):
        assert not hasattr(win, gone), (
            f"PlateWindow.{gone} is back. The reference plane lives on each window's z-slider; "
            "a second copy on the plate is what produced the orphan window.")
    shutdown_plate_window(qapp, win)


def test_the_focus_worker_itself_survives_because_the_windows_use_it(qapp,
                                                                    squid_dataset):
    assert hasattr(V, "_FocusWorker")
    from squidxplorer._region_viewer import RegionViewer

    assert hasattr(RegionViewer, "_focus_reference_plane")
    assert hasattr(RegionViewer, "_on_reference_plane")
