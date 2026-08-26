"""Drag-onto-the-.app (macOS): the FileOpen event reaches ``PlateWindow.ingest``."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # headless Qt; must precede any Qt import

import sys  # noqa: E402

import pytest  # noqa: E402

pytest.importorskip("qtpy")
if "PySide6" in sys.modules or "PySide2" in sys.modules:
    pytest.skip(
        "PySide already loaded (napari/pytest-qt) — Qt binding conflict; run with "
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 to run the PyQt GUI tests.",
        allow_module_level=True,
    )

from qtpy.QtCore import QEvent  # noqa: E402

from squidxplorer import _viewer as V  # noqa: E402
from .test_viewer import qapp  # noqa: E402,F401  (fixture)


class _FakeFileOpen:
    """The two facts of a QFileOpenEvent, on a class PyQt cannot refuse to build."""

    def __init__(self, path: str) -> None:
        self._path = path

    def type(self):                          # noqa: A003 - the Qt spelling
        return QEvent.Type.FileOpen

    def file(self):
        return self._path


class _FakeOther:
    def type(self):                          # noqa: A003
        return QEvent.Type.User


def _window_with_recorder(qapp):
    win = V.PlateWindow(None)
    opened = []
    win.ingest = opened.append               # the same entry dropEvent uses
    return win, opened


def test_a_file_open_event_ingests_its_path(qapp, squid_dataset):
    root, _ = squid_dataset
    win, opened = _window_with_recorder(qapp)
    try:
        flt = V._FileOpenFilter()
        flt.attach(win)
        assert flt.eventFilter(qapp, _FakeFileOpen(str(root))) is True, \
            "the event must be consumed, or Qt keeps propagating it"
        assert opened == [str(root)], "the dropped path never reached ingest"
    finally:
        win.close()


def test_a_launch_document_is_buffered_until_the_window_exists(qapp, squid_dataset):
    """THE MEASURED TIMING."""
    root, _ = squid_dataset
    flt = V._FileOpenFilter()
    assert flt.eventFilter(qapp, _FakeFileOpen(str(root))) is True
    assert flt.eventFilter(qapp, _FakeFileOpen(str(root) + "-newer")) is True

    win, opened = _window_with_recorder(qapp)
    try:
        flt.attach(win)
        assert opened == [str(root) + "-newer"], \
            "attach must replay the LAST buffered document, and only that one"
        assert flt.eventFilter(qapp, _FakeFileOpen(str(root))) is True
        assert opened[-1] == str(root), "after attach, events must go straight through"
    finally:
        win.close()


def test_an_empty_path_is_ignored_and_other_events_pass_through(qapp):
    win, opened = _window_with_recorder(qapp)
    try:
        flt = V._FileOpenFilter()
        flt.attach(win)
        assert flt.eventFilter(qapp, _FakeFileOpen("")) is True
        assert opened == [], "an empty FileOpen must not ingest"
        assert flt.eventFilter(win, _FakeOther()) is False, \
            "a non-FileOpen event must keep propagating"
        assert opened == []
    finally:
        win.close()


def test_the_bundle_declares_folders_so_finder_accepts_the_drop():
    """The OTHER half: without public.folder in CFBundleDocumentTypes, Finder refuses the drop and no event ever exists for the filter to catch."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "bootstrap", Path(__file__).resolve().parents[1] / "scripts" / "installer" / "bootstrap.py")
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)

    import plistlib

    plist = plistlib.loads(bootstrap._INFO_PLIST.encode())
    types = plist.get("CFBundleDocumentTypes") or []
    assert any("public.folder" in (t.get("LSItemContentTypes") or []) for t in types), \
        "the .app no longer declares folders; drag-onto-the-app is dead at the Finder"
    assert all(t.get("LSHandlerRank") == "Alternate" for t in types), \
        "the app must never claim to be the system's default handler for folders"
