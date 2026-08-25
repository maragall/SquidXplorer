"""The peak footprint has to be a real number on Windows."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from squidxplorer._footprint import rss_mb

_MB = 1024 * 1024


class _Info:
    """Stand-in for `psutil.Process().memory_info()`; only Windows builds carry `peak_wset`."""

    def __init__(self, rss: int, peak_wset: "int | None" = None) -> None:
        self.rss = rss
        if peak_wset is not None:
            self.peak_wset = peak_wset


class _Process:
    def __init__(self, info: _Info) -> None:
        self._info = info

    def memory_info(self) -> _Info:
        return self._info


class _Usage:
    """Stand-in for `resource.getrusage(RUSAGE_SELF)`."""

    def __init__(self, ru_maxrss: int) -> None:
        self.ru_maxrss = ru_maxrss


def test_the_windows_peak_is_the_peak_working_set_never_the_posix_module_and_never_zero():
    """PeakWorkingSetSize, not the 0 it used to print; `peak_wset == 0` is no measurement, so fall through to rss."""
    proc = _Process(_Info(rss=91 * _MB, peak_wset=419 * _MB))
    assert rss_mb(platform="win32", process=proc, usage=None) == (419.0, 91.0)
    assert rss_mb(platform="win32", process=proc, usage=_Usage(ru_maxrss=1024))[0] == 419.0
    zero = _Process(_Info(rss=91 * _MB, peak_wset=0))
    assert rss_mb(platform="win32", process=zero, usage=None) == (91.0, 91.0)


def test_posix_reads_ru_maxrss_in_the_platforms_unit_and_prefers_it_over_the_current_rss():
    assert rss_mb(platform="linux", process=None, usage=_Usage(ru_maxrss=419 * 1024)) == (419.0, None)
    assert rss_mb(platform="darwin", process=None, usage=_Usage(ru_maxrss=419 * _MB))[0] == 419.0
    proc = _Process(_Info(rss=91 * _MB))
    assert rss_mb(platform="darwin", process=proc, usage=_Usage(ru_maxrss=419 * _MB)) == (419.0, 91.0)


def test_with_no_peak_source_the_rss_is_the_floor_and_nothing_measurable_never_raises():
    """This runs in an excepthook, i.e. while the app is already dying; it must never raise."""
    proc = _Process(_Info(rss=91 * _MB))
    assert rss_mb(platform="win32", process=proc, usage=None) == (91.0, 91.0)
    assert rss_mb(platform="win32", process=None, usage=None) == (0.0, None)

    class _Broken:
        def memory_info(self):
            raise OSError("access denied")

    assert rss_mb(platform="win32", process=_Broken(), usage=None) == (0.0, None)


def test_the_live_process_reports_a_nonzero_peak_through_the_viewer_entry_point():
    from squidxplorer._viewer import _rss_mb

    peak, cur = rss_mb()
    assert peak > 0.0 and (cur is None or cur > 0.0)
    assert _rss_mb()[0] > 0.0
