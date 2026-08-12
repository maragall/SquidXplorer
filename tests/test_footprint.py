"""The peak footprint has to be a real number on Windows.

`resource` is POSIX-only; on Windows the peak instead comes from `psutil`'s `peak_wset`
(`PeakWorkingSetSize`), sourced without ever touching `resource`. With no peak source at all,
falls back to the current rss (a true lower bound) rather than to 0 (a false statement).
"""
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


def test_the_windows_peak_is_the_peak_working_set_and_not_zero():
    proc = _Process(_Info(rss=91 * _MB, peak_wset=419 * _MB))
    peak, cur = rss_mb(platform="win32", process=proc, usage=None)
    assert peak == 419.0, "Windows must report PeakWorkingSetSize, not the 0 it used to print"
    assert cur == 91.0


def test_the_windows_peak_does_not_come_from_the_posix_resource_module():
    """Hand it a `usage` that would answer 1 MB: the Windows branch must ignore it entirely."""
    proc = _Process(_Info(rss=91 * _MB, peak_wset=419 * _MB))
    peak, _ = rss_mb(platform="win32", process=proc, usage=_Usage(ru_maxrss=1024))
    assert peak == 419.0


def test_a_windows_peak_of_zero_is_treated_as_no_answer_not_as_zero_bytes():
    """`peak_wset == 0` from a fresh/restricted process is absence of a measurement, so fall
    through to the rss rather than printing zero."""
    proc = _Process(_Info(rss=91 * _MB, peak_wset=0))
    peak, cur = rss_mb(platform="win32", process=proc, usage=None)
    assert peak == 91.0
    assert cur == 91.0


def test_the_linux_peak_is_ru_maxrss_in_kilobytes():
    peak, cur = rss_mb(platform="linux", process=None, usage=_Usage(ru_maxrss=419 * 1024))
    assert peak == 419.0
    assert cur is None


def test_the_darwin_peak_is_ru_maxrss_in_bytes():
    peak, _ = rss_mb(platform="darwin", process=None, usage=_Usage(ru_maxrss=419 * _MB))
    assert peak == 419.0


def test_posix_prefers_the_os_high_water_mark_over_the_current_rss():
    proc = _Process(_Info(rss=91 * _MB))
    peak, cur = rss_mb(platform="darwin", process=proc, usage=_Usage(ru_maxrss=419 * _MB))
    assert peak == 419.0
    assert cur == 91.0


def test_with_no_peak_source_the_current_rss_is_the_peak_floor():
    proc = _Process(_Info(rss=91 * _MB))
    peak, cur = rss_mb(platform="win32", process=proc, usage=None)
    assert peak == 91.0
    assert cur == 91.0


def test_nothing_measurable_reports_zero_and_no_current():
    peak, cur = rss_mb(platform="win32", process=None, usage=None)
    assert peak == 0.0
    assert cur is None


def test_a_process_that_raises_is_not_allowed_to_break_the_footprint_line():
    """This runs in an excepthook, i.e. while the app is already dying; it must never raise."""

    class _Broken:
        def memory_info(self):
            raise OSError("access denied")

    peak, cur = rss_mb(platform="win32", process=_Broken(), usage=None)
    assert peak == 0.0
    assert cur is None


def test_the_live_process_reports_a_nonzero_peak_here():
    peak, cur = rss_mb()
    assert peak > 0.0
    assert cur is None or cur > 0.0


def test_the_viewer_entry_point_still_returns_a_pair():
    from squidxplorer._viewer import _rss_mb

    peak, cur = _rss_mb()
    assert peak > 0.0
