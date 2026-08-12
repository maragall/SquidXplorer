"""The peak footprint has to be a real number on Windows, which is where v1 ships.

2026-07-29, Task 9. ``_rss_mb()`` took its peak from ONE source, ``resource.getrusage``, and
``resource`` is a POSIX-only module. On Windows the import raises, the ``except Exception: pass``
swallows it, and ``peak`` keeps its initialiser. So the footprint line printed ``peak 0 MB`` at
every poll, at app quit, at ``atexit`` and in the excepthook after a CRASH, for the life of the
product, on the only platform anybody runs it on. A measurement that is always zero is worse than
an absent one: it reads as "this app uses no memory" rather than as "nobody measured".

The peak is the number that matters here and it cannot be recovered afterwards. Spencer measured
91 MB to 419 MB opening a 9-well plate; a poll every 5 seconds can miss a transient fuse entirely,
which is exactly why the code went to the OS high-water mark instead of sampling. Windows keeps
that same high-water mark, as ``PeakWorkingSetSize``, and ``psutil`` (a declared dependency)
exposes it as ``memory_info().peak_wset``. So the fix is one more source, not a new mechanism.

WHAT IS PINNED HERE
  * on Windows the peak is ``peak_wset``, and it is not zero;
  * the Windows branch never depends on ``resource``, so a ``usage`` that would answer differently
    cannot influence it -- that ordering IS the bug, so it gets a test of its own;
  * POSIX is unchanged: ``ru_maxrss``, bytes on darwin and kilobytes on linux;
  * with no peak source at all the peak falls back to the CURRENT rss, which is a true lower bound,
    rather than to 0, which is a false statement;
  * only genuinely nothing measurable gives ``(0.0, None)``.

The platform and both measurement sources are injected, so every branch is exercised on the
machine running the suite. No skips: the Windows branch is tested ON macOS, on purpose, because a
branch that only runs on the platform we cannot test here is a branch nobody tests.

ONE CORRECTION TO THE BACKLOG ENTRY. It says the low-memory warning the README promises ("a memory
bar warns you before the system runs low") cannot fire on Windows. Measured: it can. That bar is
driven by ``_region_viewer._process_memory_fraction``, which tries psutil FIRST and only falls back
to ``resource``, so the warning path was always sound on Windows. What was broken is the footprint
PEAK line, and only that. Recorded here so the next person does not go hunting for a second bug.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from squidxplorer._footprint import rss_mb

_MB = 1024 * 1024


class _Info:
    """A stand-in for ``psutil.Process().memory_info()``. Windows builds carry ``peak_wset``;
    macOS and Linux builds do not have the field at all, which is why it has to be injected."""

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
    """A stand-in for ``resource.getrusage(RUSAGE_SELF)``."""

    def __init__(self, ru_maxrss: int) -> None:
        self.ru_maxrss = ru_maxrss


# --- the bug: Windows -----------------------------------------------------------------------------

def test_the_windows_peak_is_the_peak_working_set_and_not_zero():
    proc = _Process(_Info(rss=91 * _MB, peak_wset=419 * _MB))
    peak, cur = rss_mb(platform="win32", process=proc, usage=None)
    assert peak == 419.0, "Windows must report PeakWorkingSetSize, not the 0 it used to print"
    assert cur == 91.0


def test_the_windows_peak_does_not_come_from_the_posix_resource_module():
    """The ordering IS the bug. `resource` does not exist on Windows, so a peak that can only be
    sourced from it is a peak that is structurally unreachable there. Hand the call a `usage` that
    would answer 1 MB and check it is ignored: the Windows branch must not consult it at all."""
    proc = _Process(_Info(rss=91 * _MB, peak_wset=419 * _MB))
    peak, _ = rss_mb(platform="win32", process=proc, usage=_Usage(ru_maxrss=1024))
    assert peak == 419.0


def test_a_windows_peak_of_zero_is_treated_as_no_answer_not_as_zero_bytes():
    """A brand-new or restricted process can report ``peak_wset == 0``. That is the absence of a
    measurement, so it must not be printed as a measurement of zero: fall through to the rss."""
    proc = _Process(_Info(rss=91 * _MB, peak_wset=0))
    peak, cur = rss_mb(platform="win32", process=proc, usage=None)
    assert peak == 91.0
    assert cur == 91.0


# --- POSIX, unchanged ----------------------------------------------------------------------------

def test_the_linux_peak_is_ru_maxrss_in_kilobytes():
    peak, cur = rss_mb(platform="linux", process=None, usage=_Usage(ru_maxrss=419 * 1024))
    assert peak == 419.0
    assert cur is None


def test_the_darwin_peak_is_ru_maxrss_in_bytes():
    peak, _ = rss_mb(platform="darwin", process=None, usage=_Usage(ru_maxrss=419 * _MB))
    assert peak == 419.0


def test_posix_prefers_the_os_high_water_mark_over_the_current_rss():
    """``ru_maxrss`` is exact without sampling; the current rss is only a floor. When both are
    available the high-water mark wins, or a fuse that has already been freed goes unreported."""
    proc = _Process(_Info(rss=91 * _MB))
    peak, cur = rss_mb(platform="darwin", process=proc, usage=_Usage(ru_maxrss=419 * _MB))
    assert peak == 419.0
    assert cur == 91.0


# --- no source at all ----------------------------------------------------------------------------

def test_with_no_peak_source_the_current_rss_is_the_peak_floor():
    """Neither ``resource`` nor ``peak_wset``. The current rss is a TRUE lower bound on the peak,
    so reporting it is honest and reporting 0 is not."""
    proc = _Process(_Info(rss=91 * _MB))
    peak, cur = rss_mb(platform="win32", process=proc, usage=None)
    assert peak == 91.0
    assert cur == 91.0


def test_nothing_measurable_reports_zero_and_no_current():
    peak, cur = rss_mb(platform="win32", process=None, usage=None)
    assert peak == 0.0
    assert cur is None


def test_a_process_that_raises_is_not_allowed_to_break_the_footprint_line():
    """The footprint monitor runs in an excepthook, i.e. while the app is already dying. It must
    never be the thing that raises there."""

    class _Broken:
        def memory_info(self):
            raise OSError("access denied")

    peak, cur = rss_mb(platform="win32", process=_Broken(), usage=None)
    assert peak == 0.0
    assert cur is None


# --- the real process, on whatever platform is running the suite ---------------------------------

def test_the_live_process_reports_a_nonzero_peak_here():
    """Not a branch test: the end-to-end claim. This interpreter has a footprint, so both numbers
    must be real. It is the assertion that would have failed on Windows before this change."""
    peak, cur = rss_mb()
    assert peak > 0.0
    assert cur is None or cur > 0.0


def test_the_viewer_entry_point_still_returns_a_pair():
    """``_viewer._rss_mb`` is the name the footprint monitor calls; it delegates now, and the two
    call sites unpack a 2-tuple."""
    from squidxplorer._viewer import _rss_mb

    peak, cur = _rss_mb()
    assert peak > 0.0
