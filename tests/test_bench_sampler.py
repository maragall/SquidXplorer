"""External resource measurement: process-tree RSS, disk accounting, the watchdog.

The watchdog test is the one that matters. IMA-233's acceptance criterion is "produces
a table without exhausting disk", so "free space crossed the mark -> subprocess died ->
row says DISK_ABORT" has to be demonstrated, not asserted in a docstring.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from bench.sampler import (
    ResourceSampler,
    dir_bytes,
    free_bytes,
    kill_tree,
    preflight_disk,
    sample_tree_rss_mb,
    wait_for,
)


# --------------------------------------------------------------------------- disk


def test_dir_bytes_sums_a_tree(tmp_path):
    (tmp_path / "a").write_bytes(b"x" * 100)
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b").write_bytes(b"y" * 250)
    assert dir_bytes(tmp_path) == 350


def test_dir_bytes_on_missing_path_is_zero(tmp_path):
    assert dir_bytes(tmp_path / "nope") == 0


def test_dir_bytes_on_a_single_file(tmp_path):
    f = tmp_path / "f"
    f.write_bytes(b"z" * 42)
    assert dir_bytes(f) == 42


def test_dir_bytes_ignores_symlinks(tmp_path):
    """Following links would double-count, or count bytes on another filesystem."""
    real = tmp_path / "real"
    real.write_bytes(b"q" * 64)
    (tmp_path / "link").symlink_to(real)
    assert dir_bytes(tmp_path) == 64


def test_free_bytes_reports_a_positive_number(tmp_path):
    assert free_bytes(tmp_path) > 0


def test_free_bytes_walks_up_to_an_existing_ancestor(tmp_path):
    assert free_bytes(tmp_path / "does" / "not" / "exist") > 0


def test_preflight_passes_with_a_trivial_requirement(tmp_path):
    ok, free = preflight_disk(tmp_path, 1)
    assert ok and free > 0


def test_preflight_fails_when_asking_for_an_absurd_amount(tmp_path):
    ok, _ = preflight_disk(tmp_path, 1 << 62)
    assert not ok


# ---------------------------------------------------------------------- proc tree


def test_sample_tree_rss_sees_this_process():
    assert sample_tree_rss_mb(os.getpid()) > 0


def test_sample_tree_rss_of_a_dead_pid_is_zero():
    assert sample_tree_rss_mb(999_999_999) == 0.0


def test_sample_tree_rss_includes_a_child():
    """A stitcher that forks workers really is using the total, so RSS must sum."""
    proc = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(5)"])
    try:
        wait_for(lambda: sample_tree_rss_mb(proc.pid) > 0, timeout=5)
        assert sample_tree_rss_mb(os.getpid()) >= sample_tree_rss_mb(proc.pid)
    finally:
        kill_tree(proc.pid)
        proc.wait(timeout=10)


def test_kill_tree_kills_the_child():
    proc = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"])
    kill_tree(proc.pid)
    proc.wait(timeout=10)
    assert proc.returncode != 0


def test_kill_tree_on_a_dead_pid_is_harmless():
    kill_tree(999_999_999)  # must not raise


# ------------------------------------------------------------------------ sampler


def test_sampler_records_peak_rss_of_a_live_subprocess(tmp_path):
    proc = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(3)"])
    try:
        with ResourceSampler(proc.pid, tmp_path, interval_s=0.05) as s:
            wait_for(lambda: s.result.samples > 2, timeout=5)
        assert s.result.peak_rss_mb > 0
        assert s.result.samples > 0
        assert not s.result.disk_abort
    finally:
        kill_tree(proc.pid)
        proc.wait(timeout=10)


def test_sampler_tracks_output_growth(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    proc = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(2)"])
    try:
        with ResourceSampler(proc.pid, out, interval_s=0.05) as s:
            (out / "blob").write_bytes(b"0" * 5000)
            wait_for(lambda: s.result.peak_output_bytes >= 5000, timeout=5)
        assert s.result.peak_output_bytes >= 5000
    finally:
        kill_tree(proc.pid)
        proc.wait(timeout=10)


def test_watchdog_aborts_and_kills_when_free_space_crosses_the_mark(tmp_path):
    """The acceptance criterion, demonstrated: an impossible low-water mark fires the
    abort, the callback kills the process, and the flag is recorded."""
    proc = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"])
    killed = {"yes": False}

    def on_abort():
        killed["yes"] = True
        kill_tree(proc.pid)

    try:
        sampler = ResourceSampler(
            proc.pid,
            tmp_path,
            min_free_bytes=1 << 62,  # nothing can satisfy this
            interval_s=0.05,
            on_abort=on_abort,
        )
        sampler.start()
        assert wait_for(lambda: sampler.result.disk_abort, timeout=10)
        res = sampler.stop()
        assert res.disk_abort
        assert killed["yes"]
        proc.wait(timeout=10)
        assert proc.returncode != 0
    finally:
        kill_tree(proc.pid)


def test_watchdog_does_not_fire_when_there_is_room(tmp_path):
    proc = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(2)"])
    try:
        with ResourceSampler(proc.pid, tmp_path, min_free_bytes=1, interval_s=0.05) as s:
            wait_for(lambda: s.result.samples > 2, timeout=5)
        assert not s.result.disk_abort
    finally:
        kill_tree(proc.pid)
        proc.wait(timeout=10)


def test_abort_callback_that_raises_does_not_break_the_sampler(tmp_path):
    proc = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(5)"])

    def boom():
        raise RuntimeError("callback exploded")

    try:
        sampler = ResourceSampler(
            proc.pid, tmp_path, min_free_bytes=1 << 62, interval_s=0.05, on_abort=boom
        )
        sampler.start()
        assert wait_for(lambda: sampler.result.disk_abort, timeout=10)
        sampler.stop()  # must not propagate
    finally:
        kill_tree(proc.pid)
        proc.wait(timeout=10)


def test_sampler_without_a_watch_dir_only_measures_rss():
    proc = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(2)"])
    try:
        with ResourceSampler(proc.pid, None, interval_s=0.05) as s:
            wait_for(lambda: s.result.samples > 1, timeout=5)
        assert s.result.peak_output_bytes == 0
        assert not s.result.disk_abort
    finally:
        kill_tree(proc.pid)
        proc.wait(timeout=10)


def test_wait_for_times_out_and_returns_false():
    t0 = time.monotonic()
    assert wait_for(lambda: False, timeout=0.2, interval=0.05) is False
    assert time.monotonic() - t0 >= 0.2
