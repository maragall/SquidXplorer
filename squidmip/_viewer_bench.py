"""Odon vs ndviewer_light on ONE identical task — IMA-235.

HISTORICAL AS OF 2026-07-30: ndviewer_light was deleted from this product, so the second
side of this comparison can no longer be run. The file is kept for the METHODOLOGY below,
which is the part with lasting value: it states the wrong benchmark explicitly so it is not
accidentally rebuilt, and it defines one identical task before measuring anything. Do not
expect `tools/viewer_benchmark.py` to produce numbers any more; it will not find
ndviewer_light. Whether to keep or bin this is Julio's call, not a migration detail.

Why this module exists separately from ``_odon_bench.py``
=========================================================
``_odon_bench.py`` answers "local vs http-served, for Odon". It measures ONE tool.
This module answers a different and more dangerous question: "is Odon faster than the
viewer we already ship?" — dangerous because the obvious way to measure it is wrong.

The wrong benchmark, stated so it is not accidentally rebuilt
------------------------------------------------------------
Time ``odon --check`` (a native binary's IO probe) against ``import ndviewer_light``
(a Python package pulling in PyQt5), and conclude the native binary is ~20x faster.
That comparison is real arithmetic on unreal work: one side decodes a tile, the other
side builds a GUI toolkit. It would be true and useless. Everything below exists to
make both sides do the SAME work, and to refuse to print a number where they cannot.

THE TASK, defined before measuring
==================================
    T1 — "first tile of a local OME-Zarr image group"

    Starting from NOTHING (no warm process, no open handle), as a fresh OS process:
      1. resolve a local OME-Zarr multiscale image group,
      2. read its multiscales metadata,
      3. select the COARSEST pyramid level,
      4. decode exactly one channel-plane tile from it into memory,
      5. print a proof line naming the level and the decoded shape,
      6. exit.

This is not a task invented to flatter either side. It is literally what Odon's only
headless path already does, discovered by running it rather than by reading its docs::

    $ odon --check <field>
    OK: loaded tile level 4 path '4' shape [1, 4, 1, 130, 130] -> subset [1, 1, 1, 130, 130]

Coarsest level, one channel, one z, full 130x130 = exactly one chunk. ndviewer_light is
then asked for the same five numbers, through its OWN data path
(``core.open_zarr_tensorstore``), not through a hand-rolled zarr reader.

What each timer INCLUDES and EXCLUDES
=====================================
Both sides are timed the same way: wall clock around ``/usr/bin/time -l <argv>``, i.e.
fork+exec to exit, measured by the parent. Same instrument, same boundary, both sides.

  INCLUDED, both:  process spawn and dynamic linking; runtime/interpreter init; the
                   tool's own module/binary load; zarr group + array metadata parse;
                   chunk fetch; codec decode; the proof print; process teardown.

  EXCLUDED, both:  window/GPU context creation, texture upload, on-screen paint,
                   and any interaction after the first tile.

The exclusion is forced, not chosen: Odon has no headless render loop, so nothing past
"pixels are in RAM" is observable for it. Both sides therefore stop at the same place.
"first paint" in this module always means "first tile decoded into memory" and NEVER
"photons on a display". Anything named `paint` elsewhere in this repo means the same.

The import-cost objection, and why N-A is the fair comparand
------------------------------------------------------------
``ndviewer_light/core.py`` imports PyQt5 at module scope, and ``__init__`` imports
``core``, so touching its data path costs a Qt import even with no window. It is
tempting to call that unfair and strip it out.

It is not unfair. Odon's binary is one monolith with ``eframe``/OpenGL linked in; its
``--check`` path pays that binary's load cost too and simply does not RUN the GUI. So:

  N-A  imports ndviewer_light.core (Qt import paid, no window built)  <- PRIMARY, fair
  N-B  tensorstore only, replicating core's zarr3 spec, no ndviewer   <- DIAGNOSTIC

N-B is reported to ATTRIBUTE any gap (interpreter+Qt vs the data layer), never as
ndviewer_light's score: no user can obtain N-B's timing from the shipped viewer.

Cold vs warm
============
``purge`` requires root on this machine and sudo is not available to this harness, so
the OS page cache CANNOT be dropped. This is re-checked at runtime, not assumed, and
recorded in the report.

Consequently there is NO true cold-disk number here, and none is estimated. What is
reported is run 1 of a series vs runs 2..N, labelled exactly that way:
``first-run-in-series`` (dyld/interpreter caches possibly cold, PAGE CACHE WARM) and
``warm``. A previously observed 112.6 ms first / 7.1 ms warm split for odon is
consistent with that being a process-startup effect, not a disk effect.

Scaling: O(plate) or O(viewport)?
=================================
The 1536-well fixture is a RAW acquisition of symlinks (6144 links to one 8.6 MB TIFF,
50 GB logical). Converting it to a real plate is ~12 GB of pixels and is refused here:
it does not fit the disk budget and it is not needed to answer the question.

Instead ``build_symlink_plate`` synthesises a plate of N wells whose well groups are
real zarr.json files and whose image groups are SYMLINKS to one real written field —
the same trick sim_1536wp itself uses. Cost is ~200 bytes per well.

  This measures METADATA/DISCOVERY scaling, which IS the O(plate)-vs-O(viewport)
  question, and explicitly NOT pixel-IO scaling, since all wells share one field's
  bytes. Stated in the report every time so the number is never over-read.

Two scaling probes, kept apart because only one of them is a comparison:

  T3 (MATCHED)   T1 on a field that SITS INSIDE an N-well plate. Both tools run it.
                 If first-tile time is flat in N, the tool is O(viewport).
  T2 (ONE-SIDED) time to enumerate the whole plate. ndviewer_light does this in
                 ``discover_zarr_v3_fovs``. Odon HAS NO PLATE MODEL — it opens one
                 image group and never enumerates a plate; squidmip's samplesheet does
                 that flattening for it. So there is no Odon number to compare against
                 and none is fabricated; T2 is reported for ndviewer_light alone.

Licensing, which is a real cost and belongs next to any speed win
=================================================================
Odon is GPL-3. It can only be used at arm's length as a separate process (which is what
``squidmip._odon`` does). It cannot be imported, vendored, or linked into squidmip
without licensing the result GPL-3. Any "Odon is faster" conclusion carries that.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

# Odon's headless probe prints this on success, and ndviewer's probe is made to print the
# same shape of line. Presence of the marker — not a zero exit code — is what makes a run
# count: a process that exits 0 having decoded nothing has not done the task.
OK_MARKER = "OK: loaded tile"

_TIMEOUT_S = 180.0
_TIME_BIN = "/usr/bin/time"

NDV_ROOT = "/Users/julioamaragall/Cephla/projects/ndviewer_light"
PROFILING_ROOT = "/Users/julioamaragall/CEPHLA/projects/stitcher"


# --------------------------------------------------------------------------------------
# Probe subjects. Each is an argv that performs T1 and prints an OK_MARKER line.
# --------------------------------------------------------------------------------------

# N-A: through ndviewer_light's OWN data path. Qt import is paid and NOT hidden.
# StageTimer/RSSSampler come from Julio's profiling suite rather than a local reimplementation;
# if that suite is not importable the probe still runs and just omits the breakdown, because
# the stage split is a diagnostic and T1 is the measurement.
_NDV_FULL = r"""
import sys, json, time
sys.path.insert(0, {ndv!r}); sys.path.insert(0, {prof!r})
t0 = time.perf_counter()
try:
    from profiling.stages import StageTimer
    from profiling.sampler import RSSSampler
    timer, sampler = StageTimer(t0), RSSSampler(t0, interval_s=0.01); sampler.start()
except Exception:
    timer = sampler = None
from contextlib import contextmanager, nullcontext
stage = timer.stage if timer else (lambda _n: nullcontext())
from pathlib import Path
with stage("import"):
    from ndviewer_light.core import open_zarr_tensorstore
with stage("open"):
    store = open_zarr_tensorstore(Path({field!r}), {level!r})
    if store is None: raise SystemExit("open_zarr_tensorstore returned None")
with stage("read"):
    arr = store[0:1, 0:1, 0:1, :, :].read().result()
print("OK: loaded tile level {level} path '{level}' shape %s -> subset %s"
      % (list(store.shape), list(arr.shape)))
print("SUM %d" % int(arr.sum()))
if timer:
    peak = max((s.rss_mb for s in sampler.stop()), default=0.0)
    print("STAGES " + json.dumps({{"spans": timer.spans, "inproc_peak_rss_mb": peak}}))
"""

# N-B: the data layer alone, replicating core.open_zarr_tensorstore's zarr3 spec exactly.
# Diagnostic only — no user can get this timing out of the shipped viewer.
_NDV_TS_ONLY = r"""
import sys, time
import tensorstore as ts
store = ts.open({{"driver": "zarr3",
                 "kvstore": {{"driver": "file", "path": {full!r}}},
                 "recheck_cached_metadata": True}}, read=True).result()
arr = store[0:1, 0:1, 0:1, :, :].read().result()
print("OK: loaded tile level {level} path '{level}' shape %s -> subset %s"
      % (list(store.shape), list(arr.shape)))
print("SUM %d" % int(arr.sum()))
"""


def ndv_argv(field_dir, level: str, *, variant: str = "full") -> list:
    """argv for the ndviewer_light side of T1."""
    if variant == "full":
        src = _NDV_FULL.format(ndv=NDV_ROOT, prof=PROFILING_ROOT,
                               field=str(field_dir), level=level)
    else:
        src = _NDV_TS_ONLY.format(full=str(Path(field_dir) / level), level=level)
    return [sys.executable, "-c", src]


def odon_argv(binary: str, field_dir) -> list:
    return [binary, "--check", str(field_dir)]


# --------------------------------------------------------------------------------------
# One measured run
# --------------------------------------------------------------------------------------

# /usr/bin/time -l reports "<bytes>  maximum resident set size" on macOS. Using the OS's
# own accounting for BOTH tools is the only way to get a peak-RSS number that means the
# same thing on each side; psutil can only see processes we are inside.
_MAXRSS_RE = re.compile(r"^\s*(\d+)\s+maximum resident set size", re.MULTILINE)


@dataclass
class Run:
    wall_ms: float = 0.0
    peak_rss_mb: float = float("nan")
    rc: int = 0
    ok: bool = False
    output: str = ""

    @property
    def stages(self) -> dict:
        for line in self.output.splitlines():
            if line.startswith("STAGES "):
                try:
                    return json.loads(line[len("STAGES "):])
                except ValueError:
                    return {}
        return {}


def run_once(argv: Sequence[str], *, cwd: Optional[str] = None) -> Run:
    """Execute *argv* under /usr/bin/time -l; wall clock measured by this parent.

    A scratch cwd because odon writes ``odon.log`` next to wherever it is started, and a
    benchmark that litters the tree is one nobody runs twice. The same scratch cwd is
    given to the Python side so neither tool gets a cwd advantage.
    """
    made_cwd = cwd is None
    cwd = cwd or tempfile.mkdtemp(prefix="vbench-cwd-")
    full = [_TIME_BIN, "-l", *[str(a) for a in argv]]
    try:
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(full, capture_output=True, text=True,
                                  timeout=_TIMEOUT_S, cwd=cwd)
        except subprocess.TimeoutExpired:
            return Run(wall_ms=_TIMEOUT_S * 1000.0, rc=-1, output="TIMEOUT")
        wall = (time.perf_counter() - t0) * 1000.0
    finally:
        if made_cwd:
            shutil.rmtree(cwd, ignore_errors=True)

    out = ((proc.stdout or "") + (proc.stderr or "")).strip()
    m = _MAXRSS_RE.search(proc.stderr or "")
    return Run(wall_ms=wall,
               peak_rss_mb=(int(m.group(1)) / 1e6) if m else float("nan"),
               rc=proc.returncode, ok=proc.returncode == 0 and OK_MARKER in out,
               output=out)


@dataclass
class Series:
    """A tool's T1 repeated N times: run 1 kept apart from the warm remainder."""
    label: str = ""
    runs: list = field(default_factory=list)
    available: bool = True
    reason: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.runs) and all(r.ok for r in self.runs)

    @property
    def first_ms(self) -> float:
        return self.runs[0].wall_ms if self.runs else float("nan")

    @property
    def _warm(self) -> list:
        return [r.wall_ms for r in self.runs[1:]] or [r.wall_ms for r in self.runs]

    @property
    def warm_median_ms(self) -> float:
        return statistics.median(self._warm) if self._warm else float("nan")

    @property
    def warm_min_ms(self) -> float:
        return min(self._warm) if self._warm else float("nan")

    @property
    def warm_max_ms(self) -> float:
        return max(self._warm) if self._warm else float("nan")

    @property
    def warm_iqr_ms(self) -> float:
        """Spread as IQR. A median without a spread invites over-reading a 2 ms gap."""
        w = sorted(self._warm)
        if len(w) < 4:
            return float("nan")
        mid = len(w) // 2
        lo = statistics.median(w[:mid])
        hi = statistics.median(w[mid + (len(w) % 2):])
        return hi - lo

    @property
    def peak_rss_mb(self) -> float:
        vals = [r.peak_rss_mb for r in self.runs if r.peak_rss_mb == r.peak_rss_mb]
        return statistics.median(vals) if vals else float("nan")

    @property
    def proof(self) -> str:
        for r in self.runs:
            for line in r.output.splitlines():
                if OK_MARKER in line:
                    return line.strip()
        return ""


def measure(label: str, argv: Sequence[str], repeats: int) -> Series:
    s = Series(label=label)
    for _ in range(max(1, repeats)):
        r = run_once(argv)
        s.runs.append(r)
        if not r.ok:
            break   # a failing series must not be averaged into a plausible-looking number
    return s


# --------------------------------------------------------------------------------------
# Fixtures: a big plate for ~200 bytes a well
# --------------------------------------------------------------------------------------

def _well_ids(n: int) -> list:
    """(row_name, col_name) for the first *n* wells of a 32x48 (1536) layout."""
    rows = [chr(ord("A") + i) if i < 26 else "A" + chr(ord("A") + i - 26) for i in range(32)]
    out = []
    for r in rows:
        for c in range(1, 49):
            if len(out) >= n:
                return out
            out.append((r, str(c)))
    return out


def build_symlink_plate(real_field, out_dir, n_wells: int) -> Path:
    """A plate of *n_wells* whose image groups are symlinks to one real field.

    Real: the plate group, every well group, and the pyramid itself. Shared: the pixels.
    So this is a valid store for any reader that walks it, and it is a VALID instrument
    for discovery/metadata scaling and an INVALID one for pixel-IO scaling. Callers must
    carry that caveat into the report; :func:`format_report` does.
    """
    real_field = Path(real_field).resolve()
    plate = Path(out_dir) / "plate.ome.zarr"
    if plate.exists():
        shutil.rmtree(plate)
    plate.mkdir(parents=True)

    wells = _well_ids(n_wells)
    plate_rows = sorted({r for r, _ in wells}, key=lambda s: (len(s), s))
    plate_cols = sorted({c for _, c in wells}, key=int)
    (plate / "zarr.json").write_text(json.dumps({
        "zarr_format": 3, "node_type": "group",
        "attributes": {"ome": {"version": "0.5", "plate": {
            "name": "plate",
            "rows": [{"name": r} for r in plate_rows],
            "columns": [{"name": c} for c in plate_cols],
            "wells": [{"path": f"{r}/{c}",
                       "rowIndex": plate_rows.index(r),
                       "columnIndex": plate_cols.index(c)} for r, c in wells],
            "field_count": 1}}}}))

    well_json = json.dumps({"zarr_format": 3, "node_type": "group",
                            "attributes": {"ome": {"version": "0.5",
                                                   "well": {"images": [{"path": "0"}]}}}})
    for r, c in wells:
        wd = plate / r / c
        wd.mkdir(parents=True, exist_ok=True)
        (wd / "zarr.json").write_text(well_json)
        # Symlink the IMAGE GROUP, so each well presents a complete, openable pyramid.
        os.symlink(real_field, wd / "0")
    return plate


# --------------------------------------------------------------------------------------
# Environment facts, re-derived rather than trusted
# --------------------------------------------------------------------------------------

def can_drop_caches() -> dict:
    """Re-check, live, whether the page cache can be dropped without sudo."""
    if not Path("/usr/sbin/purge").exists():
        return {"possible": False, "reason": "/usr/sbin/purge not present"}
    try:
        p = subprocess.run(["/usr/sbin/purge"], capture_output=True, text=True, timeout=60)
    except Exception as exc:
        return {"possible": False, "reason": f"purge failed to execute: {exc}"}
    out = ((p.stdout or "") + (p.stderr or "")).strip()
    if "not permitted" in out.lower() or p.returncode != 0:
        return {"possible": False, "reason": f"purge refused: {out or 'rc=%d' % p.returncode}"}
    return {"possible": True, "reason": "purge ran"}


def odon_remote_refusal(odon_bin: str, url: str) -> dict:
    """Re-derive Odon's inability to open an http-served store, verbatim.

    Re-run every time instead of quoting a comment, so the finding cannot go stale
    against a future build that gains the capability.
    """
    r = run_once(odon_argv(odon_bin, url))
    return {"url": url, "rc": r.rc, "ok": r.ok, "output": r.output}


def env_facts(odon_bin: str) -> dict:
    try:
        import tensorstore  # noqa: F401
        ts_ver = getattr(tensorstore, "__version__", "unknown (module exposes none)")
    except Exception as exc:
        ts_ver = f"unimportable: {exc}"
    try:
        sys.path.insert(0, NDV_ROOT)
        from ndviewer_light import __version__ as ndv_ver
    except Exception as exc:
        ndv_ver = f"unimportable: {exc}"
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True).stdout.strip(),
        "python": sys.version.split()[0],
        "tensorstore": ts_ver,
        "ndviewer_light": ndv_ver,
        "odon_binary": odon_bin,
        # odon has no --version flag: every non-flag argv falls through to the GUI launch
        # path and blocks forever, so the version is the pinned/verified one, not probed.
        "odon_version": "0.1.5 (pinned; binary exposes no --version — see note)",
    }


# --------------------------------------------------------------------------------------
# T2: plate enumeration. ndviewer_light only, and the report says why.
# --------------------------------------------------------------------------------------

_NDV_DISCOVER = r"""
import sys, time
sys.path.insert(0, {ndv!r})
from pathlib import Path
t0 = time.perf_counter()
from ndviewer_light.core import discover_zarr_v3_fovs, detect_format
t_import = (time.perf_counter() - t0) * 1e3
base = Path({base!r})
t1 = time.perf_counter()
fmt = detect_format(base)
fovs, _ = discover_zarr_v3_fovs(base)
t_disc = (time.perf_counter() - t1) * 1e3
print("OK: loaded tile level - path '-' shape [] -> subset []")   # marker: probe completed
print("DISCOVER %s %d %.3f %.3f" % (fmt, len(fovs), t_import, t_disc))
"""


def ndv_discover_argv(base_dir) -> list:
    return [sys.executable, "-c", _NDV_DISCOVER.format(ndv=NDV_ROOT, base=str(base_dir))]


def parse_discover(run: "Run") -> dict:
    for line in run.output.splitlines():
        if line.startswith("DISCOVER "):
            _, fmt, n, ti, td = line.split()
            return {"format": fmt, "n_fovs": int(n),
                    "import_ms": float(ti), "discover_ms": float(td)}
    return {}


# --------------------------------------------------------------------------------------
# Whole run
# --------------------------------------------------------------------------------------

@dataclass
class ScalePoint:
    n_wells: int = 0
    plate: str = ""
    odon: Series = field(default_factory=Series)
    ndv: Series = field(default_factory=Series)
    discover: dict = field(default_factory=dict)


@dataclass
class Report:
    task: str = ""
    field_dir: str = ""
    level: str = ""
    env: dict = field(default_factory=dict)
    cache: dict = field(default_factory=dict)
    odon: Series = field(default_factory=Series)
    ndv_full: Series = field(default_factory=Series)
    ndv_ts: Series = field(default_factory=Series)
    scale: list = field(default_factory=list)
    steady: dict = field(default_factory=dict)
    remote: dict = field(default_factory=dict)
    not_measured: list = field(default_factory=list)


def coarsest_level(field_dir) -> str:
    """The level Odon's --check picks: the last entry of multiscales[0].datasets."""
    meta = json.loads((Path(field_dir) / "zarr.json").read_text())
    attrs = meta.get("attributes", {})
    ome = attrs.get("ome", attrs)
    return str(ome["multiscales"][0]["datasets"][-1]["path"])


def run(field_dir, *, odon_bin=None, repeats: int = 9, scale_wells=(1, 96, 1536),
        scratch=None, level=None) -> Report:
    from squidmip._odon import find_odon

    field_dir = Path(field_dir).resolve()
    level = level or coarsest_level(field_dir)
    rep = Report(field_dir=str(field_dir), level=level)
    rep.task = (
        "T1: as a fresh OS process, open a local OME-Zarr image group, parse its "
        f"multiscales, select the coarsest pyramid level ({level}), decode one "
        "channel-plane tile into memory, print a proof line, exit. Timed with "
        "/usr/bin/time -l on both sides. Excludes window creation, GPU upload and "
        "on-screen paint on BOTH sides (Odon has no headless render loop, so nothing "
        "past 'pixels in RAM' is observable for it).")

    try:
        binary = str(Path(odon_bin) if odon_bin else find_odon())
    except FileNotFoundError as exc:
        binary = ""
        rep.odon = Series(label="odon --check", available=False, reason=str(exc))
    rep.env = env_facts(binary or "NOT FOUND")
    rep.cache = can_drop_caches()

    if binary:
        rep.odon = measure("odon --check", odon_argv(binary, field_dir), repeats)
    rep.ndv_full = measure("ndviewer_light (core)",
                           ndv_argv(field_dir, level, variant="full"), repeats)
    rep.ndv_ts = measure("tensorstore only", ndv_argv(field_dir, level, variant="ts"),
                         repeats)

    # ---- scaling ----
    scratch = Path(scratch or tempfile.mkdtemp(prefix="vbench-scale-"))
    for n in scale_wells:
        d = scratch / f"plate{n}"
        d.mkdir(parents=True, exist_ok=True)
        plate = build_symlink_plate(field_dir, d, n)
        sp = ScalePoint(n_wells=n, plate=str(plate))
        inner = plate / "A" / "1" / "0"     # a field INSIDE the N-well plate
        if binary:
            sp.odon = measure(f"odon @{n}w", odon_argv(binary, inner), max(3, repeats // 3))
        sp.ndv = measure(f"ndv @{n}w", ndv_argv(inner, level, variant="full"),
                         max(3, repeats // 3))
        sp.discover = parse_discover(run_once(ndv_discover_argv(d)))
        rep.scale.append(sp)

    rep.steady = parse_steady(run_once(ndv_steady_argv(field_dir, level)))

    # ---- what did not run ----
    rep.not_measured = [
        "Framerate / sustained pan-zoom, EITHER tool: Odon exposes no headless render "
        "loop, no frame counter and no timing output, so there is no interface to read a "
        "framerate through. Comparing ndviewer_light's framerate against nothing would "
        "not be a comparison, so no framerate is reported for either side.",
        "True COLD-DISK first tile, either tool: the OS page cache could not be dropped "
        f"({rep.cache.get('reason')}). Run 1 of each series is labelled "
        "'first-run-in-series', which is process-cold but PAGE-CACHE-WARM. No cold-disk "
        "number is estimated.",
        "Pixel-IO scaling with plate size: the large plates share one field's bytes by "
        "symlink (a real 1536-well plate is ~12 GB and exceeds this machine's disk "
        "budget). The scaling rows therefore measure METADATA/DISCOVERY scaling only.",
        "Odon plate enumeration (T2): Odon has no HCS/plate model at all — it opens one "
        "image group and never enumerates a plate (squidmip's samplesheet does that "
        "flattening for it). There is no Odon operation to time here, so T2 is reported "
        "for ndviewer_light alone and NOT as a head-to-head.",
        "Odon steady-state per-tile cost (T4): `--check` decodes exactly one tile and "
        "exits, and Odon offers no headless way to ask an already-running process for a "
        "second tile. So the number that matters for INTERACTIVE use — cost per tile once "
        "the tool is up — exists for ndviewer_light and does not exist for Odon. T4 is "
        "reported one-sided and is never divided by an Odon number.",
    ]
    if binary:
        rep.remote = odon_remote_refusal(binary, "http://127.0.0.1:8899/plate.ome.zarr/A/1/0")
        rep.not_measured.append(
            "Odon on an http-served store: refused live during this run; its verbatim "
            "output is in the REMOTE section. Any remote comparison is therefore "
            "ndviewer_light-only and no head-to-head remote number exists.")
    return rep


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------

def _row(s: Series) -> str:
    if not s.available:
        return f"  {s.label:<24} NOT RUN — {s.reason}"
    if not s.ok:
        return f"  {s.label:<24} FAILED (rc={s.runs[-1].rc if s.runs else '?'}): " \
               f"{(s.runs[-1].output[:160] if s.runs else '')}"
    return (f"  {s.label:<24} {s.first_ms:>10.1f} {s.warm_median_ms:>10.1f} "
            f"{s.warm_min_ms:>8.1f} {s.warm_max_ms:>8.1f} {s.warm_iqr_ms:>8.1f} "
            f"{s.peak_rss_mb:>9.1f}  {len(s.runs)}")


_HEAD = (f"  {'tool':<24} {'first_ms':>10} {'warm_med':>10} {'min':>8} {'max':>8} "
         f"{'IQR':>8} {'peakRSS':>9}  n")


def format_report(r: Report) -> str:
    L = ["=" * 100, "ODON vs ndviewer_light — one identical task", "=" * 100, "",
         "TASK", "----"]
    L += ["  " + ln for ln in _wrap(r.task, 96)]
    L += ["", f"  field : {r.field_dir}", f"  level : {r.level} (coarsest; the one Odon "
          "--check picks, re-derived from multiscales rather than hardcoded)", ""]

    L += ["ENVIRONMENT", "-----------"]
    for k, v in r.env.items():
        L.append(f"  {k:<16} {v}")
    L.append(f"  {'page cache':<16} CANNOT be dropped — {r.cache.get('reason')}")
    L += ["", "T1 — FIRST TILE, LOCAL (all times ms; /usr/bin/time -l around a fresh "
          "process, both sides)", "-" * 100,
          "  'first' = run 1 of the series: process-cold but PAGE-CACHE-WARM. It is NOT a "
          "cold-disk number;", "  the page cache could not be dropped (see above). 'warm' "
          "= median of runs 2..n, with min/max/IQR.", "", _HEAD, "  " + "-" * (len(_HEAD) - 2)]
    L += [_row(r.odon), _row(r.ndv_full), _row(r.ndv_ts)]
    L += ["", "  proof lines (the run only counts if the tile was actually decoded):"]
    for s in (r.odon, r.ndv_full, r.ndv_ts):
        if s.proof:
            L.append(f"    {s.label:<24} {s.proof}")

    if r.odon.ok and r.ndv_full.ok:
        ratio = r.ndv_full.warm_median_ms / r.odon.warm_median_ms
        L += ["", f"  HEAD-TO-HEAD (warm median): Odon is {ratio:.1f}x faster than "
              f"ndviewer_light on T1", f"    ({r.odon.warm_median_ms:.1f} ms vs "
              f"{r.ndv_full.warm_median_ms:.1f} ms; absolute gap "
              f"{r.ndv_full.warm_median_ms - r.odon.warm_median_ms:.0f} ms)."]
        if r.ndv_ts.ok:
            data = r.ndv_ts.warm_median_ms
            L += [f"  ATTRIBUTION: the data layer alone (tensorstore, no Qt, no ndviewer "
                  f"import) is {data:.1f} ms,",
                  f"    so ~{r.ndv_full.warm_median_ms - data:.0f} ms of ndviewer_light's "
                  f"{r.ndv_full.warm_median_ms:.0f} ms is Python interpreter start + "
                  "importing core.py (which pulls PyQt5),",
                  "    and only the remainder is reading the tile. That row is a "
                  "DIAGNOSTIC, not ndviewer_light's score:",
                  "    no user can obtain it from the shipped viewer, whose data path "
                  "lives in the module that imports Qt."]

    # ---- scaling ----
    L += ["", "T3 — SAME FIRST TILE, FIELD SITTING INSIDE AN N-WELL PLATE (matched; ms)",
          "-" * 100,
          "  Large plates are symlink fixtures: real plate/well/multiscale metadata, "
          "SHARED pixels.",
          "  Valid for metadata/discovery scaling; NOT valid for pixel-IO scaling. A real "
          "1536-well plate is",
          "  ~12 GB and was refused on disk budget.", "",
          f"  {'wells':>6} {'odon warm_med':>15} {'ndviewer warm_med':>19}",
          "  " + "-" * 42]
    for sp in r.scale:
        o = f"{sp.odon.warm_median_ms:.1f}" if sp.odon.ok else "n/a"
        n = f"{sp.ndv.warm_median_ms:.1f}" if sp.ndv.ok else "n/a"
        L.append(f"  {sp.n_wells:>6} {o:>15} {n:>19}")
    L += ["", "  Flat in well count => O(viewport). Growing with well count => O(plate)."]

    L += ["", "T2 — WHOLE-PLATE ENUMERATION (ndviewer_light ONLY — see NOT MEASURED)",
          "-" * 100,
          f"  {'wells':>6} {'fovs found':>11} {'discover_ms':>13} {'import_ms':>11}",
          "  " + "-" * 45]
    for sp in r.scale:
        d = sp.discover
        if d:
            L.append(f"  {sp.n_wells:>6} {d['n_fovs']:>11} {d['discover_ms']:>13.1f} "
                     f"{d['import_ms']:>11.1f}")
        else:
            L.append(f"  {sp.n_wells:>6} {'FAILED':>11}")

    if r.steady:
        L += ["", "T4 — STEADY-STATE PER-TILE COST, PROCESS ALREADY UP "
              "(ndviewer_light ONLY — no Odon equivalent exists)", "-" * 100,
              "  T1 respawns the tool for every tile. That is how squidmip launches Odon, "
              "but it is NOT how",
              "  ndviewer_light is used: it is a window that imports once and then serves "
              "every pan/zoom/channel",
              "  toggle from a live process. Below, every read is a chunk NOT read before "
              "in that process, so",
              "  each is a real fetch+decode and not a tensorstore cache hit.", "",
              f"  tile at level {r.level} (the exact T1 tile)   median "
              f"{r.steady.get('tile_l4_median', float('nan')):6.2f} ms   "
              f"n={len(r.steady.get('tile_l4_ms', []))}",
              f"  full-res 1024x1024 level-0 chunk        median "
              f"{r.steady.get('chunk_l0_median', float('nan')):6.2f} ms   "
              f"n={len(r.steady.get('chunk_l0_ms', []))}",
              "",
              "  Read this NEXT TO T1, not instead of it. T1 says Odon starts far faster. "
              "T4 says that once",
              "  ndviewer_light is up, a tile costs it a fraction of a millisecond — while "
              "Odon pays its full",
              "  ~7 ms process spawn for every tile, because that is the only interface it "
              "offers. Which number",
              "  governs depends entirely on whether the viewer is launched per-tile or "
              "left open."]

    if r.remote:
        L += ["", "REMOTE — Odon against an http-served store (re-derived live this run)",
              "-" * 100, f"  $ odon --check {r.remote['url']}",
              f"  rc={r.remote['rc']}  ok={r.remote['ok']}"]
        for ln in r.remote["output"].splitlines()[:8]:
            L.append(f"  | {ln}")

    L += ["", "NOT MEASURED (stated, never estimated)", "-" * 100]
    for item in r.not_measured:
        L.append("  * " + "\n    ".join(_wrap(item, 94)))

    L += ["", "LICENSING", "-" * 100,
          "  Odon is GPL-3. It is usable only at arm's length as a separate process "
          "(squidmip._odon already",
          "  does this). It cannot be imported, vendored or linked into squidmip without "
          "making squidmip GPL-3.",
          "  Any speed win below must be read together with that adoption cost.", ""]
    return "\n".join(L)


def _wrap(text: str, width: int) -> list:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def write_json(r: Report, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def ser(s: Series) -> dict:
        return {"label": s.label, "available": s.available, "reason": s.reason,
                "ok": s.ok, "first_ms": s.first_ms, "warm_median_ms": s.warm_median_ms,
                "warm_min_ms": s.warm_min_ms, "warm_max_ms": s.warm_max_ms,
                "warm_iqr_ms": s.warm_iqr_ms, "peak_rss_mb": s.peak_rss_mb,
                "proof": s.proof, "all_wall_ms": [x.wall_ms for x in s.runs],
                "stages": s.runs[-1].stages if s.runs else {}}

    path.write_text(json.dumps({
        "task": r.task, "field": r.field_dir, "level": r.level, "env": r.env,
        "page_cache": r.cache, "odon": ser(r.odon), "ndviewer_light": ser(r.ndv_full),
        "tensorstore_only": ser(r.ndv_ts),
        "scale": [{"n_wells": sp.n_wells, "odon": ser(sp.odon), "ndviewer": ser(sp.ndv),
                   "discover": sp.discover} for sp in r.scale],
        "steady_state_ndviewer_only": r.steady,
        "remote": r.remote, "not_measured": r.not_measured,
    }, indent=2, default=str))
    return path


# --------------------------------------------------------------------------------------
# T4: steady state. ONE-SIDED, and the most important caveat on T1.
# --------------------------------------------------------------------------------------
#
# T1 respawns the tool for every tile, which structurally rewards a native binary: Odon's
# 7 ms is almost entirely process spawn, and it pays that again for every single tile.
# But ndviewer_light is not used that way. It is a WINDOW: the interpreter and Qt are
# imported once at launch, and every pan, zoom and channel toggle thereafter is a tile
# read inside a process that is already up.
#
# So T1 alone would answer "which tool is faster to START", and quietly let that be read
# as "which tool is faster to USE". T4 measures the second question for ndviewer_light.
#
# There is NO Odon number here and none is invented: `--check` is one-shot, decodes
# exactly one tile and exits, and Odon offers no headless way to ask a running process
# for a second tile. T4 is therefore reported as a ONE-SIDED measurement, never as a
# ratio against Odon.
#
# Cache honesty: re-reading the SAME tile would measure tensorstore's cache. Every read
# below is a chunk that has not been read before in that process, so each is a real
# fetch+decode.

_NDV_STEADY = r"""
import sys, time, json, statistics
sys.path.insert(0, {ndv!r})
from pathlib import Path
from ndviewer_light.core import open_zarr_tensorstore
F = Path({field!r})
s4 = open_zarr_tensorstore(F, {level!r})
nc = s4.shape[1]
t4 = []
for c in range(nc):                      # each channel plane: never read before
    t = time.perf_counter(); s4[0:1, c:c+1, 0:1, :, :].read().result()
    t4.append((time.perf_counter() - t) * 1e3)
s0 = open_zarr_tensorstore(F, "0")
H, W = s0.shape[3], s0.shape[4]
t0 = []
for c in range(nc):                      # each full-res chunk: never read before
    for y in range(0, H, 1024):
        for x in range(0, W, 1024):
            t = time.perf_counter()
            s0[0:1, c:c+1, 0:1, y:min(y+1024, H), x:min(x+1024, W)].read().result()
            t0.append((time.perf_counter() - t) * 1e3)
print("OK: loaded tile level {level} path '{level}' shape %s -> subset []" % list(s4.shape))
print("STEADY " + json.dumps({{"tile_l4_ms": t4, "chunk_l0_ms": t0,
                              "tile_l4_median": statistics.median(t4),
                              "chunk_l0_median": statistics.median(t0)}}))
"""


def ndv_steady_argv(field_dir, level: str) -> list:
    return [sys.executable, "-c",
            _NDV_STEADY.format(ndv=NDV_ROOT, field=str(field_dir), level=level)]


def parse_steady(run: "Run") -> dict:
    for line in run.output.splitlines():
        if line.startswith("STEADY "):
            try:
                return json.loads(line[len("STEADY "):])
            except ValueError:
                return {}
    return {}
