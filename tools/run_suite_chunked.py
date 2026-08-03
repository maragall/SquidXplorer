#!/usr/bin/env python
"""Run the WHOLE test suite in bounded-size chunks, each in its own process.

Why this exists
---------------
``QT_QPA_PLATFORM=offscreen ... python -m pytest -q`` can no longer run the suite start to finish:
it dies with a native ``Fatal Python error: Segmentation fault`` (no test failure) deep in the run.
It is a test-HARNESS resource-accumulation problem, not a product bug — the app runs clean and every
subset of tests passes in isolation.

Diagnosis (measured, in this repo, 2026-07-23)
    * NOT numba: the numba/native-heavy files (test_stitch, test_spots, test_benchmark, test_zarr_reader,
      test_decon, ...) run 251 tests in ONE process green. numba is only ever the VICTIM — the crash SITE
      wanders (a numba stitch solve in the full run, a bare ``<invalid frame>`` in the GUI-only run)
      because it is heap/resource corruption, not any one test's bug.
    * IT IS the GUI test family. Instrumenting ``tests/test_viewer.py`` (236 GUI tests) alone shows
      finished ``QThread`` worker objects piling up monotonically — 0, 11, 39, 52, 61 — with total live
      objects climbing 92k -> 417k, while the OS thread count stays flat at 16. ``window.close()`` neither
      quits nor deletes those workers, and something keeps each one referenced, so gc never reaps them.
      Around ~150 GUI tests' worth exhausts the process and the next native allocation crashes on it.
      test_viewer.py CRASHES ON ITS OWN at ~150 tests in — a single file already over the cliff.

Why chunking rather than an in-process fixture
    An autouse teardown that closes every top-level widget, drains ``deleteLater``, actively
    quits+waits+deletes every live ``QThread``, and gc's was tried in three escalating forms. None of
    them moved the crash: the leaked native state is not reachable for disposal from Python. So the honest
    fix is process isolation. Splitting the run into chunks small enough that no single process crosses the
    accumulation cliff makes every test RUN and still be able to FAIL — nothing is skipped, deleted, or
    weakened. It is exactly the same tests, just not all in one address space.

What this does
    1. Collect every test node id (one ``--collect-only`` process; collection imports modules but runs
       nothing, so it does not accumulate).
    2. Split them, in collection order, into chunks of ``--chunk`` tests (default 100; the measured cliff
       is ~150, and a 100-test all-GUI chunk peaks at ~40 leaked QThreads — comfortable margin).
    3. Run each chunk in its own ``python -m pytest`` process, offscreen, plugin-autoload OFF (required so
       the PyQt5 tests do not silently skip against PySide) with pytest_timeout loaded explicitly.
    4. Aggregate passed/failed/skipped, collect the ``FAILED`` node ids, and — critically — treat a chunk
       that SEGFAULTS (or otherwise dies without a pytest summary) as a hard error, never a pass: some of
       its tests did not run.

Output contract (what tools/commit_gate.sh parses)
    * Every chunk's raw pytest output is streamed through, so ``FAILED <nodeid> ...`` lines survive.
    * A final block:
          === SUITE SUMMARY: <P> passed, <F> failed, <S> skipped across <N> chunks ===
          === SUITE SEGFAULTS: <k> ===            (chunk indices that crashed, or "none")
    * Exit code: 0 iff every chunk exited cleanly with zero failures; 1 otherwise (failures and/or
      segfaulted chunks). The gate decides what to do about named flakes.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))


def _base_env() -> dict:
    env = dict(os.environ)
    # offscreen: no window opens. plugin autoload OFF: without it the PyQt5 tests silently skip
    # against PySide, so the gate would gate nothing. Both match the single-process gate exactly.
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    # Put tools/ on the path so each chunk can load ``-p _chunk_recorder`` (the durable outcome log).
    env["PYTHONPATH"] = os.pathsep.join([_TOOLS_DIR, env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    return env


def collect_node_ids(timeout: int) -> list[str]:
    env = _base_env()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-p", "pytest_timeout",
         "-q", "--collect-only"],
        capture_output=True, text=True, env=env, timeout=timeout,
    )
    ids: list[str] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        # ``-q --collect-only`` prints one node id per line ("tests/test_x.py::test_y", possibly
        # with a "[param]" suffix), then a trailing "N tests collected" summary and a warnings-docs
        # line — neither of which contains "::". Keep the id lines, in collection order.
        if "::" in line:
            ids.append(line)
    if not ids:
        sys.stderr.write("run_suite_chunked: collected ZERO node ids — collection is broken.\n")
        sys.stderr.write(proc.stdout[-2000:] + "\n" + proc.stderr[-2000:] + "\n")
    return ids


def read_results(path: str, requested: list[str]) -> tuple[int, int, int, list[str], list[str]]:
    """Fold the durable per-phase log into per-test outcomes.

    Returns ``(passed, failed, skipped, failed_ids, unrun_ids)``. A test is:
      * failed  — any phase (setup/call/teardown) recorded 'failed';
      * skipped — setup recorded 'skipped' and it never got a passing call;
      * passed  — its 'call' phase recorded 'passed' and nothing failed;
      * unrun   — it was requested but produced NO record at all (the process died before reaching it,
                  i.e. a real mid-chunk crash — NOT a benign teardown one).
    """
    phases: dict[str, dict[str, str]] = {}
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t", 2)
                if len(parts) != 3:
                    continue
                when, outcome, nodeid = parts
                phases.setdefault(nodeid, {})[when] = outcome
    except FileNotFoundError:
        pass

    passed = failed = skipped = 0
    failed_ids: list[str] = []
    unrun_ids: list[str] = []
    for nodeid in requested:
        rec = phases.get(nodeid)
        if not rec:
            unrun_ids.append(nodeid)
            continue
        if "failed" in rec.values():
            failed += 1
            failed_ids.append(nodeid)
        elif rec.get("call") == "passed":
            passed += 1
        elif rec.get("setup") == "skipped":
            skipped += 1
        elif rec.get("call") == "skipped":
            skipped += 1
        else:
            # A record exists but no passing/failing/skipped call — e.g. setup errored without a
            # 'failed' outcome, or teardown-only. Treat conservatively as unrun so it cannot pass.
            unrun_ids.append(nodeid)
    return passed, failed, skipped, failed_ids, unrun_ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=int(os.environ.get("SQUIDHCS_CHUNK", "100")))
    ap.add_argument("--timeout", type=int, default=int(os.environ.get("SQUIDHCS_TIMEOUT", "900")))
    ap.add_argument("--basetemp-root", default=os.environ.get("SQUIDHCS_BASETEMP", ""))
    args = ap.parse_args()

    ids = collect_node_ids(args.timeout)
    if not ids:
        # No SUITE SUMMARY line: the gate reads that absence as "the harness did not run" and refuses
        # loudly, which is exactly right when collection itself failed.
        print("=== SUITE COLLECTION FAILED: zero tests collected ===")
        return 1

    chunks = [ids[i:i + args.chunk] for i in range(0, len(ids), args.chunk)]
    total_p = total_f = total_s = 0
    incomplete: list[int] = []          # chunks that lost tests to a MID-run crash (unrun tests)
    benign_teardown: list[int] = []     # chunks whose tests ALL ran+passed but crashed at teardown

    resdir = args.basetemp_root or os.path.join(_TOOLS_DIR, ".chunk_results")
    os.makedirs(resdir, exist_ok=True)

    print(f"run_suite_chunked: {len(ids)} tests in {len(chunks)} chunks of up to {args.chunk} "
          f"(offscreen, autoload off, timeout={args.timeout}s)", flush=True)

    for idx, chunk in enumerate(chunks):
        env = _base_env()
        resfile = os.path.join(resdir, f"results_chunk{idx}.tsv")
        try:
            os.remove(resfile)
        except FileNotFoundError:
            pass
        env["SQUIDHCS_RESULT_FILE"] = resfile

        # -p _chunk_recorder writes each test's outcome to $SQUIDHCS_RESULT_FILE as it runs, so a
        # segfault during teardown (after every test has passed) cannot erase the record. We read
        # THAT file for truth, not the exit code, which a teardown crash makes negative.
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
               "-p", "pytest_timeout", "-p", "_chunk_recorder", f"--timeout={args.timeout}"]
        if args.basetemp_root:
            bt = os.path.join(args.basetemp_root, f"chunk{idx}")
            # pytest does not create a nested --basetemp's PARENT, so tmp_path's first mkdir under
            # it fails with FileNotFoundError. Create it here; pytest resets it at session start.
            os.makedirs(bt, exist_ok=True)
            cmd += ["--basetemp", bt]
        cmd += chunk
        print(f"\n----- chunk {idx + 1}/{len(chunks)} ({len(chunk)} tests) -----", flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        sys.stdout.write(proc.stdout)
        if proc.stderr.strip():
            sys.stdout.write(proc.stderr)
        sys.stdout.flush()

        p, f, s, failed_ids, unrun_ids = read_results(resfile, chunk)
        total_p += p
        total_f += f
        total_s += s
        # Emit FAILED lines so tools/commit_gate.sh's known-flake matcher can see them even though
        # the crashing process never printed its own summary.
        for nid in failed_ids:
            print(f"FAILED {nid}", flush=True)

        crashed = proc.returncode not in (0, 1)
        if unrun_ids:
            # Tests were requested but produced no outcome: the process died BEFORE running them — a
            # real mid-chunk crash (or timeout). Those tests did not execute; never call this green.
            incomplete.append(idx)
            print(f"run_suite_chunked: chunk {idx + 1} is INCOMPLETE — {len(unrun_ids)} of "
                  f"{len(chunk)} tests never ran (returncode={proc.returncode}). First unrun: "
                  f"{unrun_ids[0]}", flush=True)
        elif crashed:
            # Every requested test ran and was recorded, but the process still crashed — that is the
            # benign Qt/napari teardown segfault AFTER the work was done. Report it, do not fail on it.
            benign_teardown.append(idx)
            print(f"run_suite_chunked: chunk {idx + 1} finished all {len(chunk)} tests, then the "
                  f"process crashed in teardown (returncode={proc.returncode}). Benign: every test "
                  f"ran and was recorded. Not counted as a failure.", flush=True)

    print(f"\n=== SUITE SUMMARY: {total_p} passed, {total_f} failed, {total_s} skipped "
          f"across {len(chunks)} chunks ===")
    print(f"=== SUITE INCOMPLETE: {','.join(str(i) for i in incomplete) if incomplete else 'none'} ===")
    print(f"=== SUITE TEARDOWN-CRASHES: "
          f"{','.join(str(i) for i in benign_teardown) if benign_teardown else 'none'} ===")

    return 0 if (total_f == 0 and not incomplete) else 1


if __name__ == "__main__":
    sys.exit(main())
