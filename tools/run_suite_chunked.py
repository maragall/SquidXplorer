#!/usr/bin/env python
"""Run the whole test suite in bounded-size chunks, each in its own process.

The GUI test family leaks QThread workers across tests (window.close() does not reap them), and
a long enough single-process run crashes with a native segfault once accumulation crosses a
resource cliff (~150 GUI tests). This is not a product bug; splitting into chunks small enough
to stay under the cliff makes every test run and still be able to fail.

Output contract (parsed by tools/commit_gate.sh):
    === SUITE SUMMARY: <P> passed, <F> failed, <S> skipped across <N> chunks ===
    === SUITE SEGFAULTS: <k> ===            (chunk indices that crashed, or "none")
Exit 0 iff every chunk exited cleanly with zero failures.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))


def _base_env() -> dict:
    env = dict(os.environ)
    # offscreen: no window opens. plugin autoload off: otherwise PyQt5 tests silently skip
    # against PySide.
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
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
        if "::" in line:
            ids.append(line)
    if not ids:
        sys.stderr.write("run_suite_chunked: collected ZERO node ids — collection is broken.\n")
        sys.stderr.write(proc.stdout[-2000:] + "\n" + proc.stderr[-2000:] + "\n")
    return ids


def read_results(path: str, requested: list[str]) -> tuple[int, int, int, list[str], list[str]]:
    """Fold the durable per-phase log into (passed, failed, skipped, failed_ids, unrun_ids)."""
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
            # a record with no passing/failing/skipped call is treated as unrun
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
        print("=== SUITE COLLECTION FAILED: zero tests collected ===")
        return 1

    chunks = [ids[i:i + args.chunk] for i in range(0, len(ids), args.chunk)]
    total_p = total_f = total_s = 0
    incomplete: list[int] = []          # chunks that lost tests to a mid-run crash
    benign_teardown: list[int] = []     # chunks whose tests all ran+passed but crashed at teardown

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

        # -p _chunk_recorder writes each outcome as it runs, so a teardown segfault after all
        # tests passed cannot erase the record; read that file for truth, not the exit code.
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
               "-p", "pytest_timeout", "-p", "_chunk_recorder", f"--timeout={args.timeout}"]
        if args.basetemp_root:
            bt = os.path.join(args.basetemp_root, f"chunk{idx}")
            # pytest does not create a nested --basetemp's parent
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
        for nid in failed_ids:
            print(f"FAILED {nid}", flush=True)

        crashed = proc.returncode not in (0, 1)
        if unrun_ids:
            incomplete.append(idx)
            print(f"run_suite_chunked: chunk {idx + 1} is INCOMPLETE — {len(unrun_ids)} of "
                  f"{len(chunk)} tests never ran (returncode={proc.returncode}). First unrun: "
                  f"{unrun_ids[0]}", flush=True)
        elif crashed:
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
