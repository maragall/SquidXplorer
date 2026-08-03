#!/bin/sh
# Refuse a commit whose tests do not pass.
#
# Why this exists. Agents were told to commit before testing during a shutdown scare, so that
# in-flight work would survive. That was right at the time and then never unwound, and red
# commits quietly became normal - including one that landed with failing tile-composite tests
# and said so only in its report.
#
# The rule now:
#   * a branch ending in -wip may commit red. It can never merge; it is a life raft, nothing else.
#   * every other branch must be green, or the commit is refused.
#   * SQUIDHCS_STOP_ORDER=1 allows one red commit for a genuine stop order (machine going down).
#     It must be set deliberately, and the commit message must say why.
#
# Linked worktrees share $GIT_DIR/hooks with the main checkout, so installing this once covers
# every agent worktree.

set -e

BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "?")

case "$BRANCH" in
  *-wip)
    echo "commit-gate: '$BRANCH' is a -wip branch, allowing a red commit."
    echo "             it must never merge. re-land the work green on a real branch."
    exit 0
    ;;
esac

if [ "${SQUIDHCS_STOP_ORDER:-0}" = "1" ]; then
  echo "commit-gate: SQUIDHCS_STOP_ORDER=1, allowing one red commit."
  echo "             say why in the commit message, and re-land green."
  exit 0
fi

ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"

# WHY NOT ONE `pytest -q`. A single-process run of the whole suite no longer finishes: it dies with
# a native `Fatal Python error: Segmentation fault` (no test failure) deep in the run. It is a
# test-HARNESS accumulation bug, not a product bug - the app runs clean and every subset passes in
# isolation. Measured cause: the GUI test family leaks finished QThread workers (and native Qt/GL
# state) that nothing reaps; ~150 GUI tests' worth exhausts the process and the next native
# allocation (a numba stitch solve, or Qt teardown) crashes on it. In-process cleanup was tried in
# three escalating forms (close all widgets, drain deleteLater, quit+wait+delete every live
# QThread) and none of it moved the crash - the leaked state is not reachable from Python. So the
# suite is run in bounded-size CHUNKS, each in its own process, by tools/run_suite_chunked.py.
# Every test still runs and can still fail; they are just not all in one address space. See that
# file for the full diagnosis.
#
# PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 (set inside the runner) is required or the PyQt5 tests silently
# skip against PySide. pytest_timeout is loaded explicitly there. No -x anywhere: the whole suite
# runs and every failure is re-checked, so a real `assert False` can never hide behind a known flake.
#
# The runner distinguishes two kinds of crash. A benign TEARDOWN crash (every test in the chunk ran
# and was recorded to a durable per-test log, then the process died on the way out) is reported and
# tolerated - nothing was lost. An INCOMPLETE chunk (the process died mid-chunk, so some tests never
# ran) is a HARD refusal below: never a pass, never a "flake". If chunks go INCOMPLETE, the mid-run
# accumulation cliff moved; lower SQUIDHCS_CHUNK (default 100) rather than waving it through.
echo "commit-gate: running the suite in chunks before allowing this commit ..."
GATE_BASETEMP="${TMPDIR:-/tmp}/squidhcs_gate_bt.$$"
if ! SQUIDHCS_BASETEMP="$GATE_BASETEMP" \
     python tools/run_suite_chunked.py --chunk "${SQUIDHCS_CHUNK:-100}" --timeout 900 \
     >/tmp/squidhcs_gate.$$ 2>&1; then
  rm -rf "$GATE_BASETEMP" 2>/dev/null || true
  # If the runner never produced a suite summary, the harness is broken, not the tree. Refuse loudly.
  if ! grep -qE "^=== SUITE SUMMARY:" /tmp/squidhcs_gate.$$; then
    echo ""
    tail -20 /tmp/squidhcs_gate.$$
    rm -f /tmp/squidhcs_gate.$$
    echo ""
    echo "commit-gate: REFUSED - the chunked runner produced no SUITE SUMMARY (it did not run)."
    echo "  the gate is broken, not the tree. fix the gate before committing."
    exit 1
  fi
  # An INCOMPLETE chunk means tests that never ran (the process died BEFORE reaching them - a real
  # mid-chunk crash or timeout). That is NOT a flake and NOT a pass: refuse immediately. A benign
  # TEARDOWN-CRASH (every test ran and was recorded, then the process crashed on the way out) is
  # reported by the runner but is NOT a refusal - the durable per-test record already accounted for
  # every test. Only INCOMPLETE gates the commit.
  if ! grep -qE "^=== SUITE INCOMPLETE: none ===" /tmp/squidhcs_gate.$$; then
    echo ""
    grep -E "^=== SUITE (SUMMARY|INCOMPLETE|TEARDOWN-CRASHES):|is INCOMPLETE" /tmp/squidhcs_gate.$$
    tail -20 /tmp/squidhcs_gate.$$
    rm -f /tmp/squidhcs_gate.$$
    echo ""
    echo "commit-gate: REFUSED - a chunk was INCOMPLETE, so some tests never ran."
    echo "  this is the accumulation cliff (a mid-chunk crash), not a flake. lower SQUIDHCS_CHUNK"
    echo "  and re-run; do NOT commit until the whole suite has actually executed."
    exit 1
  fi
  FAILED=$(grep -E "^FAILED " /tmp/squidhcs_gate.$$ | sed 's/^FAILED //; s/ .*//' || true)
  if [ -n "$FAILED" ]; then
    # ALWAYS say what failed. The previous version printed only "known flakes (IMA-258)" and then
    # deleted the log, so nobody could tell WHICH tests failed or whether they were the known set.
    echo "commit-gate: these tests failed in the full run:"
    echo "$FAILED" | sed 's/^/    /'

    # A flake is tolerated only if it is NAMED here. The previous version re-ran whatever failed
    # and, if the re-run passed, announced "known flakes (IMA-258)" -- a claim it never checked,
    # against a set that is defined NOWHERE in this repo. Worse, re-running a SUBSET is a strictly
    # WEAKER condition than the full run: any test that fails only under full-suite conditions
    # (ordering, shared state, resource pressure) passes the re-run and gets waved through as a
    # "flake". Real breakage of that shape was indistinguishable from a flake.
    #
    # This list was EMPTY on purpose. Exactly one entry has been earned since, WITH the evidence:
    #
    #   test_ima188_sim1536_scaling_measured_no_regression
    #     It asserts a THREAD-SCALING SPEEDUP (workers=1 vs workers=8) on a warm-cache 24-well
    #     projection. The measured ratio is ~1.2-1.5x and the work is bandwidth-bound, so the
    #     margin is thin by construction and collapses whenever the machine is busy -- which,
    #     under a full-suite run, it is. It measures the HOST, not the code.
    #     EVIDENCE that this is not breakage: checked out clean origin/main (aee948b) into a
    #     throwaway worktree with zero local changes and ran the full suite -- same single
    #     failure, "1 failed, 1072 passed". It also passes in isolation every time.
    #     This is the weakness the comment above warns about (a test that fails only under
    #     full-suite resource pressure), so it is named rather than waved through, and the
    #     isolation re-run below still has to pass.
    #     TODO: the real fix is to make that test measure per-well cost against a cold cache, or
    #     mark it as a benchmark and take it out of the commit gate. Then delete this entry.
    KNOWN_FLAKES="tests/test_integration.py::test_ima188_sim1536_scaling_measured_no_regression"

    UNKNOWN=""
    for t in $FAILED; do
      case " $KNOWN_FLAKES " in
        *" $t "*) ;;
        *) UNKNOWN="$UNKNOWN $t" ;;
      esac
    done

    if [ -n "$UNKNOWN" ]; then
      echo ""
      echo "commit-gate: REFUSED. These failures are NOT named known flakes:"
      echo "$UNKNOWN" | tr ' ' '\n' | sed '/^$/d; s/^/    /'
      echo ""
      echo "  a failure is only tolerated if it is listed in KNOWN_FLAKES with its ticket."
      echo "  fix it, or name it there and say why it races."
      rm -f /tmp/squidhcs_gate.$$
      exit 1
    fi

    echo "commit-gate: re-running the named flakes in isolation ..."
    if QT_QPA_PLATFORM=offscreen PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
       python -m pytest -q $FAILED >/dev/null 2>&1; then
      echo "commit-gate: named flakes passed in isolation. Allowing."
      rm -f /tmp/squidhcs_gate.$$
      exit 0
    fi
    echo "commit-gate: a NAMED flake failed even in isolation - that is breakage, not a race."
  fi
  echo ""
  tail -30 /tmp/squidhcs_gate.$$
  rm -f /tmp/squidhcs_gate.$$
  echo ""
  echo "commit-gate: REFUSED. Tests fail and they are not the known flakes."
  echo "  fix them, or commit on a -wip branch that can never merge,"
  echo "  or SQUIDHCS_STOP_ORDER=1 git commit ... for a real stop order."
  exit 1
fi

rm -f /tmp/squidhcs_gate.$$
rm -rf "$GATE_BASETEMP" 2>/dev/null || true
echo "commit-gate: green."
exit 0
