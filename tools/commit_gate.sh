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

# PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 is required (without it the PyQt5 tests silently skip against
# PySide) but it also stops pytest-timeout loading, so `--timeout=` becomes an UNRECOGNISED
# ARGUMENT and pytest dies in 0.115s having collected nothing. That is how this gate shipped: it
# never ran a single test, and every agent learned to reach for -wip. Load the plugin explicitly.
#
# No -x either. -x stops at the first failure, so the failure list handed to the flake re-run below
# was truncated to one name; a real `assert False` sharing a run with a known flake could exit 0.
# The whole suite runs, and every failure is re-checked.
echo "commit-gate: running the suite before allowing this commit ..."
if ! QT_QPA_PLATFORM=offscreen PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
     python -m pytest -q -p pytest_timeout --timeout=900 >/tmp/squidhcs_gate.$$ 2>&1; then
  # The four known-flaky tests (IMA-258) are races that pass in isolation. A flake must not
  # block a commit, but it must not silently pass one either - so re-run just the failures.
  # If pytest never collected anything, this is a BROKEN GATE, not a green tree. Refuse loudly.
  # The gate shipped in exactly this state and silently gated nothing for five worktrees.
  if ! grep -qE "^[0-9]+ (passed|failed)|passed|failed" /tmp/squidhcs_gate.$$; then
    echo ""
    tail -20 /tmp/squidhcs_gate.$$
    rm -f /tmp/squidhcs_gate.$$
    echo ""
    echo "commit-gate: REFUSED - pytest did not run (no pass/fail summary)."
    echo "  the gate is broken, not the tree. fix the gate before committing."
    exit 1
  fi
  FAILED=$(grep -E "^FAILED " /tmp/squidhcs_gate.$$ | sed 's/^FAILED //; s/ .*//' || true)
  if [ -n "$FAILED" ]; then
    echo "commit-gate: re-running failures in isolation to separate flakes from breakage ..."
    if QT_QPA_PLATFORM=offscreen PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
       python -m pytest -q $FAILED >/dev/null 2>&1; then
      echo "commit-gate: all failures passed in isolation - known flakes (IMA-258). Allowing."
      rm -f /tmp/squidhcs_gate.$$
      exit 0
    fi
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
echo "commit-gate: green."
exit 0
