# CI gates work counts, not wall clock

Performance regressions here are regressions in *work done*, chiefly reading the same plane more
than once, and CI has only synthetic fixtures and shared, noisy runners. So automated gates assert
counts of work, such as the number of `reader.read` calls an open-then-run sequence makes, which
are exact, deterministic and host-independent. Durations are measured, recorded and compared, but
never asserted in CI.

## Considered options

**Wall clock in CI** was the obvious choice and was tried. `tools/commit_gate.sh` had to quarantine
`test_ima188_sim1536_scaling_measured_no_regression` because, in its own words, *"It measures the
HOST, not the code."* That test has since been rewritten to gate peak concurrency instead of a
timing ratio. `tests/test_benchmark.py` states the second half of the problem: *"a benchmark
asserted against synthetic data would be measuring the fixture."* `.spec/STATE.md` records the same
conclusion reached a third time: *"timing asserts flake; repo precedent quarantines them."*

This ADR exists because that conclusion has now been reached independently three times and
re-derived from scratch each time.

**No automated gate at all**, relying on manual before-and-after comparison, was rejected because
it is the status quo that produced a dead regression baseline, a benchmark harness that imports
from another repository, and a viewer benchmark whose subject was deleted.

## Consequences

A regression that makes the app slower without making it do more work will not be caught
automatically. That is accepted: durations are still recorded per run and persisted, so the
comparison remains possible by hand, and latency acceptance criteria are stated relative to a
control run on real data rather than as absolute budgets in CI.
