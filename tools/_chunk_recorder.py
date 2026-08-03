"""pytest plugin: durably record every test's outcome as it happens.

Why. Some chunks run every test to ``[100%]`` and pass, then SEGFAULT during interpreter/Qt teardown
(accumulated napari/Qt C++ objects finalized in an order the libraries do not survive — a headless
harness hazard, not a test failure). That crash kills the process before pytest prints its summary and
turns the exit code negative, so neither the summary line nor the exit code can be trusted to tell us
what happened.

So we do not rely on either. This plugin appends one line per test-phase report to
``$SQUIDHCS_RESULT_FILE``, flushed and fsynced immediately, so the record survives even if the very
next thing the process does is crash. ``tools/run_suite_chunked.py`` reads this file — not the exit
code — to decide pass/fail, and cross-checks that EVERY requested test actually produced a ``call``
(or was skipped at setup). A test with no record did not run: a crash that happens PART-WAY through a
chunk is therefore still caught (its later tests are missing) and can never be waved through as green.

Line format:  ``<when>\t<outcome>\t<nodeid>``   e.g. ``call\tpassed\ttests/test_x.py::test_y``
"""

import os

_RESULT_FILE = os.environ.get("SQUIDHCS_RESULT_FILE")


def pytest_runtest_logreport(report):
    if not _RESULT_FILE:
        return
    # setup/call/teardown each produce a report. We keep all three: a setup 'failed'/'skipped' means
    # the call never happens, and a teardown 'failed' must still fail the test. The runner folds the
    # phases per nodeid (any failed -> failed; call passed -> passed; setup skipped -> skipped).
    line = f"{report.when}\t{report.outcome}\t{report.nodeid}\n"
    # Open-append-flush-fsync per line: the process may segfault immediately after any test, and a
    # buffered write would be lost exactly when we most need it.
    with open(_RESULT_FILE, "a") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())
