"""pytest plugin: durably record every test's outcome as it happens."""

import os

_RESULT_FILE = os.environ.get("SQUIDHCS_RESULT_FILE")


def pytest_runtest_logreport(report):
    if not _RESULT_FILE:
        return
    line = f"{report.when}\t{report.outcome}\t{report.nodeid}\n"
    with open(_RESULT_FILE, "a") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())
