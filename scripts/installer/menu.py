"""The install-time checkbox menu, generated FROM the operator registry.

One row per install payload (``extra=`` on the operator record; ``core`` is everything
undeclared). The rows are plain data so a future GUI and bootstrap.py consume the same
table; ``--print`` renders them for a terminal. Stdlib only, Qt-free.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

from bootstrap import gpu_backend

Probe = Callable[[], tuple[str, str]]

# Which rows start ticked lives with the extra, not per operator: core + stitch + decon in.
DEFAULT_CHECKED = frozenset({"core", "stitch", "decon"})

# decon installs everywhere; the GPU probe is the row's NOTE (which backend will run), never
# a shade. bootstrap.default_extras adds the CUDA payload when the probe says CUDA.
GPU_NOTED = frozenset({"decon"})


@dataclass(frozen=True)
class ExtraRow:
    """One checkbox: an extra, the operators it unlocks, and a note for the row."""

    extra: str
    operators: tuple[str, ...]
    requires: tuple[str, ...]
    checked: bool
    note: str = ""


def build_menu(probe: Probe = gpu_backend) -> tuple[ExtraRow, ...]:
    """Group the registry by ``extra=``: core first, then the extras alphabetically."""
    import squidxplorer

    by_extra: dict[str, list[str]] = {"core": []}
    for name in squidxplorer.runnable_operators():
        by_extra.setdefault(squidxplorer.operator_extra(name) or "core", []).append(name)

    rows = []
    for extra in ["core"] + sorted(k for k in by_extra if k != "core"):
        operators = tuple(sorted(by_extra[extra]))
        requires = tuple(sorted(
            {m for n in operators for m in squidxplorer.operator_requires(n)}))
        note = probe()[1] if extra in GPU_NOTED else ""
        rows.append(ExtraRow(extra, operators, requires,
                             checked=extra in DEFAULT_CHECKED, note=note))
    return tuple(rows)


def render(rows: tuple[ExtraRow, ...]) -> str:
    lines = []
    for row in rows:
        box = "[x]" if row.checked else "[ ]"
        line = f"{box} {row.extra:<8} {', '.join(row.operators)}"
        if row.requires:
            line += f"  (needs {', '.join(row.requires)})"
        if row.note:
            line += f"  {row.note}"
        lines.append(line)
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print", action="store_true", dest="print_menu",
                        help="render the menu for a terminal")
    args = parser.parse_args(argv)
    if args.print_menu:
        print(render(build_menu()))
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
