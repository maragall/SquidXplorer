"""The install-time checkbox menu, generated FROM the operator registry.

One row per install payload (``extra=`` on the operator record; ``core`` is everything
undeclared). The rows are plain data so a future GUI and bootstrap.py consume the same
table; ``--print`` renders them for a terminal. Stdlib only, Qt-free.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Optional

from bootstrap import cuda12_available

Probe = Callable[[], tuple[bool, str]]

# Which rows start ticked lives with the extra, not per operator: core + stitch + decon in,
# segment out (cellpose's torch payload, excluded for size).
DEFAULT_CHECKED = frozenset({"core", "stitch", "decon"})

# petakit is cupy-cuda12x: "has a GPU" is not enough, the driver must speak CUDA 12.
GPU_GATED = frozenset({"decon"})


@dataclass(frozen=True)
class ExtraRow:
    """One checkbox: an extra, the operators it unlocks, and whether it may be chosen."""

    extra: str
    operators: tuple[str, ...]
    requires: tuple[str, ...]
    checked: bool
    enabled: bool = True
    reason: str = ""


def build_menu(probe: Probe = cuda12_available) -> tuple[ExtraRow, ...]:
    """Group the registry by ``extra=``: core first, then the extras alphabetically."""
    import squidxplorer

    by_extra: dict[str, list[str]] = {"core": []}
    for name in squidxplorer.runnable_operators():
        by_extra.setdefault(squidxplorer.operator_extra(name) or "core", []).append(name)

    probed: Optional[tuple[bool, str]] = None
    rows = []
    for extra in ["core"] + sorted(k for k in by_extra if k != "core"):
        operators = tuple(sorted(by_extra[extra]))
        requires = tuple(sorted(
            {m for n in operators for m in squidxplorer.operator_requires(n)}))
        enabled, reason = True, ""
        if extra in GPU_GATED:
            if probed is None:
                probed = probe()
            enabled, reason = probed
        rows.append(ExtraRow(extra, operators, requires,
                             checked=enabled and extra in DEFAULT_CHECKED,
                             enabled=enabled, reason=reason))
    return tuple(rows)


def render(rows: tuple[ExtraRow, ...]) -> str:
    lines = []
    for row in rows:
        box = "[x]" if row.checked else ("[ ]" if row.enabled else "[-]")
        line = f"{box} {row.extra:<8} {', '.join(row.operators)}"
        if row.requires:
            line += f"  (needs {', '.join(row.requires)})"
        if not row.enabled:
            line += f"  — shaded: {row.reason}"
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
