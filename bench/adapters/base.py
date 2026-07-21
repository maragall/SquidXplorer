"""The contract every stitcher adapter implements.

An adapter's job is narrow and deliberately so: say whether the tool is installed, say
what version it is, produce an argv the runner can spawn, and afterwards hand back the
tile positions the tool solved for. It does NOT measure anything -- measurement belongs
to the runner, so that every tool is measured by identical code (IMA-233 D3).

Adapters also declare what CANNOT be measured about them. That is what keeps the
report honest: a tool whose work happens inside a Docker cgroup sets
``measurable_rss = False`` and its RSS cell renders ``n/a`` instead of a small,
confident, wrong number.

Positions are the load-bearing output. The quality metric re-reads input tiles and
places them at these positions, so an adapter that cannot produce them cannot be
scored on quality -- and says so via ``quality_na_reason`` rather than returning 0.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from bench.dataset import Acquisition


@dataclass
class StitchRequest:
    """Everything an adapter needs to build a command."""

    acquisition: Acquisition
    region: str
    channel: str
    z: int
    out_dir: Path
    threads: int = 1
    compression: str = "none"


@dataclass
class AdapterOutcome:
    """What the adapter can say after the subprocess finished."""

    positions_px: dict[int, tuple[float, float]] | None = None
    quality_na_reason: str = ""
    extra: dict = field(default_factory=dict)


class StitcherAdapter(ABC):
    """Base class for a benchmarked stitcher."""

    name: str = "unnamed"

    #: False when the real work happens outside our process tree (containers), so
    #: process-tree RSS is structurally blind and must not be reported as a number.
    measurable_rss: bool = True
    #: False when output lands in a container volume our `du` cannot see.
    measurable_output_bytes: bool = True
    #: False when the tool's transform model has no scalar per-tile shift
    #: (BigStitcher non-rigid), making the seam metric undefined rather than unwritten.
    supports_quality: bool = True
    #: Populated when supports_quality is False; a key of report.QUALITY_NA_REASONS.
    quality_na_reason: str = ""

    @abstractmethod
    def is_available(self) -> bool:
        """True when the tool can actually be invoked on this machine."""

    @abstractmethod
    def version(self) -> str:
        """Tool version string for the row's provenance. Never the repo git sha."""

    @abstractmethod
    def build_command(self, req: StitchRequest) -> list[str]:
        """argv for the subprocess the runner will spawn and measure."""

    def prepare(self, req: StitchRequest) -> None:
        """Optional input conversion before the command runs. Default: nothing."""

    def collect(self, req: StitchRequest) -> AdapterOutcome:
        """Read back what the tool solved for, after a successful run.

        Default implementation reads ``positions.json`` from the output directory --
        the contract the bundled drivers write. Tools with a native position format
        (BigStitcher XML, PetaKit5D .mat) override this.
        """
        return AdapterOutcome(positions_px=read_positions_json(req.out_dir))


def read_positions_json(out_dir: str | Path) -> dict[int, tuple[float, float]] | None:
    """Read ``positions.json`` -> ``{fov: (row_px, col_px)}``; None when absent/invalid."""
    p = Path(out_dir) / "positions.json"
    if not p.is_file():
        return None
    try:
        raw = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    positions: dict[int, tuple[float, float]] = {}
    for k, v in (raw.get("positions_px") or raw).items():
        try:
            positions[int(k)] = (float(v[0]), float(v[1]))
        except (TypeError, ValueError, IndexError):
            continue
    return positions or None


def python_module_available(module: str) -> bool:
    """True when ``module`` is importable, without importing it into this process.

    Deliberately out-of-process: importing ``tilefusion`` pulls numba, GPU probing and
    basicpy through its ``__init__``, which is slow and can fail outright. The harness
    must be able to *ask* without paying that cost.
    """
    try:
        return (
            subprocess.run(
                ["python", "-c", f"import {module}"],
                capture_output=True,
                timeout=120,
            ).returncode
            == 0
        )
    except (OSError, subprocess.SubprocessError):
        return False


def executable_available(name: str) -> bool:
    return shutil.which(name) is not None
