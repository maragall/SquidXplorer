"""tilefusion adapter -- the incumbent, and the baseline the others are measured against."""

from __future__ import annotations

import subprocess
import sys

from bench.adapters.base import StitcherAdapter, StitchRequest, python_module_available


class TileFusionAdapter(StitcherAdapter):
    """Cephla's own stitcher (``ian-stitcher`` / tilefusion).

    Run as a subprocess like every other tool even though it is importable Python.
    That is not ceremony: measuring it in-process would use a different instrument
    (tracemalloc) than the one used for the Java and MATLAB tools, and a column whose
    rows come from two instruments cannot be compared -- which is the whole point of
    the table.
    """

    name = "tilefusion"

    def is_available(self) -> bool:
        return python_module_available("tilefusion")

    def version(self) -> str:
        try:
            out = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import tilefusion;print(getattr(tilefusion,'__version__','unknown'))",
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            return out.stdout.strip() or "unknown"
        except (OSError, subprocess.SubprocessError):
            return "unknown"

    def build_command(self, req: StitchRequest) -> list[str]:
        return [
            sys.executable,
            "-m",
            "bench.drivers.tilefusion_driver",
            "--input", str(req.acquisition.root),
            "--region", req.region,
            "--channel", req.channel,
            "--z", str(req.z),
            "--out", str(req.out_dir),
            "--threads", str(req.threads),
        ]
