"""ASHLAR adapter (Harvard LSP) -- the first external stitcher in the comparison."""

from __future__ import annotations

import subprocess
import sys

from bench.adapters.base import StitcherAdapter, StitchRequest, python_module_available


class AshlarAdapter(StitcherAdapter):
    """ASHLAR, driven through its Python API by ``bench.drivers.ashlar_driver``.

    ASHLAR solves a pure translation per tile, so its positions drop straight into the
    seam-residual metric with no model conversion -- which is exactly why it is one of
    the two tools IMA-233 starts with.
    """

    name = "ashlar"

    def is_available(self) -> bool:
        return python_module_available("ashlar")

    def version(self) -> str:
        try:
            out = subprocess.run(
                [sys.executable, "-c", "import ashlar;print(getattr(ashlar,'__version__','unknown'))"],
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
            "bench.drivers.ashlar_driver",
            "--input", str(req.acquisition.root),
            "--region", req.region,
            "--channel", req.channel,
            "--z", str(req.z),
            "--out", str(req.out_dir),
            "--threads", str(req.threads),
        ]
