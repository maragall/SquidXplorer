"""The reader contract: the one name Squid depends on across the repo boundary.

``SquidAcquisitionReader`` is structural, so Squid's own reader — a live one over a running
acquisition included — satisfies it by shape alone, with no import pointing back at this
package's readers. ``metadata`` promises an ``Acquisition`` mapping (see
``squidxplorer._acquisition.Acquisition``): ``regions``, ``fovs_per_region``
(``{region: [fov, ...]}``), ``fov_positions_um``, ``channels``, ``n_z`` / ``z_levels`` /
``dz_um``, ``n_t``, ``frame_shape``, ``dtype``, ``pixel_size_um`` — micrometres throughout,
``_um``-suffixed. ``read`` returns one 2-D plane; ``plane_ref`` names where that plane lives.

This module stays import-light (typing only) on purpose: it is the contract, not the readers.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class SquidAcquisitionReader(Protocol):
    """Structural Protocol satisfied by every reader :func:`squidxplorer.open_reader` can return."""

    @property
    def metadata(self) -> dict:  # pragma: no cover - protocol declaration
        ...

    @property
    def source_id(self) -> str:  # pragma: no cover - protocol declaration
        """Stable identity of the acquisition READ — the directory whose sidecars date it.

        This is what cache keys and staleness tokens are built from. It was a de-facto
        contract member for years, spelled ``reader._path`` and reached by five modules as
        private state — with a different meaning per reader (the Zarr reader's ``_path`` is
        the STORE, while its sidecars live at the acquisition root, so the staleness token
        statted files that never exist). Declared, it means ONE thing.
        """
        ...

    def read(self, region, fov, channel, z_level, time_point=0):  # pragma: no cover - protocol declaration
        ...

    def plane_ref(self, region, fov, channel, z_level, time_point=0) -> tuple:  # pragma: no cover - protocol
        ...
