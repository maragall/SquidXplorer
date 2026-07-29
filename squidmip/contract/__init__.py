"""The plate contract: what is on disk, what is promised about it, and how to check.

Written form: ``docs/plate-contract.md``. That document is the contract; this package is the part
of it a machine can enforce.

    version.py   ``PLATE_CONTRACT_VERSION``, stamped by ``_output`` and COMPARED by ``reader``,
                 plus the mismatch policy (refuse on a major, warn on a minor).
    paths.py     ``field_path``, the one place that knows ``{row}/{col}/{fov}/{level}``.
    validate.py  errors (stable violations) versus warnings (missing optional sidecars), runnable
                 as ``python -m squidmip.contract.validate <plate.ome.zarr>``.

Only ``version`` and ``paths`` are re-exported here, and both are import-cheap: ``_output`` pulls
this package on every plate write, and ``validate`` reaches back into ``reader``. Import the
validator explicitly (``from squidmip.contract.validate import validate_plate``).
"""

from squidmip.contract.paths import field_levels, field_path
from squidmip.contract.version import (
    CONTRACT_KEY,
    CONTRACT_NAMESPACE,
    PLATE_CONTRACT_VERSION,
    PlateContractError,
    check_plate_contract,
    compare_contract_version,
    contract_stamp,
    read_contract_version,
)

__all__ = [
    "CONTRACT_KEY",
    "CONTRACT_NAMESPACE",
    "PLATE_CONTRACT_VERSION",
    "PlateContractError",
    "check_plate_contract",
    "compare_contract_version",
    "contract_stamp",
    "field_levels",
    "field_path",
    "read_contract_version",
]
