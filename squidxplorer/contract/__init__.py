"""Machine-enforceable half of the plate contract (see ``docs/plate-contract.md``).

Only ``version`` and ``paths`` are re-exported; import the validator explicitly.
"""

from squidxplorer.contract.paths import field_levels, field_path
from squidxplorer.contract.version import (
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
