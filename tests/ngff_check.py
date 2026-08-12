"""Back-compat shim. The real validator is ``squidxplorer.contract.validate``.

Kept because five test modules import this name; use
``python -m squidxplorer.contract.validate /path/to/plate.ome.zarr`` directly otherwise.
"""

from squidxplorer.contract.validate import assert_valid_ngff_plate  # noqa: F401
