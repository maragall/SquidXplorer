"""`python -m squidxplorer` -> the CLI (IMA-186).

``sys.exit(main())`` and not a bare ``main()``: this is the invocation
``docs/hcs-viewer-quickstart.md`` tells users to run, and dropping the return value would exit 0
over an empty plate on exactly the surface the console script (which already does
``sys.exit(main())``) now reports honestly. See ``squidxplorer._cli`` for the exit codes.
"""

import sys

from squidxplorer._cli import main

if __name__ == "__main__":
    sys.exit(main())
