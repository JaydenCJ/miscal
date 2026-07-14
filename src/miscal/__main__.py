"""Allow ``python -m miscal`` to behave exactly like the ``miscal`` script."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
