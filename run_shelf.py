"""Compatibility entry point for checkpoint listing and the archaeology CLI."""

import sys

from runshelf.cli import main
from runshelf.legacy import format_size, main as list_checkpoints


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in {"scan", "list", "--version"}:
        sys.exit(main())
    sys.exit(list_checkpoints())
