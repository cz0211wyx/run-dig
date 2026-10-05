"""Support the original `python run_shelf.py DIRECTORY` command."""

import sys

from runshelf.cli import format_size, main


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] not in {"scan", "list", "--version"}:
        args.insert(0, "list")
    sys.exit(main(args))
