"""Support the original `python run_dig.py DIRECTORY` command."""

import sys

from rundig.cli import format_size, main


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] not in {"scan", "list", "inspect", "compare", "--version"}:
        args.insert(0, "list")
    sys.exit(main(args))
