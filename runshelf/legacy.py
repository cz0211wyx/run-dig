"""Recursively list checkpoint files without loading their contents."""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path


EXTENSIONS = {".pt", ".pth", ".ckpt"}


def format_size(size):
    """Format bytes using decimal units, such as 20MB or 1.5GB."""
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if size < 1000 or unit == "PB":
            value = f"{size:.1f}".rstrip("0").rstrip(".")
            return f"{value}{unit}"
        size /= 1000


def main(argv=None):
    parser = argparse.ArgumentParser(description="Recursively find .pt, .pth, and .ckpt files.")
    parser.add_argument("directory", type=Path, help="Directory to scan")
    parser.add_argument(
        "--sort", choices=("name", "time", "size"), default="name",
        help="Sort by name (A-Z, default), time (newest first), or size (largest first)",
    )
    parser.add_argument("--name", default="", help="Filter by filename substring (case-insensitive)")
    args = parser.parse_args(argv)
    root = args.directory.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Directory does not exist or is not a folder: {root}")

    checkpoints = []
    errors = 0

    def report_error(error):
        nonlocal errors
        errors += 1
        print(f"Cannot read: {error}", file=sys.stderr)

    for folder, directories, filenames in os.walk(root, onerror=report_error):
        directories.sort()
        for filename in sorted(filenames):
            path = Path(folder) / filename
            if path.suffix.lower() not in EXTENSIONS:
                continue
            if args.name.casefold() not in filename.casefold():
                continue
            try:
                info = path.stat()
            except OSError as error:
                report_error(error)
                continue
            checkpoints.append((path, info.st_size, info.st_mtime))

    sort_keys = {
        "name": lambda item: (item[0].name.casefold(), str(item[0]).casefold()),
        "time": lambda item: (-item[2], str(item[0]).casefold()),
        "size": lambda item: (-item[1], str(item[0]).casefold()),
    }
    checkpoints.sort(key=sort_keys[args.sort])
    total_size = sum(size for _, size, _ in checkpoints)
    print(f"Found {len(checkpoints)} checkpoints, total size: {total_size / 1_000_000_000:.1f} GB")

    name_width = max([len("name")] + [len(path.name) for path, _, _ in checkpoints])
    size_width = max([len("size")] + [len(format_size(size)) for _, size, _ in checkpoints])
    print(f"\n{'name':<{name_width}}  {'size':>{size_width}}  {'modified':<19}  path")
    for path, size, modified in checkpoints:
        modified = datetime.fromtimestamp(modified).strftime("%Y-%m-%d %H:%M:%S")
        print(f"{path.name:<{name_width}}  {format_size(size):>{size_width}}  {modified}  {path}")
    if errors:
        print(f"Encountered {errors} read error(s); results may be incomplete.", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
