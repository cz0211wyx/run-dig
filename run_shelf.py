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


def main():
    parser = argparse.ArgumentParser(description="Recursively find .pt, .pth, and .ckpt files.")
    parser.add_argument("directory", type=Path, help="Directory to scan")
    args = parser.parse_args()
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
            try:
                info = path.stat()
            except OSError as error:
                report_error(error)
                continue
            modified = datetime.fromtimestamp(info.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            checkpoints.append((path, info.st_size, modified))

    total_size = sum(size for _, size, _ in checkpoints)
    print(f"Found {len(checkpoints)} checkpoints, total size: {total_size / 1_000_000_000:.1f} GB")
    print("\nPath\tSize\tLast modified")
    for path, size, modified in checkpoints:
        print(f"{path}\t{format_size(size)}\t{modified}")
    if errors:
        print(f"Encountered {errors} read error(s); results may be incomplete.", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
