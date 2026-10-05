"""Two commands: analyze experiments, or list checkpoint files."""

import argparse
import os
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

from . import __version__
from .scan import analyze
from .report import validate_output, write_report


def format_size(size):
    """Format bytes using decimal units, such as 20MB or 1.5GB."""
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if size < 1000 or unit == "PB":
            value = f"{size:.1f}".rstrip("0").rstrip(".")
            return f"{value}{unit}"
        size /= 1000


def list_checkpoints(args):
    root = args.directory.expanduser().resolve()
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
            if path.suffix.lower() not in {".pt", ".pth", ".ckpt"}:
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


def main(argv=None):
    parser = argparse.ArgumentParser(prog="run-shelf", description="Find experiments and trace their relationships.")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    scan = commands.add_parser("scan", help="Analyze a folder and create a browser report")
    scan.add_argument("directory", type=Path)
    scan.add_argument("--output", "-o", type=Path, default=Path("run-shelf-report"), help="Report folder (default: run-shelf-report)")
    scan.add_argument("--exclude", action="append", default=[], metavar="GLOB", help="Skip a folder or path pattern; repeatable")
    scan.add_argument("--open", action="store_true", help="Open the report in your browser")
    listing = commands.add_parser("list", help="List checkpoints in the terminal")
    listing.add_argument("directory", type=Path)
    listing.add_argument("--sort", choices=("name", "time", "size"), default="name", help="Name A-Z, newest first, or largest first")
    listing.add_argument("--name", default="", help="Filename contains this text (case-insensitive)")
    args = parser.parse_args(argv)
    root = args.directory.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Directory does not exist or is not a folder: {root}")
    if args.command == "list":
        return list_checkpoints(args)
    output = args.output.expanduser().resolve()
    try:
        validate_output(output)
        print(f"Scanning {root} ...", flush=True)
        report = analyze(root, output=output, excludes=args.exclude)
        html = write_report(report, output)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    stats = report.statistics
    print(f"Found {stats['checkpoints']} checkpoints in {stats['experiments']} experiment candidates.")
    print(f"Report: {html}")
    if stats["warnings"]:
        print(f"{stats['warnings']} reading notes in the report.")
    if stats["partial"]:
        print("Some files could not be scanned; see report notes.", file=sys.stderr)
    if args.open:
        webbrowser.open(html.as_uri())
    return 1 if stats["partial"] else 0
