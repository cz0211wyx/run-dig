"""Public CLI. The original checkpoint listing remains available via `list`."""

import argparse
import sys
import webbrowser
from pathlib import Path

from . import __version__, legacy
from .pipeline import analyze
from .report import validate_output, write_report


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def nonnegative(value):
    value = float(value)
    if not 0 <= value < float("inf"):
        raise argparse.ArgumentTypeError("must be a finite nonnegative number")
    return value


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "list":
        return legacy.main(argv[1:])
    parser = argparse.ArgumentParser(prog="run-shelf", description="Recover experiment metadata and evidence-backed lineage.")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    scan = subparsers.add_parser("scan", help="Analyze a directory and generate JSON + an offline HTML graph")
    scan.add_argument("directory", type=Path)
    scan.add_argument("--output", "-o", type=Path, default=Path("run-shelf-report"), help="Report directory (default: ./run-shelf-report)")
    scan.add_argument("--exclude", action="append", default=[], metavar="GLOB", help="Exclude a relative path or directory name; repeatable")
    scan.add_argument("--open", action="store_true", help="Open the HTML report in your browser")
    scan.add_argument("--max-text-mb", type=nonnegative, default=2, help="Text extraction limit per file (default: 2 MB)")
    scan.add_argument("--hash-max-mb", type=nonnegative, default=64, help="Checkpoint hash limit per file; 0 disables hashing")
    scan.add_argument("--hash-budget-mb", type=nonnegative, default=512, help="Total checkpoint hashing budget (default: 512 MB)")
    scan.add_argument("--max-files", type=positive, default=100_000)
    scan.add_argument("--max-pairs", type=positive, default=200_000, help="Similarity comparison budget")
    scan.add_argument("--similarity-threshold", type=nonnegative, default=.6)
    scan.add_argument("--neighbors", type=positive, default=3, help="Maximum inferred similarity neighbors per node")
    subparsers.add_parser("list", help="List checkpoints with --sort and --name (use list --help)")
    args = parser.parse_args(argv)
    if args.similarity_threshold > 1:
        parser.error("--similarity-threshold must be between 0 and 1")
    root, output = args.directory.expanduser().resolve(), args.output.expanduser().resolve()
    try:
        validate_output(output)
        print(f"Scanning {root} ...", flush=True)
        report = analyze(root, output=output, excludes=args.exclude,
                         text_limit=int(args.max_text_mb * 1_000_000), hash_limit=int(args.hash_max_mb * 1_000_000),
                         hash_budget=int(args.hash_budget_mb * 1_000_000), max_files=args.max_files,
                         max_pairs=args.max_pairs, similarity_threshold=args.similarity_threshold, neighbors=args.neighbors)
        html = write_report(report, output)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    stats = report.statistics
    print(f"Found {stats['checkpoints']} checkpoints, {stats['anchored_experiments']} experiments, "
          f"{stats['checkpoint_candidates']} uncertain checkpoint candidates, {stats['edges']} relationships.")
    print(f"Warnings: {stats['warnings']}; unresolved references: {stats['unresolved_references']}.")
    if stats["partial"]:
        print("Scan was partial; review report warnings and configured limits.", file=sys.stderr)
    print(f"JSON: {output / 'report.json'}\nBrowse: {html}")
    if args.open:
        webbrowser.open(html.as_uri())
    return 1 if stats["partial"] else 0
