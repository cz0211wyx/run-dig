"""Scan experiments, list checkpoints, or inspect one checkpoint."""

import argparse
import json
import os
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

from . import __version__
from .scan import CHECKPOINT_EXTENSIONS, analyze
from .metadata import ARTIFACT_FOLDERS
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


def inspect_checkpoint(args):
    checkpoint = args.checkpoint.expanduser()
    root = args.root.expanduser().resolve() if args.root else None
    checkpoint = ((root / checkpoint) if root and not checkpoint.is_absolute() else checkpoint).resolve()
    if not checkpoint.is_file() or checkpoint.suffix.lower() not in CHECKPOINT_EXTENSIONS:
        raise ValueError(f"Not a checkpoint file: {checkpoint}")
    if root is None:
        root = checkpoint.parent.parent if checkpoint.parent.name in ARTIFACT_FOLDERS else checkpoint.parent
    if not checkpoint.is_relative_to(root):
        raise ValueError("Checkpoint must be inside --root.")
    print(f"Scanning {root} ...", flush=True)
    report = analyze(root)
    relative = checkpoint.relative_to(root).as_posix()
    artifact = next((item for item in report.artifacts if item.path == relative), None)
    if artifact is None:
        raise ValueError("Checkpoint was excluded or unreadable; it was not analyzed.")
    experiment = next(item for item in report.experiments if artifact.id in item.artifacts)
    print(f"\nCheckpoint: {checkpoint}\nSize: {format_size(artifact.size)}")
    print(f"Modified: {datetime.fromtimestamp(artifact.modified):%Y-%m-%d %H:%M:%S}")
    print(f"Experiment: {experiment.path}")
    print(f"Scan root: {root}")
    if args.root is None:
        print("Use --root DIRECTORY to include sources and experiments outside this folder.")
    if experiment.kind == "checkpoint_candidate":
        print("Run unknown: no reliable experiment metadata found.")
    by_id = {item.id: item for item in report.experiments}
    descriptions = {"derived_from": "recorded continuation", "references": "checkpoint/component use", "possible_derivation": "possible continuation"}
    for heading, side in (("Sources", "target"), ("Used by", "source")):
        related = [edge for edge in report.edges if edge.directed and getattr(edge, side) == experiment.id]
        print(f"\n{heading}:")
        if not related:
            print("  No supported link found in this scan.")
        for edge in related:
            other = by_id[edge.source if side == "target" else edge.target]
            print(f"  {other.path} ({descriptions[edge.kind]}, confidence {edge.confidence:.0%})")
            for fact in edge.evidence:
                if fact.get("artifact"):
                    detail = f"component: {fact['component']}" if fact.get('component') else fact.get('field', '')
                    print(f"    Evidence: {fact['artifact']}" + (f" | {detail}" if detail else ""))
            for caveat in edge.caveats:
                print(f"    {caveat}")
    for reference in experiment.references:
        if reference.get("resolution") in {"unresolved", "ambiguous"}:
            print(f"  {reference['resolution'].capitalize()} source: {reference['value']}")
    # These are run-level facts, not values read from the model's serialized state.
    for heading, values in (("Run configuration", experiment.config), ("Run metrics", experiment.metrics)):
        print(f"\n{heading}:")
        visible = {key: value for key, value in values.items() if not any(word in key.lower() for word in ("sha256", "hash"))}
        if not visible:
            print("  Not found.")
        core = {"model", "model_type", "architecture", "seed", "lr", "learning_rate", "batch_size", "epochs"}
        keys = sorted(visible, key=lambda key: (not (key.rsplit('.', 1)[-1] in core if heading == "Run configuration" else 'best_' in key), key))
        for key in keys if args.full else keys[:8]:
            value = visible[key]
            value = f"{value:.6g}" if isinstance(value, float) else json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
            print(f"  {key}: {value[:200]}" + ("..." if len(value) > 200 else ""))
        if not args.full and len(keys) > 8:
            print(f"  ... {len(keys) - 8} more fields; use --full to show all.")
    if report.statistics["warnings"]:
        print(f"\n{report.statistics['warnings']} reading/inference notes; some metadata may be unavailable.")
    return 1 if report.statistics["partial"] else 0


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
    inspect = commands.add_parser("inspect", help="Print checkpoint information, metadata, and sources")
    inspect.add_argument("checkpoint", type=Path, help="Checkpoint file; relative to --root when provided")
    inspect.add_argument("--root", type=Path, help="Scan this directory for related experiments (default: checkpoint's folder)")
    inspect.add_argument("--full", action="store_true", help="Show all configuration and metric fields")
    args = parser.parse_args(argv)
    if args.command == "inspect":
        try:
            return inspect_checkpoint(args)
        except (OSError, ValueError) as error:
            print(f"Error: {error}", file=sys.stderr)
            return 2
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
