"""Scan experiments, list checkpoints, and inspect or compare their metadata."""

import argparse
import json
import os
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

from . import __version__
from .scan import CHECKPOINT_EXTENSIONS, analyze, is_link, npz_checkpoint
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
        kept = []
        for name in sorted(directories):
            try:
                if not is_link(Path(folder) / name):
                    kept.append(name)
            except OSError as error:
                report_error(error)
        directories[:] = kept
        for filename in sorted(filenames):
            path = Path(folder) / filename
            if path.suffix.lower() not in CHECKPOINT_EXTENSIONS:
                continue
            if args.name.casefold() not in filename.casefold():
                continue
            try:
                if is_link(path) or not path.is_file():
                    continue
                info = path.stat()
                if path.suffix.lower() == ".npz":
                    recognized, notes = npz_checkpoint(path)
                    for note in notes:
                        print(f"Note: {path}: {note}", file=sys.stderr)
                    if not recognized:
                        continue
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


def load_checkpoints(paths, root=None):
    root = root.expanduser().resolve() if root else None
    checkpoints = []
    for path in paths:
        path = path.expanduser()
        path = ((root / path) if root and not path.is_absolute() else path).resolve()
        if not path.is_file() or path.suffix.lower() not in CHECKPOINT_EXTENSIONS:
            raise ValueError(f"Not a checkpoint file: {path}")
        checkpoints.append(path)
    if root is None:
        parents = [path.parent.parent if path.parent.name in ARTIFACT_FOLDERS else path.parent for path in checkpoints]
        root = Path(os.path.commonpath(parents))
        if root == Path(root.anchor):
            raise ValueError("Please set --root to an experiment directory instead of scanning an entire drive.")
    if any(not checkpoint.is_relative_to(root) for checkpoint in checkpoints):
        raise ValueError("Checkpoints must be inside --root.")
    print(f"Scanning {root} ...", flush=True)
    report = analyze(root)
    artifacts = {item.path: item for item in report.artifacts if item.kind == "checkpoint"}
    owners = {key: item for item in report.experiments for key in item.artifacts}
    result = []
    for checkpoint in checkpoints:
        artifact = artifacts.get(checkpoint.relative_to(root).as_posix())
        if artifact is None:
            raise ValueError("File was excluded, unreadable, or not recognized as a model checkpoint.")
        result.append((checkpoint, artifact, owners[artifact.id]))
    return report, result


def display_value(value, limit=200):
    value = f"{value:.6g}" if isinstance(value, float) else json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    return value[:limit] + ("..." if len(value) > limit else "")


def print_checkpoint_metrics(artifact, full=False):
    print("\nCheckpoint metrics:")
    records = artifact.metadata.get("metric_records", [])
    if not records:
        print("  Unknown: no supported metric record for this file.")
    for record in records:
        print(f"  {record['source']} (confidence {record['confidence']:.0%})")
        print(f"    {record['basis']}")
        keys = sorted(record["values"], key=lambda key: (key != "epoch", "score" not in key, key))
        for key in keys if full else keys[:8]:
            print(f"    {key}: {display_value(record['values'][key])}")
        if not full and len(keys) > 8:
            print(f"    ... {len(keys) - 8} more fields; use --full to show all.")


def inspect_checkpoint(args):
    report, [(checkpoint, artifact, experiment)] = load_checkpoints([args.checkpoint], args.root)
    print(f"\nCheckpoint: {checkpoint}\nSize: {format_size(artifact.size)}")
    print(f"Modified: {datetime.fromtimestamp(artifact.modified):%Y-%m-%d %H:%M:%S}")
    print(f"Experiment: {experiment.path}")
    if experiment.parent_id:
        parent = next(item for item in report.experiments if item.id == experiment.parent_id)
        print(f"Collection: {parent.path} ({parent.subtree_checkpoint_count} checkpoints in sub-experiments)")
    if experiment.roles:
        print("Role: " + ", ".join(f"{key}={value}" for key, value in experiment.roles.items()))
    print(f"Scan root: {report.root}")
    for warning in artifact.warnings:
        print(f"Note: {warning}")
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
            if edge.description:
                print(f"    {edge.description}")
            for fact in edge.evidence:
                if fact.get("artifact"):
                    detail = f"component: {fact['component']}" if fact.get('component') else fact.get('field', '')
                    print(f"    Evidence: {fact['artifact']}" + (f" | {detail}" if detail else ""))
            for caveat in edge.caveats:
                print(f"    {caveat}")
    for reference in experiment.references:
        if reference.get("resolution") in {"unresolved", "ambiguous"}:
            print(f"  {reference['resolution'].capitalize()} source: {reference['value']}")
    print_checkpoint_metrics(artifact, args.full)
    # Run summaries remain useful context, but must not be mistaken for this file.
    for heading, values in (("Run configuration", experiment.config), ("Run metrics", experiment.metrics)):
        print(f"\n{heading}:")
        if heading == "Run metrics":
            print("  Run summaries / last parsed log values; not necessarily this checkpoint.")
        visible = {key: value for key, value in values.items() if not any(word in key.lower() for word in ("sha256", "hash"))}
        if not visible:
            print("  Not found.")
        core = {"model", "model_type", "architecture", "seed", "lr", "learning_rate", "batch_size", "epochs"}
        keys = sorted(visible, key=lambda key: (not (key.rsplit('.', 1)[-1] in core if heading == "Run configuration" else 'best_' in key), key))
        for key in keys if args.full else keys[:8]:
            sources = experiment.config_sources.get(key, []) if heading == "Run configuration" else []
            if any(source["value"] != visible[key] for source in sources):
                print(f"  {key}: Conflicting values")
                for source in sources:
                    print(f"    {display_value(source['value'])} | {source['artifact']}")
            else:
                print(f"  {key}: {display_value(visible[key])}")
        if not args.full and len(keys) > 8:
            print(f"  ... {len(keys) - 8} more fields; use --full to show all.")
    if report.statistics["warnings"]:
        print(f"\n{report.statistics['warnings']} reading/inference notes; some metadata may be unavailable.")
    return 1 if report.statistics["partial"] else 0


def compare_checkpoints(args):
    report, entries = load_checkpoints([args.first, args.second], args.root)
    for label, (path, artifact, experiment) in zip("AB", entries):
        print(f"{label}: {path} ({format_size(artifact.size)})")
        print(f"   Experiment: {experiment.path}")
    left, right = entries[0][2], entries[1][2]
    print("\nRun relationship:")
    relations = [edge for edge in report.edges if {edge.source, edge.target} == {left.id, right.id}]
    if left.id == right.id:
        print("  Same experiment; file-to-file ancestry is unknown.")
    elif not relations:
        print("  No supported relationship in this scan.")
    for edge in relations:
        direction = ("A -> B" if edge.source == left.id else "B -> A") if edge.directed else "A -- B"
        print(f"  {direction}: {edge.kind.replace('_', ' ')} (confidence {edge.confidence:.0%})")
        if edge.description:
            print(f"    {edge.description}")
        for caveat in edge.caveats:
            print(f"    {caveat}")

    def config(experiment):
        result = {}
        for key, value in experiment.config.items():
            if value == "[REDACTED]" or any(word in key.lower() for word in ("sha256", "hash")):
                continue
            alternatives = experiment.config_sources.get(key, [])
            if any(source["value"] != value for source in alternatives):
                value = {"conflicting_values": [source["value"] for source in alternatives]}
            elif any(source.get("inherited") for source in alternatives):
                value = f"{display_value(value)} (inferred parent config)"
            result[key] = value
        return result

    def table(title, a, b, differences=False):
        print(f"\n{title}:")
        keys = sorted(set(a) | set(b), key=lambda key: (key != "epoch", "score" not in key, key))
        if differences:
            keys = [key for key in keys if key not in a or key not in b or a[key] != b[key]]
        if not keys:
            print("  No differences in recovered fields." if differences and (a or b) else "  No comparable metadata found.")
            return
        rows = [(key, display_value(a[key]) if key in a else "(not recorded)", display_value(b[key]) if key in b else "(not recorded)")
                for key in (keys if args.full else keys[:20])]
        # Keep ordinary terminals readable; --full retains every field/value.
        if not args.full:
            rows = [tuple(display_value(cell, 52 if index else 48) for index, cell in enumerate(row)) for row in rows]
        widths = [max(len(row[index]) for row in [("Name", "A", "B"), *rows]) for index in range(3)]
        for row in [("Name", "A", "B"), *rows]:
            print("  " + " | ".join(cell.ljust(width) for cell, width in zip(row, widths)))
        if not args.full and len(keys) > 20:
            print(f"  ... {len(keys) - 20} more fields; use --full to show all.")

    table("Configuration differences", config(left), config(right), differences=True)
    recovered = []
    for label, (_, artifact, _) in zip("AB", entries):
        records = artifact.metadata.get("metric_records", [])
        values = {}
        print(f"\n{label} metric sources:")
        if not records:
            print("  Unknown; run-level metrics are not substituted.")
        for record in records:
            print(f"  {record['source']} ({record['confidence']:.0%}) - {record['basis']}")
            for key, value in record["values"].items():
                if key in values and values[key] != value:
                    values[key] = "Conflicting records; see sources."
                else:
                    values[key] = value
        recovered.append(values)
    table("Checkpoint metrics (same field names; check evaluation settings)", *recovered)
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
    compare = commands.add_parser("compare", help="Compare two checkpoints' recovered configuration and metrics")
    compare.add_argument("first", type=Path)
    compare.add_argument("second", type=Path)
    compare.add_argument("--root", type=Path, help="Scan directory; checkpoint paths are relative to it (default: common parent)")
    compare.add_argument("--full", action="store_true", help="Show every field and longer values")
    args = parser.parse_args(argv)
    if args.command in {"inspect", "compare"}:
        try:
            return inspect_checkpoint(args) if args.command == "inspect" else compare_checkpoints(args)
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
    collections = stats['collections']
    print(f"Found {stats['checkpoints']} checkpoints in {stats['experiments'] - collections} experiment candidates" +
          (f" and {collections} collections." if collections else "."))
    print(f"Report: {html}")
    if stats["warnings"]:
        print(f"{stats['warnings']} reading notes in the report.")
    if stats["partial"]:
        print("Some files could not be scanned; see report notes.", file=sys.stderr)
    if args.open:
        webbrowser.open(html.as_uri())
    return 1 if stats["partial"] else 0
