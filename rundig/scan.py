"""Scan files, recover experiments, and build a report."""

import fnmatch
import hashlib
import os
import re
import stat
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .metadata import ARTIFACT_FOLDERS, extract, identify
from .lineage import infer
from .git_history import attach_git
from .models import Artifact, Report, stable_id

CHECKPOINT_EXTENSIONS = {".pt", ".pth", ".ckpt", ".safetensors", ".npz"}
NPZ_INSPECTION_LIMIT = 64_000_000
DEFAULT_EXCLUDES = {".git", ".hg", ".svn", ".venv", "venv", "env", "node_modules",
                    "__pycache__", ".mypy_cache", ".pytest_cache", ".work", "site-packages"}
SCRIPT_EXTENSIONS = {".py", ".sh", ".bash", ".ps1", ".bat", ".ipynb"}
TEXT_EXTENSIONS = {".json", ".jsonl", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".csv", ".tsv", ".log", ".txt"}


def is_link(path):
    """Also skip Windows junctions/reparse points, which os.walk may follow."""
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def classify(path: Path) -> str | None:
    suffix, stem = path.suffix.lower(), path.stem.lower()
    if suffix in CHECKPOINT_EXTENSIONS:
        return "checkpoint"
    if suffix in SCRIPT_EXTENSIONS:
        return "script"
    if path.name.startswith("events.out.tfevents"):
        return "tensorboard"
    if suffix not in TEXT_EXTENSIONS:
        return None
    if suffix in {".yaml", ".yml", ".toml", ".ini", ".cfg"} or any(word in stem for word in ("config", "hparam", "params", "arguments", "settings", "options")) or stem == "args":
        return "config"
    if any(word in stem for word in ("history", "metric", "result", "summary", "progress")):
        return "metrics"
    if suffix == ".log" or stem in {"train", "training", "stdout", "stderr"}:
        return "log"
    return "metadata"


def npz_checkpoint(path):
    """Inspect ZIP entry names only; never deserialize arrays or pickle objects."""
    hint = bool(re.match(r"(model|weights?|params?|checkpoint|ckpt|best|last|final|epoch|step)($|[_\W]|\d)",
                          path.stem, re.I)) or path.parent.name.lower() in ARTIFACT_FOLDERS
    # Bound ZIP directory parsing even for archives with millions of entries.
    if path.stat().st_size > NPZ_INSPECTION_LIMIT:
        return hint, ["NPZ inspection skipped above 64 MB; model status is inferred from its filename or folder only."]
    try:
        with zipfile.ZipFile(path) as archive:
            names = {entry.filename[:-4].lower() for entry in archive.infolist()
                     if entry.filename.lower().endswith(".npy") and not entry.is_dir()}
    except (zipfile.BadZipFile, ValueError) as error:
        return hint, [f"Cannot read NPZ archive ({type(error).__name__}); checkpoint contents are unverified."]
    if not names:
        return hint, ["NPZ archive has no array entries; checkpoint contents are unverified."]
    parameters = {name for name in names if re.search(
        r"(^|[./_])((weight|bias|kernel|embedding|param(?:eter)?)s?($|[./_\d])|[wb]\d*$)", name)}
    state = any(re.search(r"(^|[./_])(epoch|global_step|optimizer|velocity|momentum)($|[./_])", name)
                for name in names)
    if len(parameters) >= 2 or (parameters and state):
        return True, []
    # A feature/data cache must not become a checkpoint just because it is NPZ.
    if names & {"features", "ids", "labels", "targets", "samples", "x_train", "y_train"}:
        return False, []
    if parameters and hint:
        return True, []
    # Unnamed NumPy arrays can be a model when the save location says so. Named
    # non-parameter arrays (predictions, explanations, etc.) are not weights.
    guessed = hint and all(re.fullmatch(r"arr_\d+", name) for name in names)
    return guessed, (["NPZ checkpoint inferred from its filename or folder; array names do not establish model parameters."]
                     if guessed else [])


def inventory(root: Path, *, excludes=(), output: Path | None = None, text_limit=2_000_000,
              hash_limit=64_000_000, hash_budget=512_000_000, max_files=100_000):
    artifacts, warnings = [], []
    stats = {"visited_files": 0, "skipped_directories": 0, "skipped_symlinks": 0,
             "hashed_checkpoint_bytes": 0, "unhashed_checkpoints": 0}
    patterns = list(excludes)

    def excluded(path):
        relative = path.relative_to(root).as_posix()
        return path.name in DEFAULT_EXCLUDES or any(
            fnmatch.fnmatchcase(relative, pattern) or fnmatch.fnmatchcase(path.name, pattern)
            for pattern in patterns
        ) or (output is not None and path.resolve() == output)

    def onerror(error):
        warnings.append({"path": str(error.filename), "message": str(error), "kind": "read_error"})

    for folder, directories, filenames in os.walk(root, followlinks=False, onerror=onerror):
        kept = []
        for name in sorted(directories):
            path = Path(folder) / name
            try:
                if is_link(path) or not path.resolve().is_relative_to(root) or excluded(path):
                    stats["skipped_directories"] += 1
                else:
                    kept.append(name)
            except OSError as error:
                onerror(error)
        directories[:] = kept
        for name in sorted(filenames):
            stats["visited_files"] += 1
            if stats["visited_files"] > max_files:
                warnings.append({"path": ".", "message": f"File limit ({max_files}) reached; scan incomplete.", "kind": "limit"})
                return artifacts, warnings, stats
            path = Path(folder) / name
            try:
                if is_link(path):
                    stats["skipped_symlinks"] += 1
                    continue
                if excluded(path):
                    continue
                kind = classify(path)
                if kind is None:
                    continue
                info = path.stat()
                if not stat.S_ISREG(info.st_mode):
                    continue
                relative = path.relative_to(root).as_posix()
                item = Artifact(stable_id("file", relative), relative, kind, info.st_size, info.st_mtime)
                if path.suffix.lower() == ".npz":
                    recognized, notes = npz_checkpoint(path)
                    if not recognized:
                        warnings.extend({"kind": "identification", "path": relative, "message": note} for note in notes)
                        continue
                    item.warnings.extend(notes)
                should_hash = kind in {"script", "config"} and info.st_size <= text_limit
                if kind == "checkpoint":
                    should_hash = info.st_size <= hash_limit and stats["hashed_checkpoint_bytes"] + info.st_size <= hash_budget and hash_limit > 0
                if should_hash:
                    digest = hashlib.sha256()
                    with path.open("rb") as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(chunk)
                    after = path.stat()
                    if (after.st_size, after.st_mtime_ns) != (info.st_size, info.st_mtime_ns):
                        item.warnings.append("File changed while hashing; hash discarded.")
                    else:
                        item.sha256 = digest.hexdigest()
                    if kind == "checkpoint":
                        stats["hashed_checkpoint_bytes"] += info.st_size
                if kind == "checkpoint" and item.sha256 is None:
                    stats["unhashed_checkpoints"] += 1
                artifacts.append(item)
            except OSError as error:
                onerror(error)
    return artifacts, warnings, stats


ASSUMPTIONS = [
    "Run boundaries and relationship scores are estimates; evidence is retained for review.",
    "Similarity and identical files do not prove ancestry. File modification times never orient inferred lineage.",
    "Files are read only. Checkpoints and scripts are never executed; extraction limits and skipped files are reported.",
]


def analyze(root, *, output=None, excludes=(), text_limit=2_000_000, hash_limit=64_000_000,
            hash_budget=512_000_000, max_files=100_000, similarity_threshold=.6,
            max_pairs=200_000, git=True):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Not a readable directory: {root}")
    output = Path(output).resolve() if output else None
    if output == root:
        raise ValueError("Report output must not be the scan root itself.")
    artifacts, warnings, statistics = inventory(root, excludes=excludes, output=output, text_limit=text_limit,
                                                hash_limit=hash_limit, hash_budget=hash_budget, max_files=max_files)
    for item in artifacts:
        extract(root, item, text_limit)
        warnings.extend({"kind": "extraction", "path": item.path, "message": warning} for warning in item.warnings)
    experiments, ownership = identify(artifacts)
    for experiment in experiments:
        if experiment.path == ".":
            experiment.name = root.name
    edges, inference_warnings = infer(experiments, artifacts, ownership, similarity_threshold=similarity_threshold,
                                       max_pairs=max_pairs)
    warnings.extend(inference_warnings)
    if git:
        warnings.extend(attach_git(root, experiments, artifacts, edges))
    statistics.update({
        "artifacts": len(artifacts), "artifact_kinds": dict(Counter(item.kind for item in artifacts)),
        "experiments": len(experiments), "anchored_experiments": sum(item.kind == "experiment" for item in experiments),
        "collections": sum(item.kind == "collection" for item in experiments),
        "checkpoint_candidates": sum(item.kind == "checkpoint_candidate" for item in experiments),
        "checkpoints": sum(item.kind == "checkpoint" for item in artifacts),
        "checkpoint_bytes": sum(item.size for item in artifacts if item.kind == "checkpoint"),
        "edges": len(edges), "edge_kinds": dict(Counter(item.kind for item in edges)),
        "unassigned_artifacts": len(artifacts) - len(ownership),
        "unresolved_references": sum(ref.get("resolution") in {"unresolved", "ambiguous"} for exp in experiments for ref in exp.references),
        "warnings": len(warnings),
        "partial": any(item["kind"] in {"limit", "read_error"} for item in warnings),
    })
    options = {"excludes": sorted(DEFAULT_EXCLUDES) + list(excludes), "text_limit_bytes": text_limit,
               "checkpoint_hash_limit_bytes": hash_limit, "checkpoint_hash_budget_bytes": hash_budget,
               "max_files": max_files, "similarity_threshold": similarity_threshold,
               "max_pairs": max_pairs, "git": git}
    return Report("1.0", __version__, str(root), datetime.now(timezone.utc).isoformat(), options,
                  statistics, experiments, edges, artifacts, warnings, ASSUMPTIONS)
