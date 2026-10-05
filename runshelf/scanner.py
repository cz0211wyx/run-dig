"""Bounded, read-only filesystem inventory. Never deserialize model files."""

import fnmatch
import hashlib
import os
import stat
from pathlib import Path

from .models import Artifact, stable_id

CHECKPOINT_EXTENSIONS = {".pt", ".pth", ".ckpt", ".safetensors"}
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
