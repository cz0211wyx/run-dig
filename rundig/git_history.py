"""Local Git context. Code history is never treated as model ancestry."""

import os
import re
import shutil
import subprocess
import time
from pathlib import Path


COMMIT_KEYS = {"commit", "commit_hash", "git_commit", "git_commit_hash", "git_sha",
               "git_hash", "git_revision", "git.commit", "git.commit_hash", "git.sha", "git.hash", "git.revision"}
COMMIT_ID = re.compile(r"(?:[0-9a-f]{7,40}|[0-9a-f]{64})", re.I)


def is_commit_field(key):
    key = key.lower()
    if key.startswith(("run.", "train.", "training.")):
        key = key.split(".", 1)[1]
    return key in COMMIT_KEYS


def field_commits(fields, line=None):
    return [{"field": key, "value": value, **({"line": line} if line is not None else {})}
            for key, value in fields.items() if is_commit_field(key) and value != "[REDACTED]"]


def text_commits(text):
    """Keep repeated log declarations, including conflicting versions."""
    pattern = r"(?<![\w.])(git[_. ]commit(?:_hash)?|git[_. ]sha|git[_. ]hash|git[_. ]revision|commit_hash|commit)\s*[:=]\s*['\"]?([0-9a-f]{7,64})\b"
    return [{"field": match[1].lower().replace(" ", "_"), "value": match[2].lower(), "line": number}
            for number, line in enumerate(text.splitlines(), 1) for match in re.finditer(pattern, line, re.I)]


class GitReader:
    def __init__(self, root, warnings):
        self.root, self.warnings = root, warnings
        self.executable = shutil.which("git")
        self.deadline, self.calls = time.monotonic() + 20, 0
        self.cache, self.locations, self.repositories = {}, {}, {}
        self.untrusted = set()
        self.limited = False
        # Do not inherit a caller's GIT_DIR, alternate index, tracing, or config.
        self.env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
        self.env.update(GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0",
                        GIT_NO_LAZY_FETCH="1", GIT_NO_REPLACE_OBJECTS="1")

    def run(self, repository, *args):
        key = (repository, *args)
        if key in self.cache:
            return self.cache[key]
        if not self.executable:
            return None
        remaining = self.deadline - time.monotonic()
        if self.calls >= 200 or remaining <= 0:
            if not self.limited:
                self.warnings.append({"kind": "git", "path": ".",
                                      "message": "Git inspection budget reached; some Git context is unavailable."})
                self.limited = True
            return None
        self.calls += 1
        try:
            result = subprocess.run(
                [self.executable, "--no-pager", "--literal-pathspecs", "-c", "core.fsmonitor=false",
                 "-c", "protocol.allow=never", "-C", str(repository), *args],
                capture_output=True, timeout=min(3, remaining), env=self.env,
            )
            if b"detected dubious ownership" in result.stderr:
                self.untrusted.add(repository)
            value = (result.returncode, result.stdout.decode("utf-8", errors="replace").strip())
        except (OSError, subprocess.TimeoutExpired):
            value = None
        self.cache[key] = value
        return value

    def locate(self, folder):
        if folder in self.locations:
            return self.locations[folder]
        marker = folder / ".git"
        if marker.exists():
            # A worktree's regular .git pointer file is supported; symlinks are not.
            result = None if marker.is_symlink() else self.run(folder, "rev-parse", "--show-toplevel")
            repository = Path(result[1]).resolve() if result and result[0] == 0 else None
            if repository is None:
                message = "Git is not installed; local history is unavailable." if not self.executable else "Local Git repository could not be read; history is unavailable."
                if folder in self.untrusted:
                    message = "Git refused untrusted repository ownership; history is unavailable."
                self.warnings.append({"kind": "git", "path": self.relative(folder), "message": message})
        else:
            repository = self.locate(folder.parent) if folder != folder.parent else None
        self.locations[folder] = repository
        return repository

    def relative(self, path):
        return Path(os.path.relpath(path, self.root)).as_posix()

    def commits(self, repository, revision, *paths, count=1):
        result = self.run(repository, "log", "--no-patch", "--no-show-signature", "--no-decorate",
                          "--no-notes", "--no-renames", "--no-ext-diff", "--no-textconv",
                          "--encoding=UTF-8", f"--max-count={count}", "--format=%H%x00%ct%x00%s",
                          revision, "--", *paths)
        if not result or result[0]:
            return None
        entries = []
        for line in result[1].splitlines():
            parts = line.split("\0", 2)
            if len(parts) == 3 and COMMIT_ID.fullmatch(parts[0]) and parts[1].isdigit():
                entries.append({"commit": parts[0], "date": int(parts[1]), "summary": parts[2][:300]})
        return entries

    def context(self, repository):
        if repository not in self.repositories:
            branch = self.run(repository, "symbolic-ref", "--quiet", "--short", "HEAD")
            shallow = self.run(repository, "rev-parse", "--is-shallow-repository")
            head = self.commits(repository, "HEAD", count=5)
            self.repositories[repository] = {
                "repository": self.relative(repository),
                "branch": branch[1] if branch and branch[0] == 0 else None,
                "head": head[0] if head else None,
                "recent_commits": head or [],
                "shallow": shallow[1] == "true" if shallow and shallow[0] == 0 else None,
            }
        return dict(self.repositories[repository])

    def resolve(self, repository, value):
        # Resolve immutable commit IDs only, never branch names or revision expressions.
        if not isinstance(value, str) or not COMMIT_ID.fullmatch(value):
            return None
        result = self.run(repository, "rev-parse", "--verify", "--end-of-options", value + "^{commit}")
        if not result or result[0] or not COMMIT_ID.fullmatch(result[1]):
            return None
        entries = self.commits(repository, result[1])
        return entries[0] if entries else None

    def relation(self, repository, first, second):
        if first == second:
            return "same recorded commit"
        forward = self.run(repository, "merge-base", "--is-ancestor", first, second)
        backward = self.run(repository, "merge-base", "--is-ancestor", second, first)
        if forward and forward[0] == 0:
            return "first recorded commit is an ancestor of the second"
        if backward and backward[0] == 0:
            return "second recorded commit is an ancestor of the first"
        if forward and backward and forward[0] == backward[0] == 1 and self.context(repository)["shallow"] is False:
            return "recorded commits are on separate lines of development"
        return None


def attach_git(root, experiments, artifacts, edges, *, progress=None):
    """Attach bounded, read-only history and annotate existing graph links only."""
    warnings = []
    reader = GitReader(root, warnings)
    index = {item.id: item for item in artifacts}
    by_id = {item.id: item for item in experiments}
    resolved = {}
    for number, experiment in enumerate(experiments):
        if progress:
            progress("Reading Git history", number, len(experiments))
        if experiment.kind == "collection":
            continue
        members = [index[key] for key in experiment.artifacts]
        records, seen, incomplete = [], set(), []
        for item in members:
            if item.kind not in {"config", "metadata", "log", "metrics"}:
                continue
            declarations = item.metadata.get("git_commits", field_commits(item.metadata.get("fields", {})))
            if declarations and item.metadata.get("git_records_complete") is False:
                incomplete.append(item.path)
            for declaration in declarations:
                record = {"artifact": item.path, **declaration}
                key = (item.path, record.get('line'), record['field'], str(record['value']))
                if key not in seen:
                    seen.add(key)
                    records.append(record)
        folder = root / experiment.path
        if experiment.kind == "checkpoint_candidate":
            folder = folder.parent
        repository = reader.locate(folder)
        if repository is None and not records:
            continue
        data = reader.context(repository) if repository else {"repository": None}
        recent = data.pop("recent_commits", [])
        experiment.git = data
        data.update(recorded_commits=[], history=[], history_scope="files", notes=[])
        data["notes"].append("Current checkout and file history are scan-time context, not the code version used to train a checkpoint. Uncommitted changes are not inspected.")
        if data.get("shallow"):
            data["notes"].append("Shallow repository: local history is incomplete.")
        if data.get("head") is None and repository:
            data["notes"].append("HEAD is unavailable (empty repository or unreadable history).")
        for record in records:
            same_repository = repository and reader.locate((root / record["artifact"]).parent) == repository
            commit = reader.resolve(repository, record["value"]) if same_repository else None
            data["recorded_commits"].append({**record, "resolution": "resolved" if commit else "unverified", **(commit or {})})
        verified = data["recorded_commits"]
        versions = {record["commit"] for record in verified if record["resolution"] == "resolved"}
        declared = sorted({str(record.get("commit", record["value"])).lower() for record in verified
                           if COMMIT_ID.fullmatch(str(record.get("commit", record["value"])))})
        # A short ID and its full spelling can agree even without local objects.
        if any(not second.startswith(first) for first, second in zip(declared, declared[1:])):
            data["notes"].append("Conflicting recorded commits; the training code version is unknown.")
        if any(record["resolution"] == "unverified" for record in verified):
            data["notes"].append("Some recorded commits could not be verified in the local repository.")
        if incomplete:
            data["notes"].append("Incomplete Git records; training code version remains unknown: " + ", ".join(incomplete))
        if verified and not incomplete and len(versions) == 1 and all(record["resolution"] == "resolved" for record in verified):
            resolved[experiment.id] = (repository, next(iter(versions)))
        if repository and data.get("head"):
            # Query only this run's actual files, never unrelated changes elsewhere
            # in the enclosing directory. Literal pathspecs handle brackets and '*'.
            paths = sorted({(root / item.path).relative_to(repository).as_posix() for item in members
                            if (root / item.path).is_relative_to(repository) and reader.locate((root / item.path).parent) == repository})
            if len(paths) > 100 or sum(len(path) for path in paths) > 12000:
                data["notes"].append("File history omitted: too many artifact paths for a single bounded query.")
            elif paths:
                history = reader.commits(repository, data["head"]["commit"], *paths, count=5)
                if history is None:
                    data["notes"].append("File history is unavailable.")
                else:
                    data["history"] = history
        if not data["history"] and recent:
            data["history"], data["history_scope"] = recent, "repository"
    for edge in edges:
        first, second = resolved.get(edge.source), resolved.get(edge.target)
        if not first or not second or first[0] != second[0]:
            continue
        relation = reader.relation(first[0], first[1], second[1])
        if relation:
            edge.evidence.append({"signal": "recorded Git versions", "value": relation,
                                  "source_commit": first[1], "target_commit": second[1],
                                  "artifacts": sorted({record['artifact'] for key in (edge.source, edge.target)
                                                       for record in by_id[key].git.get('recorded_commits', [])}),
                                  "repository": reader.relative(first[0])})
            edge.caveats.append("Git ancestry describes code history, not checkpoint inheritance; recorded commits may omit uncommitted changes.")
    if progress:
        progress("Reading Git history", len(experiments), len(experiments))
    return warnings
