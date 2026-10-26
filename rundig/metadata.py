"""Extract file metadata and identify experiment candidates."""

import ast
import configparser
import csv
import io
import json
import math
import posixpath
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from .models import Artifact, Experiment, stable_id
from .git_history import field_commits, is_commit_field, text_commits

SECRET = re.compile(r"(^|[._-])(password|secret|api_key|token|access_token|auth_token|credential|credentials|authorization|private_key)([._-]|$)", re.I)
METRIC = re.compile(r"(^|[._])(loss|accuracy|acc|f1|macro_f1|weighted_f1|mae|mse|rmse|auc|pearson|score|selection_score|best_epoch|epoch|step|epochs)([._]|$)", re.I)
CHECKPOINT = re.compile(r"\.(pt|pth|ckpt|safetensors|npz)$", re.I)
HASH = re.compile(r"^[0-9a-f]{64}$", re.I)
TRAIN_KEYS = {"seed", "lr", "learning_rate", "epochs", "batch_size", "optimizer", "model_type", "model", "architecture", "dataset", "dropout", "hidden_size"}
METRIC_TARGET_KEYS = {"checkpoint", "checkpoint_path", "output_checkpoint", "saved_checkpoint", "model_path"}


def scalar(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value[:2000] if isinstance(value, str) else value
    return str(value)[:2000]


def flatten(value, prefix="", depth=0, result=None):
    """Bound nesting, field count, and lists (including YAML aliases)."""
    if result is None:
        result = {}
    if depth > 12 or len(result) >= 1000:
        return result
    if SECRET.search(prefix):
        result[prefix] = "[REDACTED]"
    elif isinstance(value, dict):
        for key, child in list(value.items())[:1000]:
            flatten(child, f"{prefix}.{key}".strip("."), depth + 1, result)
            if len(result) >= 1000:
                break
    elif isinstance(value, list):
        if len(value) <= 20 and all(not isinstance(v, (dict, list)) for v in value):
            result[prefix] = [scalar(v) for v in value]
        elif len(value) <= 100:
            for i, child in enumerate(value):
                flatten(child, f"{prefix}[{i}]", depth + 1, result)
        else:
            result[prefix] = f"[omitted list of {len(value)} items]"
    else:
        result[prefix] = scalar(value)
    return result


def parse_time(value):
    if not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            return None  # An undocumented local timezone cannot orient lineage.
        return dt.astimezone(timezone.utc).timestamp()
    except (ValueError, OverflowError):
        return None


def reference_kind(key):
    key = key.lower()
    leaf = key.rsplit(".", 1)[-1]
    if leaf in {"features", "predictions", "logits", "probabilities", "labels", "dataset"} or leaf.endswith("_cache"):
        return None  # A data field inside a checkpoint record is not a model reference.
    if any(word in key for word in ("resume", "init_from", "init_checkpoint", "initial_", "initialization", "parent_run", "warm_start", "finetune_from")):
        return "derived_from"
    if any(word in key for word in ("source_checkpoint", "pretrained", "teacher", "reference", "checkpoint", "load_from",
                                    "expert", "component", "branch", "dependency", "dependencies", "input_model", "weights", "source")):
        return "references"
    return None


def expand_reference(value, fields, key):
    """Substitute a simple {seed} only when its value is unambiguous in metadata."""
    namespace = key.rsplit(".", 1)[0] + "." if "." in key else ""
    def replace(match):
        name = match.group(1)
        choices = [fields[candidate] for candidate in (namespace + name, name) if candidate in fields]
        if not choices:
            choices = [v for k, v in fields.items() if k.endswith("." + name)]
        values = {str(v) for v in choices if isinstance(v, (str, int, float)) and not isinstance(v, bool)}
        return next(iter(values)) if len(values) == 1 else match.group(0)
    return re.sub(r"\{([A-Za-z_]\w*)\}", replace, value)


def record_links(fields):
    """Connect stages explicitly declared in a manifest, without requiring hashes."""
    records = {}
    for key, value in fields.items():
        if key.rsplit(".", 1)[-1] in {"checkpoint", "checkpoint_path", "output_checkpoint"} and isinstance(value, str) and CHECKPOINT.search(value):
            prefix = key.rsplit(".", 1)[0] + "." if "." in key else ""
            if key.endswith("output_checkpoint") or any(isinstance(fields.get(prefix + label), str) for label in ("phase", "stage", "id", "name")):
                records[prefix] = value
    labels = defaultdict(list)
    for prefix, path in records.items():
        label = next((fields[prefix + key] for key in ("phase", "stage", "id", "name") if isinstance(fields.get(prefix + key), str)), None)
        if label:
            labels[label].append(path)
    result = []
    for prefix, target in records.items():
        for key, value in fields.items():
            if not key.startswith(prefix) or not reference_kind(key[len(prefix):]):
                continue
            # Do not borrow a dependency from a nested/sibling stage record.
            if any(other != prefix and len(other) > len(prefix) and key.startswith(other) for other in records):
                continue
            for source in value if isinstance(value, list) else [value]:
                if not isinstance(source, str):
                    continue
                if not CHECKPOINT.search(source):
                    matches = labels.get(source, [])
                    if len(matches) != 1:
                        continue
                    source = matches[0]
                if source != target:
                    result.append({"source": expand_reference(source, fields, key), "target": expand_reference(target, fields, prefix),
                                   "kind": reference_kind(key[len(prefix):]), "field": key})
    return result


def component_links(fields):
    """Read frozen-component bindings from multi-stage training manifests."""
    records = []
    for key, digest in fields.items():
        if not key.endswith("checkpoint_sha256") or not isinstance(digest, str) or not HASH.fullmatch(digest):
            continue
        prefix = key[:-len("checkpoint_sha256")]
        label = next((fields[prefix + name] for name in ("phase", "stage", "id", "name")
                      if isinstance(fields.get(prefix + name), str)), None)
        if label:
            records.append((prefix, label, digest.lower()))
    result = []
    for target, _, target_hash in records:
        if not any(fields.get(target + flag) is True for flag in ("frozen_experts_unchanged", "frozen_components_unchanged")):
            continue
        for source, label, source_hash in records:
            if source == target or sum(record[1] == label for record in records) != 1:
                continue
            for group in ("frozen_expert_state_sha256", "frozen_component_state_sha256"):
                field = target + group + "." + label
                count = sum(key.startswith(field + ".") and isinstance(value, str) and bool(HASH.fullmatch(value))
                            for key, value in fields.items())
                if count >= 3:
                    result.append({"component": label, "source_sha256": source_hash, "target_sha256": target_hash,
                                   "source_record": source.rstrip("."), "target_record": target.rstrip("."),
                                   "field": field, "parameter_hashes": count})
    return result


def extract_fields(fields, source, *, role):
    metadata = {"fields": fields, "metrics": {}, "references": [], "script_hashes": {}, "times": {},
                "component_links": component_links(fields), "record_links": record_links(fields)}
    for key, value in fields.items():
        if value == "[REDACTED]":
            continue
        if isinstance(value, list) and reference_kind(key):
            for index, path in enumerate(value):
                if isinstance(path, str) and CHECKPOINT.search(path.strip()):
                    metadata["references"].append({"value": path, "field": f"{key}[{index}]", "artifact": source,
                                                   "kind": reference_kind(key), "role": role})
        if isinstance(value, (int, float)) and not isinstance(value, bool) and METRIC.search(key):
            metadata["metrics"][key] = value
        lower = key.lower()
        if isinstance(value, str):
            if HASH.fullmatch(value) and any(word in lower for word in ("source_sha", "script_sha", "code_sha")):
                metadata["script_hashes"][key] = value.lower()
            kind = reference_kind(key)
            if CHECKPOINT.search(value.strip()) or (kind == "derived_from" and value.strip()):
                if kind:
                    reference = {"value": value, "resolved_value": expand_reference(value, fields, key),
                                 "field": key, "artifact": source, "kind": kind, "role": role}
                    if key == "source_checkpoint" and PurePosixPath(source).name.lower() == "source.txt":
                        reference["purpose"] = "declared_source"
                    elif role in {"metrics", "metadata"} and key in METRIC_TARGET_KEYS and any(METRIC.search(k) for k in fields):
                        reference["purpose"] = "evaluation"
                    hash_key = key.rsplit(".", 1)[0] + ".sha256"
                    if HASH.fullmatch(str(fields.get(hash_key, ""))):
                        reference["sha256"] = fields[hash_key].lower()
                    metadata["references"].append(reference)
            # A map such as source_checkpoints: {"runs/base/best.pt": "sha256"}.
            if HASH.fullmatch(value) and CHECKPOINT.search(key) and "source_checkpoints." in lower:
                reference_path = key[lower.index("source_checkpoints.") + len("source_checkpoints."):]
                metadata["references"].append({"value": reference_path, "sha256": value.lower(), "field": key, "artifact": source, "kind": "references", "role": role})
            timestamp = parse_time(value)
            if timestamp is not None:
                if any(word in lower for word in ("started_at", "start_time", "start_date")):
                    metadata["times"]["started"] = {"value": timestamp, "field": key}
                elif any(word in lower for word in ("finished_at", "end_time", "completed_at")):
                    metadata["times"]["finished"] = {"value": timestamp, "field": key}
    metadata["training_signal"] = len({key.split(".")[-1].lower() for key in fields} & TRAIN_KEYS) >= 2
    return metadata


def script_fields(text, suffix):
    if suffix != ".py":
        fields = {}
        for match in re.finditer(r'''--([\w-]+)(?:=|\s+)(?:"([^"]*)"|'([^']*)'|([^\s;]+))''', text):
            key = match.group(1).replace("-", "_")
            value = next(part for part in match.groups()[1:] if part is not None)
            if key in TRAIN_KEYS or reference_kind(key):
                fields[key] = value
        return fields
    fields = {}
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return fields
    constants = {}

    def literal(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return constants.get(node.id)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"Path", "PurePath"} and len(node.args) == 1:
            return literal(node.args[0])
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.Add)):
            left, right = literal(node.left), literal(node.right)
            if left is not None and right is not None:
                return str(PurePosixPath(left) / right) if isinstance(node.op, ast.Div) else left + right
        return None

    # Resolve only straight-line module constants. Never evaluate source code or
    # borrow values from another function/conditional scope.
    for node in tree.body:
        if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
            for changed in ast.walk(node):
                if isinstance(changed, ast.Name) and isinstance(changed.ctx, ast.Store):
                    constants.pop(changed.id, None)
        if isinstance(node, ast.Assign):
            value = literal(node.value)
            for target in node.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = value
        for call in ast.walk(node):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.If, ast.For, ast.While, ast.Try, ast.With)):
                break
            if not isinstance(call, ast.Call):
                continue
            method = call.func.attr if isinstance(call.func, ast.Attribute) else call.func.id if isinstance(call.func, ast.Name) else ""
            if method in {"load", "load_checkpoint", "load_from_checkpoint", "load_weights"} and call.args:
                path = literal(call.args[0])
                if path and CHECKPOINT.search(path):
                    fields[f"checkpoint.line_{call.lineno}"] = path
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.lower() in TRAIN_KEYS:
                    fields[target.id.lower()] = scalar(node.value.value)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"load", "load_state_dict", "load_from_checkpoint"} and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and CHECKPOINT.search(arg.value):
                    fields[f"checkpoint.line_{node.lineno}"] = arg.value
    return fields


def extract(root: Path, artifact: Artifact, text_limit=2_000_000, record_limit=2000):
    if artifact.kind in {"checkpoint", "tensorboard"}:
        if artifact.kind == "tensorboard":
            artifact.warnings.append("TensorBoard event file inventoried only; export metrics to CSV/JSON for extraction.")
        return
    path = root / artifact.path
    suffix = path.suffix.lower()
    try:
        with path.open("rb") as stream:
            raw = stream.read(text_limit + 1)
        truncated = len(raw) > text_limit
        if truncated:
            artifact.warnings.append(f"Text exceeds {text_limit} bytes; metadata extraction limited.")
            if suffix in {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".ipynb"}:
                return
            raw = raw[:text_limit].rsplit(b"\n", 1)[0]
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("utf-8-sig", errors="replace")
            artifact.warnings.append("Non-UTF-8 bytes replaced while reading text.")
        fields, records, metric_rows, git_commits = {}, 0, [], []
        git_complete = not truncated

        def remember(value, location, depth=0, record=True):
            # Retain individual rows until checkpoint attribution; never merge epochs.
            if depth > 12 or len(metric_rows) >= record_limit:
                return
            if isinstance(value, dict):
                markers = METRIC_TARGET_KEYS | {"epoch", "best_epoch"}
                separate = {key for key, child in value.items() if isinstance(child, list) or
                            isinstance(child, dict) and markers.intersection(child)}
                row = flatten({key: child for key, child in value.items() if key not in separate}) if record else {}
                nested_targets = [key.rsplit(".", 1)[0] + "." for key in row
                                  if "." in key and key.rsplit(".", 1)[-1] in METRIC_TARGET_KEYS]
                row = {key: child for key, child in row.items() if not any(key.startswith(prefix) for prefix in nested_targets)}
                relevant = markers.intersection(row) or (suffix == ".json" and location == artifact.path)
                if relevant and any(METRIC.search(key) and isinstance(v, (int, float)) for key, v in row.items()):
                    metric_rows.append({"fields": row, "source": location})
                for key, child in value.items():
                    if not SECRET.search(str(key)) and isinstance(child, (dict, list)):
                        # Each nested evaluation keeps its own checkpoint scope.
                        remember(child, f"{location}#{key}", depth + 1, record=key in separate)
            elif isinstance(value, list):
                for index, child in enumerate(value[:record_limit]):
                    remember(child, f"{location}[{index}]", depth + 1)

        if artifact.kind == "script":
            fields = script_fields(text, suffix)
        elif suffix == ".json":
            value = json.loads(text)
            fields = flatten(value)
            remember(value, artifact.path)
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError:
                artifact.warnings.append("Install PyYAML to extract YAML metadata.")
                return
            fields = flatten(yaml.safe_load(text))
        elif suffix == ".toml":
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib
            fields = flatten(tomllib.loads(text))
        elif suffix in {".ini", ".cfg"}:
            parser = configparser.ConfigParser(interpolation=None)
            parser.read_string(text)
            fields = flatten({section: dict(parser[section]) for section in parser.sections()})
        elif suffix in {".jsonl", ".csv", ".tsv"}:
            if suffix == ".jsonl":
                rows = text.splitlines()
                for number, line in enumerate(rows[:record_limit], 1):
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                        row_fields = flatten(value)
                        git_commits.extend(field_commits(row_fields, number))
                        fields.update(row_fields)
                        remember(value, f"{artifact.path}:{number}")
                        records += 1
                    except (ValueError, RecursionError):
                        artifact.warnings.append("Malformed JSONL record skipped.")
                        git_complete = False
                if len(rows) > record_limit:
                    git_complete = False
            else:
                reader = csv.DictReader(io.StringIO(text), delimiter="\t" if suffix == ".tsv" else ",")
                for row in reader:
                    if records >= record_limit:
                        git_complete = False
                        break
                    fields = {str(k): scalar(v) for k, v in row.items() if k is not None}
                    git_commits.extend(field_commits(flatten(fields), reader.line_num))
                    for key, value in list(fields.items()):
                        if is_commit_field(key):
                            continue
                        try:
                            fields[key] = scalar(float(value))
                        except (ValueError, TypeError):
                            pass
                    records += 1
                    remember(fields, f"{artifact.path}:{reader.line_num}")
            if records >= record_limit:
                artifact.warnings.append(f"Record limit ({record_limit}) reached; metrics reflect the last parsed record, not necessarily the final epoch.")
            fields = flatten(fields)
        else:
            if path.name.lower() == "source.txt" and len(text.strip().splitlines()) == 1:
                value = text.strip().strip('"\'')
                if CHECKPOINT.search(value):
                    fields["source_checkpoint"] = value
            # Conservative key=value metadata and explicit checkpoint-loading messages.
            for line in text.splitlines()[:record_limit]:
                for key, value in re.findall(r"\b([\w.]+)\s*[:=]\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", line):
                    fields[key] = scalar(float(value))
                match = re.search(r'''(?:resum\w*|load\w*|initializ\w*|warm.start)\s+(?:\w+\s+){0,3}?(?:from\s+)?(?:["']([^"']+\.(?:pt|pth|ckpt|safetensors|npz))["']|([^\s"']+\.(?:pt|pth|ckpt|safetensors|npz)))''', line, re.I)
                if match:
                    prefix = "resume" if "resum" in line.lower() else "checkpoint"
                    fields[f"{prefix}.{len(fields)}"] = match.group(1) or match.group(2)
        artifact.metadata = extract_fields(fields, artifact.path, role=artifact.kind)
        if suffix in {".log", ".txt"}:
            lines = text.splitlines()
            git_commits = text_commits("\n".join(lines[:record_limit]))
            git_complete = git_complete and len(lines) <= record_limit
        elif suffix not in {".jsonl", ".csv", ".tsv"}:
            git_commits = field_commits(fields)
        artifact.metadata["git_commits"] = git_commits
        artifact.metadata["git_records_complete"] = git_complete
        # Arbitrarily named JSON can still contain a training configuration.
        leaves = {key.split(".")[-1].lower() for key in fields}
        if artifact.kind == "metadata" and suffix == ".json" and len(leaves & {"lr", "learning_rate", "batch_size", "optimizer", "dropout"}) >= 2:
            artifact.kind = "config"
        artifact.metadata["records_read"] = records
        if metric_rows and artifact.kind != "config":
            if len(metric_rows) >= record_limit:
                artifact.warnings.append(f"Metric record limit ({record_limit}) reached; checkpoint attribution may be incomplete.")
            artifact.metadata["_metric_rows"] = metric_rows
            artifact.metadata["_history_complete"] = not truncated and not artifact.warnings and records < record_limit
        # Warnings should not grow with every corrupt line.
        artifact.warnings = list(dict.fromkeys(artifact.warnings))
    except Exception as error:
        # A bad user file must not abort the rest of the read-only scan.
        artifact.warnings.append(f"Metadata parse failed ({type(error).__name__}): {str(error)[:180]}")


ARTIFACT_FOLDERS = {"checkpoints", "checkpoint", "weights", "ckpts", "models"}
SUPPORT_FOLDERS = ARTIFACT_FOLDERS | {"logs", "source", "src", "scripts", "code", "configs", ".hydra"}


def checkpoint_metrics(members):
    """Attribute recorded metrics without opening model files or guessing from mtime."""
    checkpoints = [item for item in members if item.kind == "checkpoint"]
    records = [(item, row) for item in members for row in item.metadata.get("_metric_rows", [])]
    target_keys = METRIC_TARGET_KEYS

    def values(fields, best=False):
        result = {}
        for key, value in fields.items():
            if "[" in key or not METRIC.search(key) or not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            if best:
                if key.startswith("best_"):
                    result[key[5:]] = value
            elif key != "epochs":
                result[key.removeprefix("metrics.")] = value
        return result

    def add(checkpoint, row, selected, confidence, basis):
        if selected:
            checkpoint.metadata.setdefault("metric_records", []).append({
                "values": selected, "source": row["source"], "confidence": confidence, "basis": basis,
            })

    for item, row in records:
        fields = row["fields"]
        declared = [value for key, value in fields.items() if key in target_keys and isinstance(value, str) and CHECKPOINT.search(value)]
        if declared:
            paths = set()
            for value in declared:
                value = value.replace("\\", "/")
                paths.update((posixpath.normpath(value), posixpath.normpath(str(PurePosixPath(item.path).parent / value))))
            matches = [cp for cp in checkpoints if cp.path in paths]
            if len(declared) == 1 and len(matches) == 1:
                add(matches[0], row, values(fields), .95, "Checkpoint path recorded with metrics")
            continue
        # A matching sidecar is useful evidence, but still a naming convention.
        source = PurePosixPath(item.path)
        if source.suffix.lower() != ".json" or row["source"] != item.path:
            continue  # Prediction rows and nested evaluations are not file summaries.
        matches = [cp for cp in checkpoints if source.parent == PurePosixPath(cp.path).parent and
                   source.stem in {PurePosixPath(cp.path).stem, PurePosixPath(cp.path).stem + ".metrics", PurePosixPath(cp.path).name}]
        if len(matches) == 1:
            add(matches[0], row, values(fields, best="best_epoch" in fields and "epoch" not in fields), .8, "Matching metadata filename; model contents not verified")

    best_epochs = {row["fields"]["best_epoch"] for _, row in records
                   if isinstance(row["fields"].get("best_epoch"), (int, float)) and not isinstance(row["fields"]["best_epoch"], bool)
                   and not any(key in row["fields"] for key in target_keys)}
    best_names = {"best", "best_model", "model_best"}
    last_names = {"last", "last_model", "model_last", "latest"}
    histories = defaultdict(list)
    for item, row in records:
        fields = row["fields"]
        epoch = fields.get("epoch")
        sequence = PurePosixPath(item.path).suffix.lower() in {".jsonl", ".csv", ".tsv"} or "[" in row["source"][len(item.path):]
        if sequence and isinstance(epoch, (int, float)) and not isinstance(epoch, bool) and "best_epoch" not in fields and not any(key in fields for key in target_keys):
            histories[item.id].append(row)
    # A final row is usable only from a complete, strictly increasing history.
    last_histories = {item.id: histories[item.id] for item in members if histories.get(item.id)
                      and item.metadata.get("_history_complete")
                      and all(a["fields"]["epoch"] < b["fields"]["epoch"]
                              for a, b in zip(histories[item.id], histories[item.id][1:]))}
    resume_paths = {posixpath.normpath(ref.get("resolved_value", ref["value"]).replace("\\", "/"))
                    for item in members for ref in item.metadata.get("references", [])
                    if ref["kind"] == "derived_from" and "resume" in ref["field"].lower()}
    resume_epochs = [(value, item.path) for item in members if not item.warnings
                     for key, value in item.metadata.get("fields", {}).items()
                     if key.rsplit(".", 1)[-1].lower() == "resume_epoch"]
    resume_basis = ""
    if (len(resume_paths) == 1 and resume_epochs and len(last_histories) == len(histories)
            and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) and value >= 0 and value == int(value) for value, _ in resume_epochs)
            and len({value for value, _ in resume_epochs}) == 1):
        boundary = resume_epochs[0][0]
        before = {key: rows for key, rows in last_histories.items() if rows[-1]["fields"]["epoch"] == boundary}
        after = {key: rows for key, rows in last_histories.items() if rows[0]["fields"]["epoch"] == boundary + 1}
        # Explicit resume evidence and adjacent, disjoint histories are required.
        # Overlapping attempts (including reused directories) remain ambiguous.
        if before and after and len(before) + len(after) == len(last_histories):
            last_histories = after
            sources = ", ".join(sorted({source for _, source in resume_epochs}))
            resume_basis = f"Inferred from last filename and history after resume_epoch={int(boundary)} ({sources}); save not verified"
    for cp in checkpoints:
        if cp.metadata.get("metric_records"):
            continue  # Recorded paths / sidecars take precedence over broad conventions.
        stem = PurePosixPath(cp.path).stem.lower()
        selected_epoch, basis, confidence = None, "", .6
        if stem in best_names and sum(PurePosixPath(other.path).stem.lower() in best_names for other in checkpoints) == 1:
            if len(best_epochs) != 1:
                continue
            selected_epoch = next(iter(best_epochs))
            basis, confidence = "Inferred from best filename and recorded best_epoch", .7
            for _, row in records:
                if row["fields"].get("best_epoch") == selected_epoch and not any(key in row["fields"] for key in target_keys):
                    add(cp, row, values(row["fields"], best=True), confidence, basis)
        else:
            match = re.fullmatch(r"(?:.*[_-])?epoch[=_-]?(\d+)", stem)
            if match:
                selected_epoch = int(match.group(1))
                basis = "Inferred from epoch in filename; numbering not verified"
            elif stem in last_names and sum(PurePosixPath(other.path).stem.lower() in last_names for other in checkpoints) == 1:
                basis = resume_basis or "Inferred from last filename and final recorded epoch; save not verified"
                confidence = .55
            else:
                continue
        for item in members:
            rows = histories.get(item.id, [])
            if not rows:
                continue
            if selected_epoch is None:
                if item.id not in last_histories:
                    continue
                selected = [rows[-1]]
            else:
                selected = [row for row in rows if row["fields"]["epoch"] == selected_epoch]
            if len(selected) == 1:
                add(cp, selected[0], values(selected[0]["fields"]), confidence, basis)

    for cp in checkpoints:
        records = cp.metadata.get("metric_records", [])
        fields = {key for record in records for key in record["values"]}
        conflicts = []
        for key in sorted(fields):
            values_for_key = [record["values"][key] for record in records if key in record["values"]]
            if any(not math.isclose(value, values_for_key[0], rel_tol=1e-6, abs_tol=1e-9) for value in values_for_key[1:]):
                conflicts.append(key)
        if conflicts:
            cp.metadata["metric_conflicts"] = conflicts
            note = "Metric records disagree on " + ", ".join(conflicts) + "; histories or evaluation settings may differ."
            if note not in cp.warnings:
                cp.warnings.append(note)
            for record in records:
                if record["basis"].startswith("Inferred"):
                    record["confidence"] = min(record["confidence"], .4)


def attach_tracking_records(experiments, artifacts, ownership):
    """Attach W&B copies only with a unique checkpoint and compatible run facts."""
    index = {item.id: item for item in artifacts}
    paths = {item.path: item for item in artifacts if item.kind == "checkpoint"}
    runs = {item.id: item for item in experiments}
    removed = set()
    for experiment in experiments:
        folder = PurePosixPath(experiment.path)
        if "wandb" not in folder.parts or folder.name != "files" or not folder.parent.name.startswith(("run-", "offline-run-")):
            continue
        experiment.name = folder.parent.name
        members = [index[key] for key in experiment.artifacts]
        if any(item.kind == "checkpoint" for item in members):
            continue
        declared = [ref for item in members if PurePosixPath(item.path).name == "wandb-summary.json"
                    for ref in item.metadata.get("references", []) if ref.get("purpose") == "evaluation"]
        targets = set()
        for ref in declared:
            value = ref["value"].replace("\\", "/")
            bases = [PurePosixPath(ref["artifact"]).parent, *PurePosixPath(ref["artifact"]).parents]
            matches = {paths[path].id for base in bases
                       if (path := posixpath.normpath(str(base / value))) in paths}
            if len(matches) != 1:
                targets.clear()
                break
            targets.update(ownership[key] for key in matches)
        if len(targets) != 1:
            continue
        target = runs[next(iter(targets))]
        if target.id == experiment.id or "wandb" in PurePosixPath(target.path).parts or target.kind != "experiment":
            continue
        own = [index[key] for key in target.artifacts]
        def facts(items):
            configs = [item for item in items if item.kind == "config"]
            result = defaultdict(set)
            for item in configs or items:
                for key, value in item.metadata.get("fields", {}).items():
                    if isinstance(value, (str, int, float, bool)):
                        kind = "number" if isinstance(value, (int, float)) and not isinstance(value, bool) else type(value).__name__
                        result[key].add((kind, value))
            return result
        a, b = facts(own), facts(members)
        common = a.keys() & b.keys()
        if len(common) < 3 or any(len(a[key]) != 1 or a[key] != b[key] for key in common):
            continue
        for item in members:
            ownership[item.id] = target.id
            item.metadata["tracking_copy_of"] = target.path
            target.artifacts.append(item.id)
        target.evidence.append(f"Tracking records attached by checkpoint path and matching settings: {experiment.path}")
        removed.add(experiment.id)
    experiments[:] = [item for item in experiments if item.id not in removed]


def identify(artifacts):
    by_dir = defaultdict(list)
    for item in artifacts:
        by_dir[str(PurePosixPath(item.path).parent)].append(item)

    anchors = {}
    for directory, items in sorted(by_dir.items()):
        kinds = {item.kind for item in items}
        training = any(item.metadata.get("training_signal") for item in items if item.kind == "config")
        config = any(item.kind == "config" and item.metadata.get("fields") for item in items)
        metrics = any(item.metadata.get("metrics") for item in items if item.kind in {"metrics", "log"})
        references = any(item.metadata.get("references") for item in items if item.kind == "config")
        state = any(PurePosixPath(item.path).name.lower() in {"state.json", "run.json", "status.json", "metadata.json"}
                    and item.metadata.get("times") for item in items)
        child_checkpoint = any(
            item.kind == "checkpoint"
            for folder in ARTIFACT_FOLDERS
            for item in by_dir.get(str(PurePosixPath(directory) / folder), [])
        )
        # Config subfolders are supporting evidence, not independent runs.
        if PurePosixPath(directory).name in {"configs", ".hydra"}:
            continue
        if (config and ("checkpoint" in kinds or child_checkpoint or ((training or references) and metrics))) or (state and metrics) or (("checkpoint" in kinds or child_checkpoint) and metrics):
            evidence = []
            if config:
                evidence.append("Configuration and run artifacts colocated.")
            if training:
                evidence.append("Configuration contains training parameters.")
            if metrics:
                evidence.append("Numeric metrics or training history found.")
            if state:
                evidence.append("Explicit run timestamps found.")
            anchors[directory] = Experiment(stable_id("exp", directory), PurePosixPath(directory).name or ".", directory, .9 if training else .75 if config else .7, evidence)

    # Hydra / config-subfolder layouts: a config plus logs or checkpoints in its parent.
    for directory, items in sorted(by_dir.items()):
        if PurePosixPath(directory).name not in {"configs", ".hydra"}:
            continue
        parent = str(PurePosixPath(directory).parent)
        if parent in anchors or not any(item.kind == "config" and item.metadata.get("training_signal") for item in items):
            continue
        nearby = list(by_dir.get(parent, []))
        for folder in ARTIFACT_FOLDERS | {"logs"}:
            nearby.extend(by_dir.get(str(PurePosixPath(parent) / folder), []))
        checkpoints = [item for item in nearby if item.kind == "checkpoint"]
        run_records = any(item.kind in {"log", "metrics"} or item.metadata.get("times")
                          or item.metadata.get("references") for item in nearby + items)
        if PurePosixPath(directory).name == "configs" and checkpoints and not run_records:
            child_hashes = {item.sha256 for child in anchors if PurePosixPath(parent) in PurePosixPath(child).parents
                            for folder in {".", *ARTIFACT_FOLDERS}
                            for item in by_dir.get(str(PurePosixPath(child) / folder), [])
                            if item.kind == "checkpoint" and item.size > 0 and item.sha256}
            if all(item.size > 0 and item.sha256 in child_hashes for item in checkpoints):
                # Shared project settings plus files also present in child runs do
                # not establish another training run or the copies' configuration.
                for item in checkpoints:
                    item.warnings.append(f"Config in {directory} was not bound: identical checkpoint bytes exist in a child run, but no local training record ties this file to that config.")
                continue
        if any(item.kind in {"checkpoint", "log", "metrics"} for item in nearby):
            anchors[parent] = Experiment(stable_id("exp", parent), PurePosixPath(parent).name or ".", parent, .8, ["Training config in a supporting folder and run artifacts in its parent."])

    ownership = {}
    experiments = list(anchors.values())
    for item in artifacts:
        folder = PurePosixPath(item.path).parent
        owner = None
        for ancestor in [folder, *folder.parents]:
            candidate = anchors.get(str(ancestor))
            if candidate:
                remainder = folder.relative_to(ancestor)
                # A run cannot swallow arbitrary nested runs or copied backup trees.
                if str(remainder) == "." or (remainder.parts and remainder.parts[0] in SUPPORT_FOLDERS):
                    owner = candidate
                break
        if owner is None and item.kind == "checkpoint":
            owner = Experiment(
                stable_id("candidate", item.path), PurePosixPath(item.path).name, item.path, .3,
                ["Checkpoint found without a reliable run boundary."], kind="checkpoint_candidate",
                warnings=["May be an epoch, copy, export, or separate experiment. Filename and directory are insufficient to decide."],
            )
            experiments.append(owner)
        if owner:
            owner.artifacts.append(item.id)
            ownership[item.id] = owner.id

    attach_tracking_records(experiments, artifacts, ownership)
    index = {item.id: item for item in artifacts}
    for experiment in experiments:
        members = [index[key] for key in experiment.artifacts]
        shared = set()
        if experiment.kind == "experiment" and not any(item.kind == "config" for item in members):
            parent = str(PurePosixPath(experiment.path).parent)
            nearby = by_dir.get(parent, [])
            if not any(item.kind == "checkpoint" for item in nearby):
                for item in nearby:
                    if item.kind == "config" and item.metadata.get("training_signal"):
                        members.append(item)
                        shared.add(item.id)
                        experiment.warnings.append(f"Shared parent config: {item.path}; applicability to this run is inferred.")
        # Config facts take precedence over summaries; conflicts keep every source.
        members.sort(key=lambda item: (item.kind != "config", item.path))
        for item in members:
            experiment.modified = max(experiment.modified, item.modified)
            if item.kind == "checkpoint":
                experiment.checkpoint_count += 1
                experiment.checkpoint_bytes += item.size
            metadata = item.metadata
            if item.id not in shared and item.kind in {"config", "metadata", "metrics"}:
                for key, value in metadata.get("fields", {}).items():
                    # Only the run's own role declarations, not nested component
                    # summaries or a role borrowed from its parent's config.
                    leaf = key.rsplit(".", 1)[-1]
                    if leaf in {"phase", "stage", "modality", "component"} and key in {leaf, "train." + leaf, "model." + leaf} and isinstance(value, str) and value and value != "[REDACTED]":
                        experiment.role_sources.setdefault(leaf, []).append({"artifact": item.path, "field": key, "value": value})
            if item.kind == "config":
                for key, value in metadata.get("fields", {}).items():
                    if value == "[REDACTED]":
                        continue
                    source = {"artifact": item.path, "value": value}
                    if item.id in shared:
                        source["inherited"] = True
                    experiment.config_sources.setdefault(key, []).append(source)
                    if key in experiment.config and experiment.config[key] != value:
                        experiment.warnings.append(f"Conflicting config value for {key}; see config_sources.")
                    experiment.config.setdefault(key, value)
            if item.kind in {"metrics", "log", "metadata"}:
                for key, value in metadata.get("metrics", {}).items():
                    # Preserve source-specific metrics instead of merging train/validation silently.
                    metric_key = f"{PurePosixPath(item.path).name}:{key}"
                    experiment.metrics[metric_key] = value
                    experiment.metric_sources[metric_key] = item.path
            experiment.references.extend({**reference, "inherited": True} if item.id in shared else dict(reference)
                                         for reference in metadata.get("references", []))
            if item.kind == "script" and item.sha256:
                experiment.script_sources.setdefault(item.sha256, []).append(item.path)
            for key, digest in metadata.get("script_hashes", {}).items():
                experiment.script_sources.setdefault(digest, []).append(f"{item.path}#{key} (declared)")
            for key, entry in metadata.get("times", {}).items():
                if item.id in shared:
                    continue
                old = getattr(experiment, key)
                value = entry["value"]
                if old is None or (key == "started" and value < old) or (key == "finished" and value > old):
                    setattr(experiment, key, value)
                experiment.time_sources.append({"artifact": item.path, "field": entry["field"], "type": key, "value": value})
            experiment.warnings.extend(f"{item.path}: {message}" for message in item.warnings)
        experiment.script_hashes = sorted(experiment.script_sources)
        for role, sources in experiment.role_sources.items():
            values = {source["value"] for source in sources}
            if len(values) == 1:
                experiment.roles[role] = next(iter(values))
            else:
                experiment.warnings.append(f"Conflicting {role} declarations; role left unknown.")
        experiment.warnings = sorted(set(experiment.warnings))
        checkpoint_metrics([item for item in members if item.id not in shared])
    # Keep directory membership separate from model lineage. Checkpoints retain
    # one owner; collection totals are summaries, never additional files.
    by_path = {experiment.path: experiment for experiment in experiments}
    for experiment in experiments:
        for folder in [PurePosixPath(experiment.path), *PurePosixPath(experiment.path).parents]:
            parent = by_path.get(str(folder))
            if parent:
                parent.subtree_checkpoint_count += experiment.checkpoint_count
                parent.subtree_checkpoint_bytes += experiment.checkpoint_bytes
    for experiment in experiments:
        if experiment.checkpoint_count == 0 and experiment.subtree_checkpoint_count:
            experiment.kind = "collection"
    collections = {experiment.path: experiment for experiment in experiments if experiment.kind == "collection"}
    for experiment in sorted(experiments, key=lambda item: item.path):
        for ancestor in PurePosixPath(experiment.path).parents:
            parent = collections.get(str(ancestor))
            if parent:
                experiment.parent_id = parent.id
                parent.children.append(experiment.id)
                break
    # A sidecar can describe a loose checkpoint even when its run is unknown.
    candidates = {experiment.id for experiment in experiments if experiment.kind == "checkpoint_candidate"}
    folders = {str(PurePosixPath(item.path).parent) for item in artifacts if ownership.get(item.id) in candidates}
    for folder in folders:
        checkpoint_metrics([item for item in by_dir[folder] if item.id not in ownership or ownership[item.id] in candidates])
    for item in artifacts:
        if item.metadata.get("metric_conflicts") and item.id in ownership:
            owner = next(exp for exp in experiments if exp.id == ownership[item.id])
            owner.warnings.extend(f"{item.path}: {note}" for note in item.warnings if "Metric records disagree" in note)
        item.metadata.pop("_metric_rows", None)
        item.metadata.pop("_history_complete", None)
    return sorted(experiments, key=lambda item: item.path), ownership
