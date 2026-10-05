"""Detect evidence-backed runs; keep isolated checkpoints as uncertain candidates."""

from collections import defaultdict
from pathlib import PurePosixPath

from .models import Experiment, stable_id

ARTIFACT_FOLDERS = {"checkpoints", "checkpoint", "weights", "ckpts", "models"}
SUPPORT_FOLDERS = ARTIFACT_FOLDERS | {"logs", "source", "src", "scripts", "configs", ".hydra"}


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
        if (config and ("checkpoint" in kinds or child_checkpoint or (training and metrics))) or (state and metrics) or ("checkpoint" in kinds and metrics):
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

    index = {item.id: item for item in artifacts}
    for experiment in experiments:
        members = [index[key] for key in experiment.artifacts]
        # Config facts take precedence over summaries; conflicts keep every source.
        members.sort(key=lambda item: (item.kind != "config", item.path))
        for item in members:
            experiment.modified = max(experiment.modified, item.modified)
            if item.kind == "checkpoint":
                experiment.checkpoint_count += 1
                experiment.checkpoint_bytes += item.size
            metadata = item.metadata
            if item.kind == "config":
                for key, value in metadata.get("fields", {}).items():
                    if value == "[REDACTED]":
                        continue
                    source = {"artifact": item.path, "value": value}
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
            experiment.references.extend(metadata.get("references", []))
            if item.kind == "script" and item.sha256:
                experiment.script_sources.setdefault(item.sha256, []).append(item.path)
            for key, digest in metadata.get("script_hashes", {}).items():
                experiment.script_sources.setdefault(digest, []).append(f"{item.path}#{key} (declared)")
            for key, entry in metadata.get("times", {}).items():
                old = getattr(experiment, key)
                value = entry["value"]
                if old is None or (key == "started" and value < old) or (key == "finished" and value > old):
                    setattr(experiment, key, value)
                experiment.time_sources.append({"artifact": item.path, "field": entry["field"], "type": key, "value": value})
            experiment.warnings.extend(f"{item.path}: {message}" for message in item.warnings)
        experiment.script_hashes = sorted(experiment.script_sources)
        experiment.warnings = sorted(set(experiment.warnings))
    return sorted(experiments, key=lambda item: item.path), ownership
