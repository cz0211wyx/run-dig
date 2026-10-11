"""Typed graph inference; similarity and shared bytes do not imply ancestry."""

import posixpath
import re
from collections import defaultdict
from itertools import combinations
from pathlib import PurePosixPath

from .models import Edge, stable_id


def edge(source, target, kind, confidence, evidence, *, directed=True, caveats=None, changes=None):
    if not directed:
        source, target = sorted((source, target))
    return Edge(stable_id("edge", f"{source}|{kind}|{target}"), source, target, kind, directed,
                round(confidence, 3), evidence, caveats or [], changes or {})


def normalized(value):
    return posixpath.normpath(value.replace("\\", "/")).removeprefix("./")


def comparable_config(experiment):
    result = {}
    if experiment.kind == "collection":
        return result
    for key, value in experiment.config.items():
        lower = key.lower()
        if any(word in lower for word in ("sha256", "hash", "checkpoint", "references", "source", "output", "device", "gpu", "version", "time", "seed", "samples", "counts", "weights")):
            continue
        if isinstance(value, (str, int, float, bool)) and value != "[REDACTED]" and not str(value).startswith("[omitted"):
            if any(source["value"] != value for source in experiment.config_sources.get(key, [])):
                continue
            result[key] = value
    return result


def training_seed(experiment):
    for key in ("train.seed", "training.seed", "seed", "config.seed"):
        if key in experiment.config:
            value = experiment.config[key]
            return value if not any(source["value"] != value for source in experiment.config_sources.get(key, [])) else None
    return None


def local_field(experiment, key):
    return any(not source.get("inherited") for source in experiment.config_sources.get(key, []))


def inherited_configs(experiment):
    return {source["artifact"] for sources in experiment.config_sources.values() for source in sources if source.get("inherited")}


def local_scripts(experiment):
    inherited = inherited_configs(experiment)
    return {digest for digest, sources in experiment.script_sources.items()
            if any(source.split("#", 1)[0] not in inherited for source in sources)}


def similarity_details(left, right, changes, matched, shared_scripts, score):
    """Describe observed differences; borrowed configuration is weaker evidence."""
    roles = set(left.roles) & set(right.roles)
    ambiguous_roles = (set(left.role_sources) - set(left.roles)) | (set(right.role_sources) - set(right.roles))
    role_changes = {key: {"source": left.roles[key], "target": right.roles[key]}
                    for key in sorted(roles) if left.roles[key] != right.roles[key]}
    shared_parent = inherited_configs(left) & inherited_configs(right)
    same_collection = left.parent_id is not None and left.parent_id == right.parent_id
    local_matches = [key for key in matched if local_field(left, key) and local_field(right, key)]
    shared_local_scripts = set(shared_scripts) & local_scripts(left) & local_scripts(right)
    independent = len(local_matches) >= 3 or bool(shared_local_scripts) or bool(roles and not role_changes)
    if role_changes and not (same_collection or shared_parent):
        return None  # Different components across batches are not model variants.
    if not independent and not shared_parent:
        return None
    confidence = min(score, .85)
    caveats = []
    if len(local_matches) < len(matched) / 2:
        confidence = min(confidence, .65)
        caveats.append("Most matching settings come from inferred parent configuration; component applicability is not verified.")
    if role_changes:
        description = "Different recorded roles in the same collection" if same_collection else "Different recorded roles sharing parent configuration"
        confidence = min(confidence, .55)
    elif ambiguous_roles:
        description = "Shared configuration; conflicting role declarations"
        confidence = min(confidence, .5)
        caveats.append("Conflicting role declarations prevent identifying a component or seed variant.")
    elif not independent:
        description = "Shared parent configuration; run similarity unverified"
        confidence = min(confidence, .5)
    else:
        seed_a, seed_b = training_seed(left), training_seed(right)
        seed_changed = seed_a is not None and seed_b is not None and seed_a != seed_b
        if not changes and seed_changed:
            if local_scripts(left) and local_scripts(right) and local_scripts(left) != local_scripts(right):
                description = f"Recorded seed differs ({seed_a} vs {seed_b}); code snapshots also differ"
            else:
                description = f"Recorded settings match except seed ({seed_a} vs {seed_b})"
        elif changes:
            structural = [key for key in changes if re.search(r"(^|[._])(architecture|model|model_type|module|layers|num_layers|hidden_dim|hidden_size|head_dim)([._]|$)", key, re.I)]
            dataset = [key for key in changes if re.search(r"(^|[._])(dataset|data|split)([._]|$)", key, re.I)]
            keys = structural or dataset or list(changes)
            category = "Model configuration differs" if structural else "Dataset configuration differs" if dataset else "Training settings differ"
            description = category + ": " + ", ".join(keys[:3]) + (f" (+{len(keys) - 3} more)" if len(keys) > 3 else "")
            if seed_changed:
                description += f"; seed {seed_a} vs {seed_b}"
        else:
            description = "Matching recorded configuration" + (" and code" if shared_local_scripts else "")
        if seed_changed:
            changes["seed"] = {"source": seed_a, "target": seed_b}
    for key, value in role_changes.items():
        changes["role." + key] = value
    facts = [{"signal": "run-local configuration support", "matched_fields": local_matches,
              "inherited_match_count": len(matched) - len(local_matches)}]
    for side, experiment in (("source", left), ("target", right)):
        for role in sorted(experiment.role_sources):
            facts.extend({"signal": f"recorded {side} role", **source} for source in experiment.role_sources.get(role, []))
    if shared_parent:
        facts.append({"signal": "shared parent configuration", "artifacts": sorted(shared_parent)})
    return description, confidence, caveats, facts, bool(role_changes or ambiguous_roles), len(local_matches)


def name_tokens(name):
    # Generic names and revision numbers contribute no relationship evidence.
    tokens = set(re.findall(r"[a-z]+", name.lower()))
    return tokens - {"final", "new", "folder", "best", "last", "run", "runs", "pt", "pth", "ckpt", "s", "seed", "v", "copy", "checkpoint"}


def infer(experiments, artifacts, ownership, *, similarity_threshold=.6, max_pairs=200_000):
    edges, warnings = {}, []
    exp_index = {item.id: item for item in experiments}
    file_index = {item.path: item for item in artifacts}
    checkpoints = [item for item in artifacts if item.kind == "checkpoint"]
    hash_index = defaultdict(list)
    for item in checkpoints:
        if item.sha256 and item.size > 0:
            hash_index[item.sha256].append(item)

    def add(item):
        existing = edges.get(item.id)
        if existing:
            existing.evidence.extend(e for e in item.evidence if e not in existing.evidence)
            existing.confidence = max(existing.confidence, item.confidence)
        else:
            edges[item.id] = item

    def resolve(reference):
        value = normalized(reference.get("resolved_value", reference["value"]))
        source_dir = str(PurePosixPath(reference["artifact"]).parent)
        bases = [PurePosixPath(source_dir), *PurePosixPath(source_dir).parents]
        # Original relative context takes precedence; suffix matching handles moved trees.
        paths = [normalized(str(base / value)) for base in bases] + [value]
        matches = []
        for candidate in dict.fromkeys(paths):
            if candidate in file_index and file_index[candidate].kind == "checkpoint":
                matches = [file_index[candidate]]
                break
            runs = [exp for exp in experiments if exp.path == candidate]
            if reference["kind"] == "derived_from" and len(runs) == 1:
                return [runs[0].id], "explicit run path", []
        method = "explicit checkpoint path"
        if not matches and "/" in value:
            parts = value.strip("/").split("/")
            for length in range(len(parts), 1, -1):
                suffix = "/".join(parts[-length:])
                matches = [item for item in checkpoints if item.path == suffix or item.path.endswith("/" + suffix)]
                if matches:
                    method = "relocated checkpoint path suffix"
                    break
        digest = reference.get("sha256")
        if digest:
            verified = [item for item in matches if item.sha256 == digest]
            mismatched = [item for item in matches if item.sha256 and item.sha256 != digest]
            if verified:
                matches = verified
                method += " + verified SHA-256"
            elif mismatched:
                return [], "hash mismatch", [item.path for item in mismatched]
            elif not matches:
                matches = hash_index.get(digest, [])
                method = "verified SHA-256 (path unavailable)"
        owners = sorted({ownership[item.id] for item in matches if item.id in ownership})
        return owners, method, [item.path for item in matches]

    for experiment in experiments:
        for reference in experiment.references:
            targets, method, files = resolve(reference)
            if targets == [experiment.id]:
                reference["resolution"] = "self"
                continue
            reference["candidates"] = targets
            reference["resolution"] = "resolved" if len(targets) == 1 else "ambiguous" if targets else "unresolved"
            if len(targets) != 1:
                experiment.warnings.append(f"{reference['resolution'].capitalize()} reference: {reference['value']} ({method}).")
                continue
            source = exp_index[targets[0]]
            caveats = []
            confidence = .98 if "verified SHA-256" in method else .9 if method.startswith("explicit") else .78
            if reference["role"] == "script":
                confidence = min(confidence, .65)
                caveats.append("Static source reference; execution of this code is not established.")
            if source.finished is not None and experiment.started is not None and source.finished > experiment.started:
                caveats.append("Run timestamps conflict with the reference direction; inspect evidence.")
                confidence = min(confidence, .6)
            kind = reference["kind"]
            if reference.get("inherited"):
                confidence = min(confidence, .65)
                caveats.append("Shared parent configuration; its applicability to this run is inferred.")
                if kind == "derived_from":
                    kind = "possible_derivation"
            if kind == "references":
                caveats.append("A checkpoint reference can mean evaluation, teacher use, or initialization; it does not establish fine-tuning.")
            add(edge(source.id, experiment.id, kind, confidence,
                     [{"signal": method, "artifact": reference["artifact"], "field": reference["field"], "value": reference["value"], "matched_artifacts": files}], caveats=caveats))

    # Parent manifests can bind stages by paths even without checkpoint hashes.
    for artifact in artifacts:
        for binding in artifact.metadata.get("record_links", []):
            def resolve_stage(value):
                return resolve({"value": value, "artifact": artifact.path, "kind": "references"})
            sources, source_method, source_files = resolve_stage(binding["source"])
            targets, target_method, target_files = resolve_stage(binding["target"])
            if len(sources) == len(targets) == 1 and sources != targets:
                add(edge(sources[0], targets[0], binding["kind"], .9 if source_method.startswith("explicit") and target_method.startswith("explicit") else .78,
                         [{"signal": "explicit stage dependency", "artifact": artifact.path, "field": binding["field"],
                           "value": binding["source"], "target_checkpoint": binding["target"], "matched_artifacts": source_files + target_files}]))
            elif not sources or not targets or len(sources) > 1 or len(targets) > 1:
                warnings.append({"kind": "inference", "path": artifact.path, "message": f"Unresolved or ambiguous stage dependency: {binding['source']} -> {binding['target']}."})

    # A parent manifest may describe multiple child runs without belonging to any
    # one of them. Bind both ends by checkpoint bytes, never by directory names.
    for artifact in artifacts:
        for binding in artifact.metadata.get("component_links", []):
            source_files = hash_index.get(binding["source_sha256"], [])
            target_files = hash_index.get(binding["target_sha256"], [])
            sources = {ownership[item.id] for item in source_files if item.id in ownership}
            targets = {ownership[item.id] for item in target_files if item.id in ownership}
            if len(sources) != 1 or len(targets) != 1:
                warnings.append({"kind": "inference", "path": artifact.path,
                                 "message": f"Component {binding['component']}: checkpoint hashes are unavailable or match multiple runs; source link left unresolved."})
                continue
            source, target = next(iter(sources)), next(iter(targets))
            if source != target:
                add(edge(source, target, "references", .85,
                         [{"signal": "recorded frozen component; source and target checkpoint hashes verified",
                           "artifact": artifact.path, **binding,
                           "matched_artifacts": [item.path for item in source_files + target_files]}],
                         caveats=["Component mapping is declared in metadata; model weights were not independently compared."]))

    for digest, items in sorted(hash_index.items()):
        owners = sorted({ownership[item.id] for item in items if item.id in ownership})
        # A star encodes an equivalence set without a quadratic clique.
        for target in owners[1:]:
            source = owners[0]
            paths = [item.path for item in items if ownership.get(item.id) in {source, target}]
            add(edge(source, target, "same_artifact", 1.0,
                     [{"signal": "identical checkpoint bytes", "sha256": digest, "artifacts": paths}], directed=False,
                     caveats=["Shared checkpoint bytes do not identify copy direction or prove the runs are identical."]))

    configs = {exp.id: comparable_config(exp) for exp in experiments}
    # Candidate blocks require actual matching config values or source hashes, never a name alone.
    blocks = defaultdict(list)
    for exp in experiments:
        if exp.kind == "collection":
            continue
        for key, value in configs[exp.id].items():
            blocks[("config", key, str(value))].append(exp.id)
        for digest in exp.script_hashes:
            blocks[("script", digest)].append(exp.id)
    pairs = set()
    capped = False
    for key in sorted(blocks):
        for pair in combinations(sorted(set(blocks[key])), 2):
            pairs.add(pair)
            if len(pairs) >= max_pairs:
                capped = True
                break
        if capped:
            break
    if capped:
        warnings.append({"kind": "limit", "path": ".", "message": f"Similarity candidate budget ({max_pairs}) reached. Explicit references remain complete; some similarities were not compared."})

    explicit_pairs = {frozenset((item.source, item.target)) for item in edges.values() if item.directed}
    for left_id, right_id in sorted(pairs):
        left, right = exp_index[left_id], exp_index[right_id]
        if frozenset((left_id, right_id)) in explicit_pairs:
            continue
        a, b = configs[left_id], configs[right_id]
        common = set(a) & set(b)
        matched = sorted(key for key in common if a[key] == b[key])
        config_score = len(matched) / max(len(set(a) | set(b)), 1)
        shared_scripts = sorted(set(left.script_hashes) & set(right.script_hashes))
        script_score = len(shared_scripts) / max(len(set(left.script_hashes) | set(right.script_hashes)), 1)
        names_a, names_b = name_tokens(left.name), name_tokens(right.name)
        name_score = len(names_a & names_b) / max(len(names_a | names_b), 1)
        score = .65 * config_score + .25 * script_score + .1 * name_score
        if len(matched) < 3 or score < similarity_threshold:
            continue
        changes = {key: {"source": a.get(key), "target": b.get(key)} for key in sorted(set(a) | set(b)) if a.get(key) != b.get(key)}
        config_changed = bool(changes)
        details = similarity_details(left, right, changes, matched, shared_scripts, score)
        if details is None:
            continue
        description, confidence, caveats, facts, role_conflict, local_matches = details
        evidence = [{"signal": "configuration agreement", "matched_fields": matched, "score": round(config_score, 3),
                     "sources": {key: [left.config_sources.get(key, []), right.config_sources.get(key, [])] for key in matched}},
                    {"signal": "source hash overlap", "shared_hashes": shared_scripts, "score": round(script_score, 3)},
                    {"signal": "weak name token overlap", "tokens": sorted(names_a & names_b), "score": round(name_score, 3)}]
        evidence.extend(facts)
        item = edge(left_id, right_id, "similar_to", confidence, evidence, directed=False,
                    caveats=["Similarity is not ancestry. Scores are heuristic, not calibrated probabilities.", *caveats], changes=changes)
        item.description = description
        # Mtimes never orient an edge. Only strong multi-signal similarity plus explicit run times can suggest chronology.
        earlier, later = (left, right)
        if right.finished is not None and left.started is not None and right.finished <= left.started:
            earlier, later = right, left
        seeds = training_seed(earlier), training_seed(later)
        seed_conflict = all(value is not None for value in seeds) and seeds[0] != seeds[1]
        if score >= .8 and shared_scripts and config_changed and local_matches >= 3 and not role_conflict and not seed_conflict and earlier.finished is not None and later.started is not None and earlier.finished <= later.started:
            changes = {key: {"source": configs[earlier.id].get(key), "target": configs[later.id].get(key)} for key in sorted(set(a) | set(b)) if a.get(key) != b.get(key)}
            item = edge(earlier.id, later.id, "possible_derivation", min(score, .75), evidence + [
                {"signal": "explicit run chronology", "source_finished": earlier.finished, "target_started": later.started,
                 "time_sources": earlier.time_sources + later.time_sources}],
                caveats=["Chronology and shared configuration/code suggest an alternative, not proven ancestry; runs may be independent."], changes=changes)
        # Keep every supported relationship. Presentation must not prune evidence.
        add(item)

    adjacency = defaultdict(set)
    for item in edges.values():
        if item.directed:
            adjacency[item.source].add(item.target)
    for item in edges.values():
        if not item.directed:
            continue
        pending, seen = [item.target], set()
        while pending:
            current = pending.pop()
            if current == item.source:
                item.caveats.append("Directed cycle detected; references conflict or are not chronological lineage.")
                item.confidence = min(item.confidence, .6)
                break
            if current not in seen:
                seen.add(current)
                pending.extend(adjacency[current] - seen)
    return sorted(edges.values(), key=lambda item: item.id), warnings
