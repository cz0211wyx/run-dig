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
    for key, value in experiment.config.items():
        lower = key.lower()
        if any(word in lower for word in ("sha256", "hash", "checkpoint", "references", "source", "output", "device", "gpu", "version", "time", "seed", "samples", "counts", "weights")):
            continue
        if isinstance(value, (str, int, float, bool)) and value != "[REDACTED]" and not str(value).startswith("[omitted"):
            result[key] = value
    return result


def training_seed(experiment):
    for key in ("train.seed", "training.seed", "seed", "config.seed"):
        if key in experiment.config:
            return experiment.config[key]
    return None


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
        value = normalized(reference["value"])
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
            if kind == "references":
                caveats.append("A checkpoint reference can mean evaluation, teacher use, or initialization; it does not establish fine-tuning.")
            add(edge(source.id, experiment.id, kind, confidence,
                     [{"signal": method, "artifact": reference["artifact"], "field": reference["field"], "value": reference["value"], "matched_artifacts": files}], caveats=caveats))

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
        evidence = [{"signal": "configuration agreement", "matched_fields": matched, "score": round(config_score, 3),
                     "sources": {key: [left.config_sources.get(key, []), right.config_sources.get(key, [])] for key in matched}},
                    {"signal": "source hash overlap", "shared_hashes": shared_scripts, "score": round(script_score, 3)},
                    {"signal": "weak name token overlap", "tokens": sorted(names_a & names_b), "score": round(name_score, 3)}]
        item = edge(left_id, right_id, "similar_to", min(score, .85), evidence, directed=False,
                    caveats=["Similarity is not ancestry. Scores are heuristic, not calibrated probabilities."], changes=changes)
        # Mtimes never orient an edge. Only strong multi-signal similarity plus explicit run times can suggest chronology.
        earlier, later = (left, right)
        if right.finished is not None and left.started is not None and right.finished <= left.started:
            earlier, later = right, left
        seeds = training_seed(earlier), training_seed(later)
        seed_conflict = all(value is not None for value in seeds) and seeds[0] != seeds[1]
        if score >= .8 and shared_scripts and changes and not seed_conflict and earlier.finished is not None and later.started is not None and earlier.finished <= later.started:
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
