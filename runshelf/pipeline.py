"""End-to-end analysis; source files are never modified."""

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .extract import extract
from .identify import identify
from .lineage import infer
from .models import Report
from .scanner import DEFAULT_EXCLUDES, inventory

ASSUMPTIONS = [
    "Experiments are evidence-backed candidates, not a guaranteed reconstruction of past runs.",
    "Unanchored checkpoints remain separate low-confidence candidates; filenames such as final_final2 do not establish ancestry.",
    "references means checkpoint use; derived_from requires an explicit resume/init/parent field. Neither is proof code actually executed.",
    "same_artifact is undirected content identity; similar_to is undirected similarity, not lineage.",
    "possible_derivation needs configuration/code similarity and timezone-aware run timestamps. Filesystem mtimes never orient inferred lineage.",
    "Confidence values are reproducible heuristic evidence scores, not calibrated probabilities.",
    "No checkpoint unpickling, script execution, archive expansion, or symlink traversal. TensorBoard files are inventoried only.",
    "Text, record, hash, file, and comparison budgets are bounded; omissions and parse failures are reported.",
    "Stable IDs depend on relative paths; moving the whole tree preserves IDs, renaming a run does not.",
]


def analyze(root, *, output=None, excludes=(), text_limit=2_000_000, hash_limit=64_000_000,
            hash_budget=512_000_000, max_files=100_000, similarity_threshold=.6, neighbors=3,
            max_pairs=200_000):
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
    edges, inference_warnings = infer(experiments, artifacts, ownership, similarity_threshold=similarity_threshold,
                                       neighbors=neighbors, max_pairs=max_pairs)
    warnings.extend(inference_warnings)
    statistics.update({
        "artifacts": len(artifacts), "artifact_kinds": dict(Counter(item.kind for item in artifacts)),
        "experiments": len(experiments), "anchored_experiments": sum(item.kind == "experiment" for item in experiments),
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
               "max_files": max_files, "similarity_threshold": similarity_threshold, "neighbors": neighbors,
               "max_pairs": max_pairs}
    return Report("1.0", __version__, str(root), datetime.now(timezone.utc).isoformat(), options,
                  statistics, experiments, edges, artifacts, warnings, ASSUMPTIONS)
