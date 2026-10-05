# MVP design and assumptions

## Starting point and modules

The original checkpoint scanner, sizes, sorting, and filtering are retained in
`runshelf/legacy.py`; `run_shelf.py` remains compatible. No PyTorch dependency is introduced.

| Module | Responsibility |
| --- | --- |
| `scanner.py` | Read-only inventory, exclusions, bounded hashing, filesystem errors |
| `extract.py` | Bounded parsers, safe YAML, static Python AST, references |
| `identify.py` | Run candidates, artifact ownership, provenance and conflicts |
| `lineage.py` | Explicit references, content identity, similarity, tentative chronology |
| `models.py` | Artifact, Experiment, Edge, Report JSON models |
| `pipeline.py` | Orchestration, statistics, assumptions |
| `report.py`, `viewer.html` | JSON and self-contained interactive graph |
| `cli.py`, `legacy.py` | New scan and preserved listing commands |

## Data model

`schema_version: "1.0"` describes the JSON contract. Artifacts include path-based IDs,
kind, exact bytes, filesystem time, optional SHA-256, extracted fields and warnings.
Experiments include artifact IDs, identification confidence, config/metric sources,
explicit timestamps, reference resolution, and uncertainty notes. IDs are stable
across rescans or moving a whole tree, but not renaming a run.

Edges contain source/target IDs, `kind`, `directed`, `confidence`, `evidence`, `caveats`,
and configuration changes. Nodes may have multiple parents or remain isolated.
Directed cycles are retained and flagged rather than silently deleting contradictory
evidence. This is a general graph, not a forced tree.

## Experiment boundaries

Colocated config + checkpoints (or training config + numeric logs/metrics) anchors
a candidate. Explicit run times plus metrics can also anchor one. Conventional
support folders (`checkpoints`, `source`, `configs`, `.hydra`) may attach to the
nearest anchor. A separately anchored child always wins.

Checkpoint + numeric training history can anchor a run without a config (confidence 0.70).
Arbitrarily named JSON containing multiple training hyperparameters and YAML/INI/TOML
files can supply config evidence. Launcher logs for multiple child runs do not make
their parent a single experiment.

No naming template is required. Unknown nested folders are not swallowed by a parent.
Every unowned checkpoint becomes a separate candidate with identification confidence
0.30; several candidates may actually be epochs of one experiment. Other unowned
files remain inventoried and counted. Config conflicts preserve all values and sources;
the first config in lexical path order wins for comparisons. Metrics retain their source
filename. JSONL/CSV metrics are the last parsed record, not necessarily the best/final epoch.

## Relationship rules

| Type | Direction | Evidence and score |
| --- | --- | --- |
| `derived_from` | Input → run | Explicit resume/init/parent: 0.90 resolved path, 0.98 with verified hash |
| `references` | Input → consumer | Checkpoint/teacher use; same scores, no claim of fine-tuning |
| `same_artifact` | Undirected | Actual SHA-256 of nonempty bytes: 1.00; copy direction unknown |
| `similar_to` | Undirected | Config/code/name agreement; capped at 0.85 |
| `possible_derivation` | Earlier → later | Strong similarity + explicit run timestamps; capped at 0.75 |

Relocated suffix resolution scores 0.78; static script references cap at 0.65 because
execution is unproven. Missing/ambiguous targets produce no edge. Declared hashes are
verified when bytes were hashed; mismatches remain unresolved. Conflicting timestamps
or directed cycles cap confidence at 0.60 and add caveats.

Similarity = `0.65 × config agreement + 0.25 × source-hash Jaccard + 0.10 × name-token Jaccard`.
Config agreement counts equal values over the union of eligible keys; volatile fields
(seed, hardware, hashes, outputs, timestamps) are excluded. At least three matching fields
are required. Generic names (`final`, `new`, `folder`, revision digits) provide no signal.
Shared config values/source hashes generate comparison candidates; names alone never do.
Default threshold: 0.60, at most three similarity alternatives per node.

Possible derivation also requires score ≥ 0.80, shared source hashes, and an explicit
source finish no later than target start. It remains a hypothesis: runs could be independent.
There must also be a configuration change beyond volatile fields, and known training
seeds must not conflict. Sequential seed repetitions therefore stay similarity edges.
Filesystem mtimes never orient inference. Identical files form a spanning star per hash
instead of a quadratic clique. Scores are heuristics, not calibrated probabilities.

## Limits and safety

- 2 MB text/file, 2,000 log records, 1,000 flattened fields, 12 nesting levels by default.
- Checkpoint hashing: ≤64 MB/file, ≤512 MB total by default; unhashed files are counted.
- 100,000 visited files and 200,000 similarity pairs by default; limits are reported.
- Dependency/cache directories (including `.work`) and report output are excluded.
- No unpickling, experiment code execution, symlink traversal, or archive expansion.
- TensorBoard events are inventoried only. YAML uses `safe_load`.
- Embedded report JSON is escaped; dynamic UI strings use DOM text nodes.
- No remote assets, telemetry, or model calls. Reports work directly from `file://`.
- Common secret-key fields are redacted, but reports still contain local paths and research metadata.
- Exit codes: 0 complete inventory (may have parser warnings), 1 partial scan, 2 bad arguments/I/O.

## Non-goals

No claim to recover scientific intent, architectures from weights, git history,
TensorBoard scalars, or relationships without evidence. No deletion/organization of
research files and no learned classifier. Human merge/correction workflows, parser
plugins, and very large log search are future extensions, not current capabilities.
