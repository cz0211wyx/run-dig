# run-shelf

**Experiment archaeology for messy ML folders.** Recover configurations, metrics,
checkpoint references, and possible experiment relationships in an offline lineage explorer.

## Quick start

Python 3.10+:

```sh
python -m pip install -e .
run-shelf scan ./experiments --open
```

This creates `run-shelf-report/index.html` and `report.json`. Open the HTML file
in any modern browser: search experiments, filter relationships by type/confidence,
and inspect the evidence behind each node or edge. No server or internet required.

If the command is not on your PATH, use `python -m runshelf scan ./experiments`.
On Windows, use your actual Python interpreter if `python` resolves to the Microsoft Store alias.

```sh
run-shelf scan ./experiments -o ./reports/my-project --exclude backups
run-shelf scan ./experiments --hash-max-mb 0   # skip checkpoint hashing
run-shelf scan --help
```

## What it recovers

- Checkpoints (`.pt`, `.pth`, `.ckpt`, `.safetensors`), configurations, logs, metrics, and script snapshots.
- JSON, JSONL, YAML, INI, CSV/TSV, common text logs, and TOML on Python 3.11+.
- Config/metric values with source files; run timestamps when explicitly recorded.
- Explicit resume/input references, identical checkpoint bytes, and configuration/code similarity.

Relations form a **graph**, including multiple inputs, alternatives, and isolated nodes.
Checkpoint references, shared files, similarity, and tentative ancestry are separate edge types.
Every edge includes confidence and evidence. Confidence is a heuristic score, not a probability.

Names like `final_final2.pt` and `new folder` are not treated as proof of lineage.
Loose checkpoints stay uncertain candidates; copied files do not imply a direction.
Filesystem modification times never establish inferred ancestry.

## Original checkpoint listing

```sh
run-shelf list ./experiments --sort size --name best
python run_shelf.py ./experiments --sort time
```

The existing `name / size / modified / path` output is preserved.
`--sort name|time|size` orders by name, newest first, or largest first;
`--name TEXT` matches a filename substring, ignoring case.

## Try the demo

```sh
python examples/create_demo.py
run-shelf scan ./demo-experiments -o ./reports/demo --open
python -m unittest discover -s tests -v
```

The demo includes arbitrary names, a resumed run, a separate seed, a renamed copy,
loose checkpoints, broken metadata, and a missing source. Its checkpoint bytes are synthetic.

## Limits

Source files are read only. Models are never unpickled, scripts never executed,
symlinks never followed, and archives never expanded. TensorBoard event files are
inventoried but not decoded. Default exclusions include dependency/cache directories.
Large files and comparisons have configurable limits; omissions appear in Scan notes.

Directory boundaries are guesses. Config-only runs without supporting artifacts may
remain unassigned, and external or ambiguous references stay unresolved. A graph can
legitimately contain no ancestry edges. Reports contain local paths and experiment
metadata; common secret keys are redacted, but review a report before sharing it.

See [design and confidence rules](docs/design.md) and [validation](docs/validation.md).
