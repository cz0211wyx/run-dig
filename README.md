# run-dig

Recover ML experiment metadata and infer lineage from existing files. No prior tracking setup required.

Requires Python 3.10+.

```sh
pip install run-dig
run-dig scan ./experiments --open
```

Opens a local HTML report and saves `run-dig-report/index.html` and `report.json`.
Hover a graph node to highlight its links; click to see its files, metadata, and
relationship evidence. Arrows show recorded or possible sources; undirected links
show similarity or identical files.

[![Try the interactive demo](https://raw.githubusercontent.com/cz0211wyx/run-dig/main/examples/report-preview.png)](https://cz0211wyx.github.io/run-dig/#graph)

[Try the interactive demo](https://cz0211wyx.github.io/run-dig/#graph) — explore sample experiments in your browser, no installation needed.

Supports `.pt`, `.pth`, `.ckpt`, `.safetensors`, and model `.npz` files.
NPZ detection uses array names and file context to distinguish models from feature
caches; it does not verify model validity. Arrays are never loaded.

Use the terminal to list, inspect, or compare checkpoints:

```sh
run-dig list ./experiments --sort size --name best
run-dig inspect run-a/best.pt --root ./experiments
run-dig compare run-a/best.pt run-b/best.pt --root ./experiments
```

`--root` includes related runs; `--full` shows all fields and Git history for inspect and compare.
Scanning is read-only. Checkpoints and scripts are never executed.

With Git installed, scans also show local commit history and verify `git_commit`
or `git.commit` values recorded in run files. Current HEAD is only scan-time
context; Git ancestry does not prove checkpoint inheritance. No network access
is used. Pass `--no-git` to skip Git inspection.

Relationships and checkpoint metrics include evidence and confidence. Similarity
does not prove ancestry, and confidence is not a calibrated probability. Missing or
conflicting settings remain unknown; conflicting values keep their sources.
