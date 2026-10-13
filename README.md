# run-dig

Scan messy ML folders to find checkpoints, recover experiment metadata, and trace possible relationships.

Requires Python 3.10+ and Git.

```sh
pip install git+https://github.com/cz0211wyx/run-dig.git
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

`--root` includes related runs; `--full` shows all fields for inspect and compare.
Scanning is read-only. Checkpoints and scripts are never executed.

Relationships and checkpoint metrics include evidence and confidence. Similarity
does not prove ancestry, and confidence is not a calibrated probability. Missing or
conflicting settings remain unknown; conflicting values keep their sources.
