# run-dig

Scan messy ML folders to find checkpoints, recover experiment metadata, and trace possible relationships.

Requires Python 3.10+.

```sh
pip install run-dig
run-dig scan ./experiments --open
```

Opens a local HTML report and saves `run-dig-report/index.html` and `report.json`.
Hover a graph node to highlight its links; click to see its files, metadata, and
relationship evidence. Arrows show recorded or possible sources; undirected links
show similarity or identical files.

![Demo HTML report with an experiment and its relationships highlighted](https://raw.githubusercontent.com/cz0211wyx/run-dig/main/examples/report-preview.png)

*Example report using demo data, with one experiment highlighted.*

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
