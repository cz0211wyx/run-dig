# run-shelf

Scan messy ML folders to find checkpoints, recover experiment metadata, and trace possible relationships.

Python 3.10+.

```sh
python -m pip install -e .
run-shelf scan ./experiments --open
```

Switch between the experiment list and the full graph. Click an experiment to see
its sources, downstream experiments, and files. Parent collections show their child
runs and total checkpoints; similarity links explain recorded differences, not ancestry.
Results are saved to `run-shelf-report/index.html` and `report.json`.

To list checkpoints in the terminal:

```sh
run-shelf list ./experiments --sort size --name best
```

Inspect one checkpoint in the terminal (`--root` includes other runs; `--full` shows all fields):

```sh
run-shelf inspect run-a/best.pt --root ./experiments
run-shelf compare run-a/best.pt run-b/best.pt --root ./experiments
```

Checkpoint metrics include their source and confidence. Filename-based matches are
estimates; run summaries are shown separately. Model files are never loaded.
