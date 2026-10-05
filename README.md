# run-shelf

Scan messy ML folders to find checkpoints, recover experiment metadata, and trace possible relationships.

Python 3.10+.

```sh
python -m pip install -e .
run-shelf scan ./experiments --open
```

The report opens as an experiment list. Click an experiment to see its files,
configuration, metrics, and related experiments. Uncertain links are labeled.
Results are saved to `run-shelf-report/index.html` and `report.json`.

To list checkpoints in the terminal:

```sh
run-shelf list ./experiments --sort size --name best
```
