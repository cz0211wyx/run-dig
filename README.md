# run-shelf

Scan messy ML folders to find checkpoints, recover experiment metadata, and trace possible relationships.

Python 3.10+.

```sh
python -m pip install -e .
run-shelf scan ./experiments --open
```

Switch between the experiment list and the full graph. Click an experiment to see
its sources, downstream experiments, and files. Similarity alone is not ancestry.
Results are saved to `run-shelf-report/index.html` and `report.json`.

To list checkpoints in the terminal:

```sh
run-shelf list ./experiments --sort size --name best
```
