# run-shelf

Recursively find `.pt`, `.pth`, and `.ckpt` files with their paths, sizes, and last modification times.

Python 3. No dependencies.

```sh
python run_shelf.py "path/to/experiments"
python run_shelf.py "path/to/experiments" --sort size --name best
```

`--sort name|time|size`: name A-Z (default), newest first, or largest first.
`--name TEXT`: case-insensitive filename substring filter. The summary counts only matching files.

Example output:

```text
Found 2 checkpoints, total size: 1.5 GB

name       size  modified             path
best.ckpt  1.5GB  2026-09-24 09:30:00  /home/user/experiments/run-02/best.ckpt
best.pt     20MB  2026-09-23 22:42:40  /home/user/experiments/run-01/best.pt
```
