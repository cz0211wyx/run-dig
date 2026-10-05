# run-shelf

Recursively find `.pt`, `.pth`, and `.ckpt` files with their paths, sizes, and last modification times.

Python 3. No dependencies.

```sh
python run_shelf.py "path/to/experiments"
```

Example output:

```text
Found 2 checkpoints, total size: 1.5 GB

Path                                      Size       Last modified
/home/user/experiments/run-01/best.pt       20MB       2026-09-23 22:42:40
/home/user/experiments/run-02/best.ckpt     1.5GB      2026-09-24 09:30:00
```
