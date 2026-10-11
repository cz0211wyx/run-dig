"""Create a synthetic archaeology dataset without requiring PyTorch."""

import argparse
import json
from pathlib import Path


def create_demo(root):
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise ValueError("Choose an empty demo directory.")
    root.mkdir(parents=True, exist_ok=True)

    def write(path, value):
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, bytes):
            target.write_bytes(value)
        else:
            target.write_text(json.dumps(value, indent=2) if isinstance(value, dict) else value, encoding="utf-8")

    config = {"model": "small-transformer", "dataset": "synthetic", "lr": .001,
              "batch_size": 32, "epochs": 10, "dropout": .1, "seed": 42}
    write("new folder/config.json", config)
    write("new folder/checkpoints/final_final2.pt", b"synthetic checkpoint A; not a real model")
    write("new folder/history.jsonl", '{"epoch":1,"loss":1.2}\n{"epoch":2,"loss":0.8}\n')
    write("new folder/state.json", {"started_at_utc": "2026-01-01T10:00:00Z", "finished_at_utc": "2026-01-01T11:00:00Z"})
    write("new folder/source/train.py", "# Static snapshot; never executed.\nMODEL = 'small-transformer'\n")
    write("retry after lunch/config.json", {**config, "lr": .0003, "resume_from": "../new folder/checkpoints/final_final2.pt"})
    write("retry after lunch/best.ckpt", b"synthetic checkpoint B; not a real model")
    write("retry after lunch/metrics.csv", "epoch,loss,accuracy,checkpoint\n1,0.7,0.68,\n2,0.4,0.75,best.ckpt\n")
    write("retry after lunch/state.json", {"started_at_utc": "2026-01-02T10:00:00Z"})
    write("different seed/config.json", {**config, "seed": 123})
    write("different seed/last.pth", b"synthetic independent run C")
    write("different seed/history.jsonl", '{"epoch":1,"loss":1.1}\n')
    write("backup/renamed.pt", b"synthetic checkpoint A; not a real model")
    write("loose final.pt", b"synthetic checkpoint D; origin unknown")
    write("broken/config.json", "{ invalid json")
    write("broken/final.pt", b"synthetic checkpoint E; incomplete metadata")
    write("missing origin/config.json", {**config, "init_checkpoint": "/old/server/unavailable/best.pt"})
    write("missing origin/last.pt", b"synthetic checkpoint F")
    write("notes.txt", "This dataset contains no real model weights or research data.\n")
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", default="demo-experiments", type=Path)
    args = parser.parse_args()
    try:
        print(f"Created demo at {create_demo(args.directory).resolve()}")
    except ValueError as error:
        parser.error(str(error))
