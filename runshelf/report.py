"""Self-contained HTML and machine-readable JSON report export."""

import json
import os
from importlib.resources import files
from pathlib import Path

MARKER = ".run-shelf-output"


def validate_output(output):
    output = Path(output)
    if output.exists():
        if not output.is_dir():
            raise ValueError(f"Output is not a directory: {output}")
        if any(output.iterdir()) and not (output / MARKER).is_file():
            raise ValueError(f"Output directory is not an existing run-shelf report; choose an empty directory: {output}")


def write_report(report, output):
    output = Path(output)
    validate_output(output)
    data = report.to_dict()
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    # Embedded JSON is data even if a scanned filename contains </script>.
    embedded = payload.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    template = files("runshelf").joinpath("viewer.html").read_text(encoding="utf-8")
    html = template.replace("__REPORT_DATA__", embedded)
    output.mkdir(parents=True, exist_ok=True)
    for name, content in (("report.json", json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)), ("index.html", html)):
        temporary = output / (name + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, output / name)
    (output / MARKER).write_text("run-shelf report schema 1.0\n", encoding="utf-8")
    return output / "index.html"
