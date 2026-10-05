# Validation

Validated locally on Windows with Python 3.12.4 on 2026-10-05.
The GitHub Actions matrix is configured for Windows/Linux and Python 3.10/3.12;
local results below do not claim those remote jobs have completed.

## Automated checks

```sh
python -m unittest discover -s tests -v
```

35 tests: 34 passed, 1 skipped locally because creating a symlink requires a Windows
privilege. The symlink case runs where the OS permits it. Coverage includes:

- Existing listing, case-insensitive filtering, and size ordering.
- Arbitrary names, flat checkpoint collections, nested runs, missing configs,
  config subfolders, and launcher directories that must not become one run.
- Explicit resume and checkpoint references, verified/mismatched hashes,
  ambiguous moved paths, cycles, copies, missing targets, and source evidence.
- Similarity without timestamp assumptions; sequential seed repeats stay undirected.
- Malformed/truncated files, YAML, JSONL, CSV, config conflicts, redaction,
  resource budgets, unreadable subtrees, empty trees, and stable rescans.
- CLI/report generation, HTML data escaping, and refusing unrelated output folders.

An editable install and a built wheel were both exercised through the scan pipeline.
The wheel includes the HTML viewer; the scan works without importing the source checkout.

## Browser checks

`tests/browser_smoke.cjs` is an optional Playwright check. Run it with a Playwright
installation and Chromium, or set `RUN_SHELF_PLAYWRIGHT` and `RUN_SHELF_BROWSER` to local paths:

```sh
node tests/browser_smoke.cjs reports/demo/index.html reports/demo-preview.png
```

Both the 7-node synthetic report and the real 159-node report were opened directly
from disk in headless Microsoft Edge. Search, reset, node selection, neighborhood focus,
relationship filters, edge evidence, scan notes, JSON download, and a narrow viewport
passed. No JavaScript errors or remote HTTP requests occurred. Screenshots were reviewed.
Large graphs start with a connected neighborhood; the full node list remains searchable.
The full graph is available through Show all nodes. Rendering is capped at 250 visible
nodes with a notice; filtering does not remove nodes from the underlying JSON.

## Real research-directory pipeline

The user's existing research tree was analyzed with default limits. No research
files, experiment metadata, reports, or model weights are included in this repository.
Aggregate results:

| Measure | Result |
| --- | ---: |
| Files visited after dependency/cache exclusions | 5,295 |
| Relevant artifacts | 2,842 |
| Checkpoints | 353 (352 `.pt`, 1 `.safetensors`) |
| Checkpoint bytes | 436,815,065 |
| Anchored experiment candidates | 148 |
| Uncertain standalone checkpoint candidates | 11 |
| Checkpoint reference edges | 37 |
| Undirected similarity edges | 165 |
| Undirected identical-artifact edges | 11 |
| Unresolved extracted references | 0 |
| Unassigned supporting artifacts | 1,702 |
| File/record-limit extraction warnings | 177 |
| Final scan plus report generation | about 22.5 seconds |

An independent whole-tree inventory exactly matched checkpoint paths and sizes.
Every checkpoint belonged to exactly one node. Source checkpoint sizes and mtimes
were unchanged after scanning. All graph endpoints, confidence ranges, and evidence
records passed integrity checks.

The 177 warnings were 85 text-size limits and 92 record limits, mainly large report
tables. One large checkpoint exceeded the default hashing limit and remained unhashed.
The filesystem inventory completed; bounded extraction is explicitly visible in the report.
There were **no inferred ancestry edges** in the final real report: the evidence supported
references, duplicates, and similarity, not a confident historical training chain.

This validates coverage and execution, not scientific ground-truth lineage accuracy.
The synthetic fixture supplies known ancestry/copy relationships for rule tests, while
the real directory exposes missing metadata, copied reports, nested runs, and large files.
