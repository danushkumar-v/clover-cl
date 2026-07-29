# CLOVER v2 analysis workspace

Post-processing only: reads `runs/*/`\ per-run CSVs and `R_matrix.npy` files
and produces tidy DataFrames, derived metrics, figures, and executed
notebooks. No training happens here.

## Setup

```
.venv\Scripts\python.exe -m pip install -r analysis/requirements.txt
```

A Jupyter kernel pointing at this repo's `.venv` must be registered once
(the notebooks' metadata expects it by name):

```
.venv\Scripts\python.exe -m ipykernel install --user --name clover-cl-analysis --display-name "clover-cl analysis (.venv)"
```

## What a batch is

A **batch** is one immutable results drop: a raw `runs/<dir>/` tree (one
subfolder per `{dataset}__{method}__{scenario}__seed{N}` run) plus its
`clover report`-aggregated `results_<name>/` CSVs, described by a
`batch.yaml`:

```yaml
runs_dir: runs/<dir>          # relative to the repo root
results_dir: results_<name>   # relative to the repo root
dataset: <dataset name>
real_backbone: true|false     # true = scientifically meaningful; false = framework validation only
git_commit: <hash the data landed in>
date: "YYYY-MM-DD"
notes: >
  free text
```

Batches never get overwritten in place -- a new results drop is a new
`analysis/batches/bNN_<name>/` folder, discovered automatically by
`core.loader.discover_batches`. Nothing under `core/` may hardcode a batch,
method, or scenario name; that's what keeps this structure durable as new
batches arrive.

## Layout

- `core/loader.py` -- batch discovery (`discover_batches`) and tidy
  DataFrame loading (`load_per_task`, `load_r_matrices`). Only `state ==
  "done"` runs are included.
- `core/metrics_ext.py` -- derived metrics not shipped by `clover report`,
  e.g. `category_level_retention` (the aliasing correction below).
- `core/viz.py` -- plotting helpers (palette, style, bar/trajectory/R-matrix
  figures), following the project's `dataviz` skill: fixed-order
  colorblind-safe categorical palette, one-hue sequential for magnitude,
  legend always present.
- `batches/bNN_.../batch.yaml` + `_build_notebooks.py` -- one batch's
  provenance and its notebook generator (prose + cell order; scenario/method
  names are fine to hardcode here, this is narrative for one specific drop).
- `batches/bNN_.../0*.ipynb` + `figures/` -- committed **executed** (outputs
  visible), 300dpi PNG+PDF.
- `cross/` -- cross-batch and v1-vs-v2 comparisons (empty until >=2
  scientifically-meaningful batches exist to compare).

## Rebuilding everything

```
.venv\Scripts\python.exe analysis/build_all.py
```

Regenerates every batch's notebooks from its `_build_notebooks.py` and
executes them in place (`jupyter nbconvert --execute --inplace`).

## Adding a new batch (e.g. b03, ImageNet-R)

1. Drop the raw runs at `runs/<new_dir>/` and aggregated CSVs at
   `results_<new_name>/` (from `clover report`), and commit them.
2. `mkdir analysis/batches/b03_<name>` and write its `batch.yaml` (dataset,
   `real_backbone: true`, `runs_dir`, `results_dir`, `git_commit`, `date`).
3. Copy an existing batch's `_build_notebooks.py` as a starting point, adjust
   the batch name, `SCENARIO_ORDER`/`METHOD_ORDER`, and prose.
4. `.venv\Scripts\python.exe analysis/build_all.py` -- it picks up the new
   folder automatically; nothing in `core/` needs to change.
5. Commit the batch folder (`batch.yaml`, executed notebooks, `figures/`).

## Metric caveats (surface these wherever the metric appears)

- `FWT` is PILOT's zero-baseline approximation, not true forward transfer.
- `RAG_mean` only scores each class's *first* revisit.
- `Long_Range_Retention` **conflates label aliasing with forgetting** in
  echo scenarios (`exact_replay`, `long_range_revisit`, `mid_range_revisit`,
  `partial_overlap`): a returning category gets a fresh label id, so the
  source block's accuracy collapsing at the echo task looks like forgetting
  but is largely the model splitting prediction mass across two ids for the
  same category. `core.metrics_ext.category_level_retention(R, source_row,
  echo_row)` sums the source and echo block's final-checkpoint accuracy to
  recover the (approximate) pre-echo accuracy -- see
  `batches/b01_cifar224_simplecil_vitb16/02_retention_aliasing.ipynb`.

## Sanity anchors

Mean over 3 seeds, cifar224 SimpleCIL (`b01`): AIA disjoint 0.822,
`exact_replay` source-final 0.306, echo-final 0.464, sum 0.770;
`cumulative_drift` Anchor_Retention 0.819. Both `b01` notebooks assert
against these -- if your numbers differ, the code is wrong, not the data.
