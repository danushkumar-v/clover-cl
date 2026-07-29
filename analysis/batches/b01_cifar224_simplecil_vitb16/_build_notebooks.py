"""Generate this batch's notebooks from a compact spec.

Run (from the batch folder): ../../../.venv/Scripts/python.exe _build_notebooks.py
Then execute via analysis/build_all.py (or directly:
  jupyter nbconvert --to notebook --execute --inplace 0*.ipynb
  run from this folder so `sys.path.insert(0, '../..')` resolves to analysis/).

The plotting/derived-metric logic lives in ../../core/; this file holds the
batch's own prose + cell order + scenario ordering (scenario names are
batch-specific narrative, not something core/ is allowed to hardcode).
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
KERNEL = {
    "kernelspec": {"display_name": "clover-cl analysis (.venv)", "language": "python", "name": "clover-cl-analysis"},
    "language_info": {"name": "python"},
}

_HEADER = [
    ("code",
     "import sys; sys.path.insert(0, '../..')\n"
     "from core import loader, viz\n"
     "viz.set_style()\n"
     "BATCH = next(b for b in loader.discover_batches('../..') if b.name == 'b01_cifar224_simplecil_vitb16')\n"
     "df = loader.load_per_task(BATCH)\n"
     "print(BATCH.name, '|', len(df), 'per-task rows |',\n"
     "      df[['method','scenario','seed']].drop_duplicates().shape[0], 'runs')"),
    ("code",
     "SCENARIO_ORDER = [\n"
     "    'disjoint_baseline', 'distribution_shift', 'cumulative_drift',\n"
     "    'exact_replay', 'partial_overlap', 'mid_range_revisit', 'long_range_revisit',\n"
     "]\n"
     "SCENARIO_LABELS = {s: s.replace('_', ' ') for s in SCENARIO_ORDER}"),
]


def build(path, title, cells):
    nb = nbf.v4.new_notebook()
    out = [nbf.v4.new_markdown_cell(title)]
    for kind, src in _HEADER + cells:
        out.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    nb.cells = out
    nb.metadata = KERNEL
    nbf.write(nb, str(HERE / path))
    print("wrote", path)


# ── 01 — overview ──────────────────────────────────────────────────────────
build("01_overview.ipynb",
"""# 01 · Overview — CIFAR224, SimpleCIL + ViT-B/16 (pretrained)

Batch `b01_cifar224_simplecil_vitb16`: 7 scenarios x 3 seeds (21 runs), the
**real** pretrained ViT-B/16 backbone -- these are the scientifically
meaningful numbers in this repo (contrast with `b02_cifar100_matrix_full`,
which is framework validation on untrained stand-in backbones only).

Caveats that apply to every metric below (see `analysis/README.md`):
`FWT` is PILOT's zero-baseline approximation, not true FWT; `RAG_mean` only
scores each class's first revisit; `Long_Range_Retention` conflates label
aliasing with forgetting in the echo scenarios -- notebook 02 is entirely
about that last one.
""",
[
("md", "### Final AIA and A_t per scenario (mean +/- SEM over 3 seeds)"),
("code",
 "aia = viz.final_task_per_seed(df, 'AIA', ['scenario'])\n"
 "at = viz.final_task_per_seed(df, 'A_t', ['scenario'])\n"
 "display(aia.set_index('scenario').reindex(SCENARIO_ORDER))"),
("code",
 "fig = viz.fig_bar_by_group(aia, 'scenario', SCENARIO_ORDER,\n"
 "    'AIA (mean +/- SEM)', 'Average incremental accuracy by scenario', SCENARIO_LABELS)\n"
 "viz.save_fig(fig, BATCH.path / 'figures', 'aia_by_scenario')"),
("code",
 "fig = viz.fig_bar_by_group(at, 'scenario', SCENARIO_ORDER,\n"
 "    'final A_t (mean +/- SEM)', 'Final-task accuracy by scenario', SCENARIO_LABELS)\n"
 "viz.save_fig(fig, BATCH.path / 'figures', 'final_at_by_scenario')"),
("md", "### Accuracy trajectory across the stream\n"
       "`A_t` per task, mean +/- SEM band over the 3 seeds. Scenarios with an "
       "echo task (exact_replay, partial_overlap, mid_range/long_range_revisit) "
       "show the drop at their final task; see notebook 02 for why it isn't "
       "pure forgetting."),
("code",
 "traj = viz.seed_mean_sem(df, 'A_t', ['scenario', 'task_idx'])\n"
 "fig = viz.fig_trajectory_by_group(traj, 'scenario', SCENARIO_ORDER,\n"
 "    'A_t', 'Overall accuracy across the stream', SCENARIO_LABELS)\n"
 "viz.save_fig(fig, BATCH.path / 'figures', 'at_trajectory')"),
("md", "### Sanity check against the documented anchors\n"
       "(`analysis/README.md` / the analysis task brief) -- if these don't "
       "match, the loader or metric code has a bug, not the data."),
("code",
 "disjoint_aia = aia.set_index('scenario').loc['disjoint_baseline', 'mean']\n"
 "print(f'disjoint_baseline AIA = {disjoint_aia:.3f} (anchor: 0.822)')\n"
 "assert abs(disjoint_aia - 0.822) < 0.01, 'AIA anchor mismatch -- debug the loader/aggregation'\n"
 "print('PASS')"),
])


# ── 02 — retention & aliasing (THE key-finding notebook) ──────────────────
build("02_retention_aliasing.ipynb",
"""# 02 · Retention and label aliasing (the key finding)

In the echo scenarios (`exact_replay`, `long_range_revisit`,
`mid_range_revisit`, `partial_overlap`) a returning category re-appears
under a **fresh label id** (cifar224 echo ids are >= 100) rather than its
original one. `Long_Range_Retention` -- the source block's accuracy at the
final checkpoint -- reads this as catastrophic forgetting: it collapses at
the echo task. But the model hasn't forgotten the category. It has split
its prediction mass between the original id and the new one: **source +
echo accuracy sums back to roughly the pre-echo accuracy.** That's label
aliasing, not forgetting.

Same-id revisits (`cumulative_drift`'s anchors) don't have this failure
mode -- there's no second id to split mass into, and retention there is
*better* than the disjoint baseline, not worse.
""",
[
("code",
 "import numpy as np, pandas as pd\n"
 "from core.metrics_ext import category_level_retention\n"
 "r_matrices = loader.load_r_matrices(BATCH)"),
("code",
 "# source_row: which R-matrix row holds the class block whose identity gets\n"
 "# echoed later. echo_row: which row holds the echo's fresh label id (the\n"
 "# last row, for every scenario that has one). Verified against\n"
 "# clover/training/trainer.py's r_matrix.update(test_exp.task_label, ...).\n"
 "ECHO_SOURCE_ROW = {\n"
 "    'exact_replay': 0, 'long_range_revisit': 0,\n"
 "    'mid_range_revisit': 4, 'partial_overlap': 0,\n"
 "}\n"
 "REFERENCE_SCENARIOS = ['disjoint_baseline', 'cumulative_drift']  # no echo id: same-id revisit or no revisit\n"
 "SEEDS = (1, 2, 3)\n"
 "\n"
 "def run_ids(scenario):\n"
 "    return [f'{BATCH.dataset}__simplecil__{scenario}__seed{s}' for s in SEEDS]\n"
 "\n"
 "mean_R = {sc: viz.mean_r_matrix(r_matrices, run_ids(sc))\n"
 "          for sc in list(ECHO_SOURCE_ROW) + REFERENCE_SCENARIOS}"),
("md", "### The aliasing table\n"
       "`before` = source-block accuracy one checkpoint before the echo; "
       "`after` = source-block accuracy at the final checkpoint (this is "
       "`Long_Range_Retention`); `echo` = the echo block's own accuracy; "
       "`sum` = after + echo, which recovers roughly `before`."),
("code",
 "rows = []\n"
 "for sc, src_row in ECHO_SOURCE_ROW.items():\n"
 "    R = mean_R[sc]\n"
 "    r = category_level_retention(R, src_row, R.shape[0] - 1)\n"
 "    rows.append({'scenario': sc, **r})\n"
 "for sc in REFERENCE_SCENARIOS:\n"
 "    R = mean_R[sc]\n"
 "    rows.append({\n"
 "        'scenario': sc,\n"
 "        'source_pre': R[0, R.shape[1] - 2],\n"
 "        'source_post': R[0, R.shape[1] - 1],\n"
 "        'echo': float('nan'), 'sum': float('nan'),\n"
 "    })\n"
 "table = pd.DataFrame(rows).set_index('scenario').round(3)\n"
 "display(table)"),
("code",
 "# Sanity anchors (analysis task brief, mean of 3 seeds, cifar224 SimpleCIL):\n"
 "# exact_replay source-final 0.306, echo-final 0.464, sum 0.770.\n"
 "er = table.loc['exact_replay']\n"
 "assert abs(er['source_post'] - 0.306) < 0.01, 'source_post anchor mismatch'\n"
 "assert abs(er['echo'] - 0.464) < 0.01, 'echo anchor mismatch'\n"
 "assert abs(er['sum'] - 0.770) < 0.01, 'sum anchor mismatch'\n"
 "print('PASS -- exact_replay matches the documented anchors')"),
("code",
 "anchor_ret = viz.final_task_per_seed(df, 'Anchor_Retention', ['scenario'])\\\n"
 "    .set_index('scenario').loc['cumulative_drift', 'mean']\n"
 "print(f'cumulative_drift Anchor_Retention = {anchor_ret:.3f} (anchor: 0.819)')\n"
 "assert abs(anchor_ret - 0.819) < 0.01\n"
 "print(f\"cumulative_drift row-0 final accuracy = {table.loc['cumulative_drift', 'source_post']:.3f} \"\n"
 "      f\"vs. disjoint_baseline {table.loc['disjoint_baseline', 'source_post']:.3f} \"\n"
 "      \"-- same-id revisit IMPROVES retention, no aliasing to correct for.\")"),
("md", "### The figure: source-block accuracy trajectory, echo drop annotated\n"
       "Every echo scenario's source block tracks the same gentle decline "
       "until its echo task, where it falls off a cliff. `AIA` stays flat "
       "(0.815-0.841) across every scenario in this batch -- the model is "
       "not doing worse overall, it is just splitting one category across "
       "two ids."),
("code",
 "fig = viz.fig_source_block_trajectory(\n"
 "    {sc: mean_R[sc] for sc in ECHO_SOURCE_ROW}, ECHO_SOURCE_ROW, list(ECHO_SOURCE_ROW),\n"
 "    labels=SCENARIO_LABELS)\n"
 "viz.save_fig(fig, BATCH.path / 'figures', 'source_block_trajectory_echo_drop')"),
("md", "### R-matrices: collapse-and-split vs. genuine same-id retention\n"
       "`exact_replay` (left) shows the source row (row 0) fading to near-zero "
       "in its final column while a bright new row appears at the bottom -- "
       "the split. `cumulative_drift` (right) has no second row to split "
       "into: the anchor stays inside row 0 throughout and never collapses."),
("code",
 "import matplotlib.pyplot as plt\n"
 "fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))\n"
 "viz.fig_r_matrix(mean_R['exact_replay'], 'exact_replay (source row 0 + echo row 9)', ax=axes[0])\n"
 "viz.fig_r_matrix(mean_R['cumulative_drift'], 'cumulative_drift (anchor stays in row 0)', ax=axes[1])\n"
 "fig.tight_layout()\n"
 "viz.save_fig(fig, BATCH.path / 'figures', 'r_matrix_aliasing_vs_control')"),
("md", "### Takeaway\n"
       "`Long_Range_Retention` alone is a misleading number for the echo "
       "scenarios: read in isolation it says the model lost 40-60 points of "
       "accuracy on the original category. Read together with the echo "
       "block's own accuracy, the loss is mostly a labeling split, not a "
       "memory failure -- `category_level_retention` in "
       "`analysis/core/metrics_ext.py` is the corrective metric."),
])

print("done. Execute with: ../../../.venv/Scripts/python.exe -m jupyter nbconvert "
      "--to notebook --execute --inplace 0*.ipynb  (run from this folder)")
