"""Generate this batch's notebook from a compact spec.

Run (from the batch folder): ../../../.venv/Scripts/python.exe _build_notebooks.py
Then execute via analysis/build_all.py.

The plotting/loading logic lives in ../../core/; this file holds the
batch's own prose + cell order (method/scenario names are batch-specific
narrative, not something core/ is allowed to hardcode).
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
     "BATCH = next(b for b in loader.discover_batches('../..') if b.name == 'b02_cifar100_matrix_full')\n"
     "df = loader.load_per_task(BATCH)\n"
     "print(BATCH.name, '|', len(df), 'per-task rows |',\n"
     "      df[['method','scenario','seed']].drop_duplicates().shape[0], 'runs done')"),
    ("code",
     "METHOD_ORDER = ['simplecil', 'ranpac', 'ease', 'aper_adapter', 'tuna',\n"
     "                'mos', 'coda_prompt', 'dualprompt', 'l2p']\n"
     "SCENARIO_ORDER = ['disjoint_baseline', 'distribution_shift', 'cumulative_drift',\n"
     "                  'exact_replay', 'partial_overlap', 'mid_range_revisit', 'long_range_revisit']\n"
     "N_SEEDS = 3"),
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


build("01_framework_validation.ipynb",
"""# 01 · Framework validation -- CIFAR-100, 9 methods, untrained stand-in backbones

**This batch is framework validation only, never a method comparison.**
Every run uses a tiny randomly-initialized backbone (no pretraining, no
real feature learning), so accuracies sit at or near chance
(1/100 = 0.01 for a 100-class stream). The only question this batch
answers is: *does the harness run every (method x scenario x seed)
combination to completion and produce finite, well-formed metrics?* Any
figure from this batch must carry the "framework validation (untrained
stand-in backbones)" label -- see `b01_cifar224_simplecil_vitb16` for the
scientifically meaningful numbers (real pretrained ViT-B/16).
""",
[
("md", "### Completeness: 189/189 runs done\n"
       "9 methods x 7 scenarios x 3 seeds = 189 expected runs."),
("code",
 "n_runs = df[['method','scenario','seed']].drop_duplicates().shape[0]\n"
 "expected = len(METHOD_ORDER) * len(SCENARIO_ORDER) * N_SEEDS\n"
 "print(f'{n_runs}/{expected} runs done')\n"
 "assert n_runs == expected == 189, 'incomplete matrix -- debug before treating this batch as validated'\n"
 "print('PASS')"),
("code",
 "counts = (df[['method','scenario','seed']].drop_duplicates()\n"
 "          .groupby(['method','scenario'], observed=True).size().unstack('scenario'))\n"
 "fig = viz.fig_completeness_heatmap(counts, METHOD_ORDER, SCENARIO_ORDER)\n"
 "viz.save_fig(fig, BATCH.path / 'figures', 'completeness_heatmap')"),
("md", "### Finite metrics per method\n"
       "`A_t` and `AIA` must be finite and non-null at every run's final "
       "task -- a silent NaN here would mean a method crashed or produced "
       "degenerate output without tripping `state=='done'`."),
("code",
 "final_at = viz.final_task_per_seed(df, 'A_t', ['method'])\n"
 "finite_counts = (df[df.metric=='A_t']\n"
 "    .loc[df[df.metric=='A_t'].groupby(['method','scenario','seed'])['task_idx'].idxmax()]\n"
 "    .groupby('method', observed=True)['value'].apply(lambda s: s.notna().sum()))\n"
 "check = final_at.set_index('method').reindex(METHOD_ORDER).assign(\n"
 "    finite_runs=finite_counts.reindex(METHOD_ORDER).to_numpy(),\n"
 "    expected_runs=len(SCENARIO_ORDER) * N_SEEDS)\n"
 "display(check[['finite_runs', 'expected_runs', 'mean', 'sem']].round(4))\n"
 "assert (check['finite_runs'] == check['expected_runs']).all(), 'a method has non-finite A_t somewhere'\n"
 "print('PASS -- every method has finite A_t at every run\\'s final task')"),
("md", "### Chance-level framing\n"
       "Final `A_t` per method, with the 1/100 chance line. These bars must "
       "**never** be read as \"which method is better\" -- with an untrained "
       "backbone there is no learned representation for any method to "
       "exploit, so differences here reflect only tiny sampling/initialization "
       "noise around chance, not a real performance ranking."),
("code",
 "import matplotlib.pyplot as plt\n"
 "fig = viz.fig_bar_by_group(final_at, 'method', METHOD_ORDER,\n"
 "    'final A_t (mean +/- SEM)',\n"
 "    'Final accuracy per method -- framework validation (untrained stand-in backbones)')\n"
 "fig.axes[0].axhline(0.01, ls='--', lw=1.2, color=viz.STATUS['warning'])\n"
 "fig.axes[0].text(0.02, 0.012, 'chance = 1/100', fontsize=8.5, color='#7a5a00')\n"
 "viz.save_fig(fig, BATCH.path / 'figures', 'final_at_by_method_chance_framing')"),
("md", "### Takeaway\n"
       "189/189 runs done, every method's final-task metrics are finite, "
       "and every method sits within a few hundredths of chance -- exactly "
       "what a correctly-wired harness with an untrained backbone should "
       "produce. This batch validates the framework; `b01` carries the "
       "science."),
])

print("done. Execute with: ../../../.venv/Scripts/python.exe -m jupyter nbconvert "
      "--to notebook --execute --inplace 0*.ipynb  (run from this folder)")
