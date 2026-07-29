# cross/

Cross-batch and v1-vs-v2 comparisons land here once there are >=2
scientifically-meaningful batches (`real_backbone: true`) to compare --
currently just `b01_cifar224_simplecil_vitb16`, so this folder is empty.

v1 (pilot-bench) comparison figures already exist at
`../../clover-pilot-bench/analysis/v2_fixed_size/figures` (read-only
reference, not reproduced here).

When batch b03 (ImageNet-R) lands, a `01_v2_cross_batch.ipynb` here should
use `core.loader.discover_batches` to pull every `real_backbone: true` batch
and compare them side by side -- no batch name hardcoded in `core/`, only in
this notebook's own cell code.
