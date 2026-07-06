# Extending CLOVER: methods, datasets, scenarios, backbones

CLOVER's core (`clover/core`) never imports a plugin by name (SPEC R3):
every method, dataset, scenario, and backbone is added by implementing a
small interface and registering it with a decorator. This doc walks
through one worked, runnable example of each — none of them touch
`clover` source outside the new file plus one import-for-registration line.

Every registry (`clover/{methods,datasets,scenarios,backbones}/__init__.py`)
follows the same shape: `register_x`/`get_x`/`list_x` come from one shared
`clover/utils/registry.py:Registry`, and new modules are imported at the
*bottom* of the package's `__init__.py` purely for their
`@register_x(...)` side effect.

---

## Dataset: `image_folder` (config-only, zero new code)

The simplest possible custom dataset needs *no new Python at all* — CLOVER
already ships a generic `ImageFolder`-backed dataset that reads its root
and class count straight from config:

```yaml
stream:
  dataset: {type: image_folder, root: /path/to/my_dataset, num_classes: 12}
  init_cls: 4
  increment: 4
method: {name: simplecil}
```

`/path/to/my_dataset` must contain `train/`/`test/` subfolders, each with
one subdirectory per class:

```
my_dataset/
  train/
    class_a/  img001.jpg  img002.jpg  ...
    class_b/  ...
  test/
    class_a/  ...
    class_b/  ...
```

Run it: `clover run configs/my_run.yaml`. That's the entire integration —
`clover/config/schema.py:StreamSection.from_dict` detects the inline
`{type, root, num_classes}` mapping and resolves it into a `StreamSpec`
with `dataset="image_folder"`, `data_root=<root>`,
`dataset_num_classes=<num_classes>`; `clover/datasets/image_folder_config
.py` cross-checks the declared count against the actual on-disk
subdirectory count at load time.

### Writing a *named* custom dataset instead

If you want your dataset selectable by a short name (and don't want to
repeat `root`/`num_classes` in every config), register your own
`CLDataset` subclass:

```python
# my_plants/dataset.py
from clover.datasets import register_dataset
from clover.datasets.base import CLDataset

@register_dataset("my_plants")
class MyPlantsDataset(CLDataset):
    input_size = 224

    def __init__(self, root="./data", train=True, transform=None):
        super().__init__(root, train, transform)
        # load samples from root, build self._paths / self._targets, etc.

    @property
    def num_classes(self) -> int:
        return 42

    def get_class_to_indices(self) -> dict[int, list[int]]:
        ...  # {original_class_id: [sample_indices]}

    def __getitem__(self, idx):
        ...  # -> (image, original_class_id)

    def __len__(self):
        ...
```

Then `import my_plants.dataset` once, anywhere that runs before
`get_dataset("my_plants")` is called (e.g. at the top of your run script),
and `dataset: my_plants` works in any config. If your dataset already
matches the `<root>/<subdir>/{train,test}/<class>/...` `ImageFolder` shape,
subclass `clover.datasets.image_folder_base.ImageFolderCLDataset` instead
(see `clover/datasets/cub200.py` for the ~5-line pattern) rather than
implementing `CLDataset` from scratch.

---

## Scenario: `distribution_shift`

A scenario is a `@register_scenario`-decorated factory:
`(dataset_info, init_cls, increment, seed, **params) -> StreamSpec`.
`clover/scenarios/distribution_shift.py` (P8) is a complete worked example:

```python
from clover.core.spec import DatasetInfo, RevisitSpec, StreamSpec
from clover.scenarios import register_scenario

@register_scenario("distribution_shift")
def distribution_shift(
    dataset_info: DatasetInfo,
    init_cls: int,
    increment: int,
    seed: int = 42,
    n_shifted: int = 3,
    split_ratio: float = 1.0,
    **params: object,
) -> StreamSpec:
    shifted = list(range(n_shifted))
    images = "new" if split_ratio == 1.0 else f"partial:{split_ratio}"
    return StreamSpec(
        dataset=dataset_info.name,
        init_cls=init_cls,
        increment=increment,
        stream_seed=seed,
        revisits=[
            RevisitSpec(classes=shifted, placement="end_of_stream", label="same", images=images)
        ],
    )
```

`n_shifted` of task 0's classes reappear once, at the end of the stream,
under their *original* label id (`label="same"`) — the label space never
grows for them, only the image distribution behind that label shifts.
Use it from a config exactly like a built-in scenario:

```yaml
stream:
  dataset: cifar100
  init_cls: 10
  increment: 10
  scenario: distribution_shift
  scenario_params: {n_shifted: 3, split_ratio: 0.5}
```

`clover/core/spec.py:RevisitSpec`'s axes (`classes`, `placement`, `label`,
`images`, `min_gap`, `times`) are documented in `docs/CONCEPTS.md` §3 — a
scenario is just a named, parameterized way of constructing one or more
`RevisitSpec`s. Not every revisit pattern fits: `RevisitSpec`'s class-to-
task assignment is sequential/budget-driven only, with no hook for
external taxonomy metadata or backward-in-time class injection — see
`docs/CONCEPTS.md` §8 for the three v1 scenario shapes (`hierarchical`,
`near_miss`, `symmetric_pair`) that don't fit this model and why.

---

## Backbone: a config-selectable base model

Backbones are either complete, ready-to-use `nn.Module`s (like
`clover/backbones/tiny_mlp.py:TinyMLP`) or mechanism wrappers over a
resolved *base model* (like `clover/backbones/prompt_pool.py`, which wraps
whatever `clover/backbones/loader.py:resolve_base_model` returns). To
register a complete backbone:

```python
# my_backbone.py
import torch.nn as nn
from clover.backbones import register_backbone

@register_backbone("my_cnn")
class MyCNN(nn.Module):
    def __init__(self, input_size: int = 224) -> None:
        super().__init__()
        self.feature_dim = 128
        self.net = ...  # -> [batch, feature_dim]

    def forward(self, x):
        return self.net(x)
```

Select it from a method config: `method: {name: simplecil, backbone: my_cnn}`.
`resolve_base_model(name, source="auto"|"timm"|"class", **kwargs)` is the
one place base-model resolution happens for wrapper mechanisms: `"auto"`
checks the backbone registry first, then falls back to
`timm.create_model(name, pretrained=False, num_classes=0)`; `source="class"`
resolves `"module.path:ClassName"` for a fully custom base model. Never
hard-code a checkpoint name inside a method (SPEC's coding standard) —
always go through config + `resolve_base_model`.

---

## Method: a minimal `CLMethod`

`clover/methods/base.py:CLMethod` is the full lifecycle contract; the
framework (`clover.training.Trainer`) owns experience iteration, dataloader
construction, checkpointing, and evaluation — a method's inner epoch loop
lives entirely in `train_experience`. `clover/methods/simple_cil.py` is
the smallest complete example (frozen backbone, no gradient step,
closed-form cosine-prototype head); its shape:

```python
from clover.methods import register_method
from clover.methods.base import CLMethod, StreamInfo, TrainContext

@register_method("my_method")
class MyMethod(CLMethod):
    name = "my_method"
    default_backbone = "tiny_vit"
    cacheable_features = False  # True only if the backbone never changes

    def build(self, stream_info: StreamInfo, cfg: dict) -> None:
        ...  # construct backbone/head from stream_info.input_size, cfg

    def train_experience(self, exp, loader, ctx: TrainContext) -> None:
        for images, targets in loader:
            ...  # exp.label_space gives revisit-safe masks; never
                 # targets >= known_classes range arithmetic

    def classifier(self):
        ...  # images -> logits module for the shared evaluator

    def state_dict(self) -> dict:
        ...

    def load_state_dict(self, state: dict) -> None:
        ...
```

Then `method: {name: my_method}` in any config. See `docs/methods.md` for
how the 9 built-in methods use `before_experience`/`after_experience` for
head growth, prompt/adapter allocation, and prototype/merge steps.
