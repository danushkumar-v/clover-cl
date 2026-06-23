"""OverlapDataManager — CLOVER's main entry point.

Drop-in replacement for PILOT's ``DataManager``.  When ``overlap_spec=None``
the class produces identical class orderings, task slices, and dataset objects
to PILOT (seed-1993 verified).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from clover.core.image_assigner import assign_images
from clover.core.overlap_spec import ImageSplit, OverlapSpec
from clover.core.task_builder import build_tasks
from clover.utils.seeding import get_rng, pilot_class_order

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, Any] = {}


def _register(name: str, cls: Any) -> None:
    _REGISTRY[name.lower()] = cls


def _get_dataset_cls(name: str) -> Any:
    key = name.lower()
    if key not in _REGISTRY:
        raise NotImplementedError(
            f"Unknown dataset {name!r}. Registered names: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[key]


def _populate_registry() -> None:
    from clover.datasets.cifar100 import CIFAR100Dataset
    from clover.datasets.cub200 import CUB200Dataset
    from clover.datasets.imagenet_a import ImageNetADataset
    from clover.datasets.imagenet_r import ImageNetRDataset
    from clover.datasets.omnibench import OmniBenchDataset
    from clover.datasets.vtab import VTABDataset

    _register("cifar100", CIFAR100Dataset)
    _register("cub200", CUB200Dataset)
    _register("cub", CUB200Dataset)
    _register("imagenetr", ImageNetRDataset)
    _register("imagenet_r", ImageNetRDataset)
    _register("imageneta", ImageNetADataset)
    _register("imagenet_a", ImageNetADataset)
    _register("omnibenchmark", OmniBenchDataset)
    _register("omnibench", OmniBenchDataset)
    _register("vtab", VTABDataset)


_populate_registry()

# ---------------------------------------------------------------------------
# CloverDataset — returned by get_dataset()
# ---------------------------------------------------------------------------


class CloverDataset(Dataset):
    """Thin dataset wrapper returned by :meth:`OverlapDataManager.get_dataset`.

    Yields ``(sample_idx, image, remapped_label)`` tuples — identical format to
    PILOT's ``DummyDataset`` so that downstream training loops need no changes.

    Args:
        images: Flat numpy array of images (N, H, W, C) or array of path strings.
        labels: Flat numpy array of *remapped* class IDs.
        trsf: Composed torchvision transform.
        use_path: If ``True``, *images* contains file paths; otherwise raw arrays.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        trsf: Any,
        use_path: bool = False,
    ) -> None:
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[int, Any, int]:
        if self.use_path:
            with open(self.images[idx], "rb") as fh:
                image = Image.open(fh).convert("RGB")
            image = self.trsf(image)
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]
        return idx, image, label


# ---------------------------------------------------------------------------
# OverlapDataManager
# ---------------------------------------------------------------------------


class OverlapDataManager:
    """Continual-learning data manager with configurable class overlap.

    This is a drop-in replacement for PILOT's ``DataManager``.  When
    ``overlap_spec=None`` it behaves identically to PILOT for the same
    ``dataset_name``, ``init_cls``, ``increment``, and ``shuffle_seed``.

    Args:
        dataset_name: One of ``"cifar100"``, ``"cub200"``, ``"imagenetr"``,
            ``"imageneta"``, ``"omnibenchmark"``, ``"vtab"``.
        init_cls: Number of classes in the first task.
        increment: Number of new classes added in each subsequent task.
        overlap_spec: Overlap configuration.  Pass ``None`` for strict PILOT-
            equivalent disjoint partitioning.
        shuffle_seed: Seed used by PILOT's class-order shuffle (default 1993).
        data_root: Root directory passed to the underlying dataset wrapper.
        test_overlap_strategy: How test images are distributed for shared
            classes.  Defaults to ``"duplicate"`` (all test images available to
            every task that contains the class).
    """

    def __init__(
        self,
        dataset_name: str,
        init_cls: int,
        increment: int,
        overlap_spec: Optional[OverlapSpec] = None,
        shuffle_seed: int = 1993,
        data_root: str = "./data",
        test_overlap_strategy: str = "duplicate",
        preserve_task_size: bool = True,
    ) -> None:
        self.dataset_name = dataset_name
        self.init_cls = init_cls
        self.increment = increment
        self.overlap_spec = overlap_spec
        self.shuffle_seed = shuffle_seed
        self.data_root = data_root
        self.test_overlap_strategy = test_overlap_strategy
        self.preserve_task_size = preserve_task_size

        # Validate spec if given
        if overlap_spec is not None:
            overlap_spec.validate()

        effective_spec = overlap_spec or OverlapSpec(mode="none")

        # --- Load underlying datasets ---
        ds_cls = _get_dataset_cls(dataset_name)
        train_ds = ds_cls(root=data_root, train=True)
        test_ds = ds_cls(root=data_root, train=False)

        self._use_path: bool = train_ds.use_path
        # Number of *real* categories in the underlying dataset.  This drives
        # the class-order permutation and the remap; it is distinct from
        # ``_total_classes`` (the classifier-head size), which may be larger
        # when echo/clone classes extend the label space.
        self._real_classes: int = train_ds.num_classes

        # Store raw data references (PILOT-compatible internals)
        if self._use_path:
            self._train_data: np.ndarray = np.array(
                [train_ds._paths[i] for i in range(len(train_ds))]
            )
            self._test_data: np.ndarray = np.array(
                [test_ds._paths[i] for i in range(len(test_ds))]
            )
        else:
            self._train_data = train_ds._data
            self._test_data = test_ds._data

        self._train_targets_orig: np.ndarray = train_ds._targets.copy()
        self._test_targets_orig: np.ndarray = test_ds._targets.copy()

        # Transforms
        self._train_trsf = train_ds.train_trsf
        self._test_trsf = train_ds.test_trsf
        self._common_trsf = train_ds.common_trsf

        # --- Class order (PILOT-compatible legacy RNG) ---
        self._class_order: List[int] = pilot_class_order(self._real_classes, shuffle_seed)
        logger.info("Class order (first 20): %s", self._class_order[:20])

        # Remap targets: target value = position of original class in _class_order
        self._train_targets: np.ndarray = _map_new_class_index(
            self._train_targets_orig, self._class_order
        )
        self._test_targets: np.ndarray = _map_new_class_index(
            self._test_targets_orig, self._class_order
        )

        # --- Echo (clone) classes ---
        # new_id -> (source_id, image_relation).  Echo ids are fresh label
        # slots whose images come from an earlier source category.
        self._echo_sources: Dict[int, tuple] = {
            e.new_id: (e.source_id, e.image_relation) for e in effective_spec.echoes
        }

        # --- Build task class lists ---
        # Two layout sources: an explicit per-task list supplied by the spec
        # (fixed-size scenarios) or the legacy backbone-slicing path.
        if effective_spec.task_class_lists is not None:
            self._task_class_lists: List[List[int]] = [
                list(t) for t in effective_spec.task_class_lists
            ]
        else:
            self._task_class_lists = build_tasks(
                total_classes=self._real_classes,
                init_cls=init_cls,
                increment=increment,
                class_order=self._class_order,
                overlap_spec=effective_spec,
                preserve_size=preserve_task_size,
            )
        self._increments: List[int] = [len(t) for t in self._task_class_lists]

        # Head size: explicit override, else one past the largest id in use.
        # The layout is validated contiguous (0..N-1), so max id + 1 == the
        # number of label slots PILOT must allocate.
        if effective_spec.total_classes_override is not None:
            self._total_classes: int = int(effective_spec.total_classes_override)
        else:
            max_id = max((max(t) for t in self._task_class_lists if t), default=-1)
            self._total_classes = max_id + 1

        # --- Build class → image-index maps (using REMAPPED class IDs) ---
        train_c2i = _build_remapped_class_to_indices(
            self._train_targets, self._real_classes
        )
        test_c2i = _build_remapped_class_to_indices(
            self._test_targets, self._real_classes
        )

        # --- Assign images via image_assigner ---
        rng = get_rng(effective_spec.seed)
        self._train_assignment: List[Dict[int, List[int]]] = assign_images(
            class_to_image_indices=train_c2i,
            task_class_lists=self._task_class_lists,
            overlap_pairs=effective_spec.pairs,
            image_split=effective_spec.image_split,
            rng=rng,
        )

        # Test: always duplicate by default so every task can evaluate all its classes
        test_split = ImageSplit(strategy=self.test_overlap_strategy, ratio=0.5)
        rng_test = get_rng(effective_spec.seed + 1)
        self._test_assignment: List[Dict[int, List[int]]] = assign_images(
            class_to_image_indices=test_c2i,
            task_class_lists=self._task_class_lists,
            overlap_pairs=effective_spec.pairs,
            image_split=test_split,
            rng=rng_test,
        )

        # --- Echo image assignment (post-step) ---
        # Echo ids own no real images, so assign_images left them empty; fill
        # them from their source category here, splitting the pool for "new".
        if self._echo_sources:
            self._assign_echo_images(train_c2i, test_c2i, base_seed=effective_spec.seed)

    # ------------------------------------------------------------------
    # PILOT-compatible properties
    # ------------------------------------------------------------------

    @property
    def nb_tasks(self) -> int:
        """Total number of tasks."""
        return len(self._task_class_lists)

    @property
    def nb_classes(self) -> int:
        """Total number of unique classes across all tasks."""
        return self._total_classes

    @property
    def total_classes(self) -> int:
        """Total number of unique classes (alias for :attr:`nb_classes`)."""
        return self._total_classes

    def get_task_size(self, task: int) -> int:
        """Number of classes in task *task* (PILOT-compatible name)."""
        return self._increments[task]

    # ------------------------------------------------------------------
    # New CLOVER API
    # ------------------------------------------------------------------

    def get_task_classes(self, task_id: int) -> List[int]:
        """Return the list of remapped class IDs in *task_id*.

        Args:
            task_id: Zero-based task index.

        Returns:
            List of remapped class IDs assigned to this task.
        """
        return list(self._task_class_lists[task_id])

    def get_dataset(
        self,
        indices_or_task_id: Union[int, List[int]],
        source: str = "train",
        mode: str = "train",
        appendent: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        ret_data: bool = False,
        m_rate: Optional[float] = None,
    ) -> Union[Dataset, Tuple[np.ndarray, np.ndarray, Dataset]]:
        """Return a :class:`CloverDataset` for a task or class list.

        Supports **two calling conventions** for PILOT drop-in compatibility:

        *New CLOVER API* — pass an integer task ID::

            dataset = dm.get_dataset(task_id=0, source="train", mode="train")

        *PILOT-compatible API* — pass a list of remapped class indices::

            dataset = dm.get_dataset([0, 1, 2, 3, 4], "train", "train")

        Args:
            indices_or_task_id: Either an integer task ID (CLOVER API) or a list
                of remapped class IDs (PILOT-compatible API).
            source: ``"train"`` or ``"test"``.
            mode: Transform mode — ``"train"``, ``"test"``, or ``"flip"``.
            appendent: Optional ``(data, targets)`` arrays appended to the
                dataset (PILOT-compatible).
            ret_data: If ``True``, return ``(data, targets, dataset)`` instead
                of just the dataset (PILOT-compatible).
            m_rate: Memory rate for ``_select_rmm``-style subsampling
                (PILOT-compatible; ``None`` disables).

        Returns:
            A :class:`CloverDataset`.  If ``ret_data=True``, returns
            ``(data_array, targets_array, CloverDataset)``.
        """
        # Resolve to a class-index list
        if isinstance(indices_or_task_id, int):
            task_id = indices_or_task_id
            indices = self._task_class_lists[task_id]
            use_assignment = True
        else:
            indices = list(indices_or_task_id)
            use_assignment = False
            # Determine task_id for assignment lookup — use first matching task
            task_id = self._find_task_for_indices(indices)

        trsf = self._make_transform(mode)

        if use_assignment and self.overlap_spec is not None:
            # Use pre-computed overlap-aware assignment
            assignment = (
                self._train_assignment[task_id]
                if source == "train"
                else self._test_assignment[task_id]
            )
            data, targets = self._collect_from_assignment(source, assignment, m_rate)
        else:
            # Baseline path: identical to PILOT's _select, plus echo handling.
            # Echo (clone) ids carry no rows in the target array, so a plain
            # _select would yield nothing — this is the path PILOT's full-range
            # test loader uses, so echo classes must be reachable here too.
            x, y = self._get_raw(source)
            data_list, targets_list = [], []
            for idx in indices:
                if idx in self._echo_sources:
                    cd, ct = self._echo_rows(source, idx)
                elif m_rate is None:
                    cd, ct = _select(x, y, idx, idx + 1)
                else:
                    cd, ct = _select_rmm(x, y, idx, idx + 1, m_rate)
                data_list.append(cd)
                targets_list.append(ct)
            if data_list:
                data = np.concatenate(data_list)
                targets = np.concatenate(targets_list)
            else:
                data = np.array([])
                targets = np.array([])

        if appendent is not None and len(appendent) != 0:
            data = np.concatenate([data, appendent[0]])
            targets = np.concatenate([targets, appendent[1]])

        ds = CloverDataset(data, targets, trsf, self._use_path)
        if ret_data:
            return data, targets, ds
        return ds

    def get_dataset_with_split(
        self,
        indices_or_task_id: Union[int, List[int]],
        source: str = "train",
        mode: str = "train",
        appendent: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        val_samples_per_class: int = 0,
    ) -> Tuple[Dataset, Dataset]:
        """Return a train/val split — PILOT-compatible method.

        Args:
            indices_or_task_id: Integer task ID or list of class indices.
            source: ``"train"`` or ``"test"``.
            mode: Transform mode.
            appendent: Optional extra data appended after splitting.
            val_samples_per_class: How many samples per class to put in val.

        Returns:
            ``(train_dataset, val_dataset)`` pair.
        """
        if isinstance(indices_or_task_id, int):
            indices = self._task_class_lists[indices_or_task_id]
        else:
            indices = list(indices_or_task_id)

        x, y = self._get_raw(source)
        trsf_train = self._make_transform(mode)
        trsf_val = self._make_transform("test")

        train_data, train_targets = [], []
        val_data, val_targets = [], []

        for idx in indices:
            cd, ct = _select(x, y, idx, idx + 1)
            val_idx = np.random.choice(len(cd), val_samples_per_class, replace=False)
            train_idx = list(set(np.arange(len(cd))) - set(val_idx))
            val_data.append(cd[val_idx])
            val_targets.append(ct[val_idx])
            train_data.append(cd[train_idx])
            train_targets.append(ct[train_idx])

        if appendent is not None:
            ax, ay = appendent
            for idx in range(0, int(np.max(ay)) + 1):
                cd, ct = _select(ax, ay, idx, idx + 1)
                val_idx = np.random.choice(len(cd), val_samples_per_class, replace=False)
                train_idx = list(set(np.arange(len(cd))) - set(val_idx))
                val_data.append(cd[val_idx])
                val_targets.append(ct[val_idx])
                train_data.append(cd[train_idx])
                train_targets.append(ct[train_idx])

        td = np.concatenate(train_data) if train_data else np.array([])
        tt = np.concatenate(train_targets) if train_targets else np.array([])
        vd = np.concatenate(val_data) if val_data else np.array([])
        vt = np.concatenate(val_targets) if val_targets else np.array([])

        return (
            CloverDataset(td, tt, trsf_train, self._use_path),
            CloverDataset(vd, vt, trsf_val, self._use_path),
        )

    # ------------------------------------------------------------------
    # Overlap diagnostics
    # ------------------------------------------------------------------

    def get_overlap_matrix(self) -> np.ndarray:
        """Return a ``[T x T]`` matrix where ``M[i, j]`` = number of classes
        shared between tasks *i* and *j*.

        The diagonal holds the number of classes in each task.

        Returns:
            Square integer numpy array of shape ``(nb_tasks, nb_tasks)``.
        """
        T = self.nb_tasks
        mat = np.zeros((T, T), dtype=int)
        for i in range(T):
            set_i = set(self._task_class_lists[i])
            for j in range(T):
                mat[i, j] = len(set_i & set(self._task_class_lists[j]))
        return mat

    def get_manifest(self) -> dict:
        """Return a full reproducibility manifest.

        Returns:
            Nested dict ``{task_id: {class_id: [image_indices]}}``.
        """
        return {
            str(t): {str(c): [int(i) for i in idxs] for c, idxs in task_map.items()}
            for t, task_map in enumerate(self._train_assignment)
        }

    def save_manifest(self, path: str) -> None:
        """Save a JSON manifest with a header describing the run configuration.

        Args:
            path: Destination file path.
        """
        import datetime

        from clover import __version__

        header = {
            "clover_version": __version__,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "dataset": self.dataset_name,
            "init_cls": self.init_cls,
            "increment": self.increment,
            "shuffle_seed": self.shuffle_seed,
            "overlap_spec": (
                self.overlap_spec.to_dict() if self.overlap_spec is not None else None
            ),
        }
        manifest = {
            "_header": header,
            "train": self.get_manifest(),
        }
        with open(path, "w") as fh:
            json.dump(manifest, fh, indent=2)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> "OverlapDataManager":
        """Construct an :class:`OverlapDataManager` from a YAML config file.

        The YAML must contain top-level keys matching the constructor arguments.
        An optional ``overlap_spec`` section is parsed by
        :meth:`~clover.core.overlap_spec.OverlapSpec.from_yaml`.

        Args:
            path: Path to the YAML config file.

        Returns:
            A ready-to-use :class:`OverlapDataManager` instance.
        """
        with open(path, "r") as fh:
            raw = yaml.safe_load(fh)

        overlap_spec: Optional[OverlapSpec] = None
        if raw.get("overlap_spec") is not None:
            overlap_spec = OverlapSpec.from_yaml(path)

        return cls(
            dataset_name=raw["dataset_name"],
            init_cls=int(raw["init_cls"]),
            increment=int(raw["increment"]),
            overlap_spec=overlap_spec,
            shuffle_seed=int(raw.get("shuffle_seed", 1993)),
            data_root=str(raw.get("data_root", "./data")),
            test_overlap_strategy=str(raw.get("test_overlap_strategy", "duplicate")),
        )

    # ------------------------------------------------------------------
    # PILOT-compatible internals (exposed for compatibility)
    # ------------------------------------------------------------------

    def getlen(self, index: int) -> int:
        """PILOT-compatible: count train samples of remapped class *index*."""
        if index in self._echo_sources:
            # Echo classes carry no rows in _train_targets; count the images
            # assigned to them across tasks instead.
            return int(sum(len(a.get(index, [])) for a in self._train_assignment))
        y = self._train_targets
        return int(np.sum(y == index))

    def get_echo_map(self) -> Dict[int, tuple]:
        """Return ``{echo_id: (source_id, image_relation)}`` for clone classes.

        Empty for non-echo scenarios.  The harness uses this to score echo
        classes as *returning* categories in the CLOVER overlap metrics.
        """
        return dict(self._echo_sources)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_raw(self, source: str) -> Tuple[np.ndarray, np.ndarray]:
        if source == "train":
            return self._train_data, self._train_targets
        elif source == "test":
            return self._test_data, self._test_targets
        raise ValueError(f"Unknown source {source!r}. Use 'train' or 'test'.")

    def _make_transform(self, mode: str) -> Any:
        if mode == "train":
            return transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "test":
            return transforms.Compose([*self._test_trsf, *self._common_trsf])
        elif mode == "flip":
            return transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
        raise ValueError(f"Unknown mode {mode!r}. Use 'train', 'test', or 'flip'.")

    def _collect_from_assignment(
        self,
        source: str,
        assignment: Dict[int, List[int]],
        m_rate: Optional[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        x, y = self._get_raw(source)
        data_list, targets_list = [], []
        for cls_id, img_idxs in assignment.items():
            if not img_idxs:
                continue
            idxs = np.array(img_idxs)
            if m_rate is not None and m_rate > 0:
                n_keep = int((1 - m_rate) * len(idxs))
                chosen = np.random.randint(0, len(idxs), size=n_keep)
                idxs = np.sort(idxs[chosen])
            data_list.append(x[idxs])
            # Label by the assignment KEY, not the raw target.  For real
            # classes the key equals the stored target; for echo (clone)
            # classes the key is the fresh id while the underlying images
            # still carry their source target, so the key is what we want.
            targets_list.append(np.full(len(idxs), cls_id, dtype=np.int64))
        if not data_list:
            return np.array([]), np.array([], dtype=int)
        return np.concatenate(data_list), np.concatenate(targets_list)

    def _assign_echo_images(
        self,
        train_c2i: Dict[int, List[int]],
        test_c2i: Dict[int, List[int]],
        base_seed: int,
    ) -> None:
        """Populate echo classes' image assignments from their source category.

        ``"same"`` reuses the source's full pool (duplicate images).  ``"new"``
        splits the source pool in half: the source's first appearance keeps one
        half and the echo gets the disjoint other half, so the revisit shows
        genuinely unseen images.  Test images always duplicate the full source
        pool so each echo class is evaluated on its whole category.
        """
        first_task: Dict[int, int] = {}
        for t, cls_list in enumerate(self._task_class_lists):
            for c in cls_list:
                first_task.setdefault(c, t)

        rng_echo = get_rng(base_seed + 7)
        for new_id, (source_id, relation) in self._echo_sources.items():
            echo_task = first_task.get(new_id)
            src_task = first_task.get(source_id)
            if echo_task is None or src_task is None:
                raise ValueError(
                    f"Echo class {new_id} (source {source_id}) is not placed "
                    f"in any task."
                )

            train_pool = list(train_c2i.get(source_id, []))
            test_pool = list(test_c2i.get(source_id, []))

            if relation == "same":
                self._train_assignment[echo_task][new_id] = list(train_pool)
            else:  # "new": disjoint split with the source's first appearance
                shuffled = list(rng_echo.permutation(train_pool))
                cut = len(shuffled) // 2
                self._train_assignment[src_task][source_id] = shuffled[:cut]
                self._train_assignment[echo_task][new_id] = shuffled[cut:]

            self._test_assignment[echo_task][new_id] = list(test_pool)

    def _echo_rows(self, source: str, echo_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(images, targets)`` for an echo class via the class-list path.

        Pulls the echo's assigned image indices (it lives in exactly one task)
        and labels them with the echo id, so a full-range test loader can
        evaluate the clone like any other class.
        """
        x, _ = self._get_raw(source)
        assignment = (
            self._train_assignment if source == "train" else self._test_assignment
        )
        all_idx: List[int] = []
        for task_map in assignment:
            all_idx.extend(task_map.get(echo_id, []))
        idxs = np.array(all_idx, dtype=int)
        return x[idxs], np.full(len(idxs), echo_id, dtype=np.int64)

    def _find_task_for_indices(self, indices: List[int]) -> int:
        """Return the first task whose class list is a superset of *indices*."""
        idx_set = set(indices)
        for t_id, cls_list in enumerate(self._task_class_lists):
            if idx_set.issubset(set(cls_list)):
                return t_id
        return 0  # fallback: use task 0 assignment


# ---------------------------------------------------------------------------
# Module-level helpers (mirror PILOT internals for correctness)
# ---------------------------------------------------------------------------


def _map_new_class_index(y: np.ndarray, order: List[int]) -> np.ndarray:
    """Remap original class IDs to their position in *order* (PILOT-compatible)."""
    inverse = np.empty(len(order), dtype=int)
    for remapped, original in enumerate(order):
        inverse[original] = remapped
    return inverse[y]


def _build_remapped_class_to_indices(
    remapped_targets: np.ndarray, n_classes: int
) -> Dict[int, List[int]]:
    c2i: Dict[int, List[int]] = {}
    for idx, label in enumerate(remapped_targets):
        c2i.setdefault(int(label), []).append(idx)
    return c2i


def _select(
    x: np.ndarray, y: np.ndarray, low: int, high: int
) -> Tuple[np.ndarray, np.ndarray]:
    idxes = np.where((y >= low) & (y < high))[0]
    return x[idxes], y[idxes]


def _select_rmm(
    x: np.ndarray, y: np.ndarray, low: int, high: int, m_rate: float
) -> Tuple[np.ndarray, np.ndarray]:
    if m_rate != 0:
        idxes = np.where((y >= low) & (y < high))[0]
        n_keep = int((1 - m_rate) * len(idxes))
        chosen = np.random.randint(0, len(idxes), size=n_keep)
        return x[np.sort(idxes[chosen])], y[np.sort(idxes[chosen])]
    idxes = np.where((y >= low) & (y < high))[0]
    return x[idxes], y[idxes]
