"""Built-in synthetic dataset (SPEC §12.2): fast, CPU, actually learnable.

20 classes x 32 samples of 8x8 "noise-with-signal": each class has a fixed
random pattern (shared across train/test splits, seeded once) plus
per-sample noise. A tiny linear model can beat chance on it in a handful of
steps -- needed for the revisit-safety gate's "accuracy above chance"
assertion. Hosts the gate (§5.3), the smoke profile (§9), and trainer/resume
tests in later phases.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from clover.datasets import register_dataset
from clover.datasets.base import CLDataset

_PATTERN_SEED = 1234


@register_dataset("synthetic")
class SyntheticDataset(CLDataset):
    input_size = 8
    NUM_CLASSES = 20
    SAMPLES_PER_CLASS = 32
    NOISE_STD = 0.3

    def __init__(self, root: str = ".", train: bool = True, transform: Optional[Any] = None) -> None:
        super().__init__(root, train, transform)

        pattern_rng = np.random.default_rng(_PATTERN_SEED)
        patterns = {
            c: pattern_rng.standard_normal((self.input_size, self.input_size)).astype(np.float32)
            for c in range(self.NUM_CLASSES)
        }

        # Same per-class pattern in both splits; only the noise draw (and
        # hence the RNG stream) differs, so a model trained on `train`
        # actually generalizes to `test`.
        sample_rng = np.random.default_rng(_PATTERN_SEED + (0 if train else 1))

        self._data: List[np.ndarray] = []
        self._targets: List[int] = []
        for c in range(self.NUM_CLASSES):
            for _ in range(self.SAMPLES_PER_CLASS):
                noise = sample_rng.standard_normal((self.input_size, self.input_size)).astype(np.float32)
                self._data.append(patterns[c] + noise * self.NOISE_STD)
                self._targets.append(c)

    @property
    def num_classes(self) -> int:
        return self.NUM_CLASSES

    def get_class_to_indices(self) -> Dict[int, List[int]]:
        result: Dict[int, List[int]] = {}
        for idx, c in enumerate(self._targets):
            result.setdefault(c, []).append(idx)
        return result

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        image = torch.from_numpy(self._data[idx])
        if self.transform is not None:
            image = self.transform(image)
        return image, self._targets[idx]

    def __len__(self) -> int:
        return len(self._data)
