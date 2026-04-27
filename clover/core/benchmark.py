"""Benchmark — top-level object bundling train/test streams and the spec."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from clover.core.data_manager import OverlapDataManager
    from clover.core.stream import Stream
    from clover.core.stream_spec import StreamSpec


class Benchmark:
    """Top-level user-facing object for a CLOVER stream experiment.

    Bundles ``train_stream``, ``test_stream``, the declarative
    :class:`~clover.core.stream_spec.StreamSpec` that produced them, and the
    underlying :class:`~clover.core.data_manager.OverlapDataManager` engine.

    Construct via :func:`~clover.core.stream_builder.build_benchmark` or
    :meth:`from_yaml`.

    Args:
        train_stream: Ordered training experiences.
        test_stream: Ordered test experiences.
        stream_spec: The declarative spec used to build the streams.
        underlying_manager: The low-level engine (exposed for power users).
    """

    def __init__(
        self,
        train_stream: "Stream",
        test_stream: "Stream",
        stream_spec: "StreamSpec",
        underlying_manager: "OverlapDataManager",
    ) -> None:
        self.train_stream = train_stream
        self.test_stream = test_stream
        self.stream_spec = stream_spec
        self.engine = underlying_manager

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Descriptive identifier derived from the spec."""
        return (
            f"{self.stream_spec.dataset}"
            f"_init{self.stream_spec.init_cls}"
            f"_inc{self.stream_spec.increment}"
            f"_seed{self.stream_spec.stream_seed}"
        )

    @property
    def nb_experiences(self) -> int:
        """Number of experiences (same for train and test streams)."""
        return len(self.train_stream)

    @property
    def total_classes(self) -> int:
        """Total unique classes in the underlying dataset."""
        return self.engine.nb_classes

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def overlap_matrix(self) -> np.ndarray:
        """Return the ``[T × T]`` class-overlap matrix from the engine.

        Delegates to :meth:`~clover.core.data_manager.OverlapDataManager.get_overlap_matrix`.
        """
        return self.engine.get_overlap_matrix()

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def save_manifest(self, path: str) -> None:
        """Save a JSON manifest including stream-level metadata.

        Args:
            path: Destination file path.
        """
        import datetime

        from clover import __version__

        stream_meta = {
            "nb_experiences": self.nb_experiences,
            "revisit_density": self.train_stream.revisit_density(),
            "average_overlap": self.train_stream.average_overlap(),
            "class_appearance_count": self.train_stream.class_appearance_count(),
        }
        manifest = {
            "_header": {
                "clover_version": __version__,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "benchmark_name": self.name,
                "stream_spec": self.stream_spec.to_dict(),
            },
            "_stream_metrics": stream_meta,
            "train": self.engine.get_manifest(),
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as fh:
            json.dump(manifest, fh, indent=2)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> "Benchmark":
        """Build a :class:`Benchmark` from a YAML stream-spec config file.

        Args:
            path: Path to YAML file conforming to the :class:`~clover.core.stream_spec.StreamSpec`
                schema.

        Returns:
            A ready-to-use :class:`Benchmark`.
        """
        from clover.core.stream_builder import build_benchmark
        from clover.core.stream_spec import StreamSpec

        spec = StreamSpec.from_yaml(path)
        return build_benchmark(spec)
