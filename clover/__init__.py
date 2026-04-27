__version__ = "0.2.0"

from clover.core.benchmark import Benchmark
from clover.core.data_manager import OverlapDataManager
from clover.core.experience import Experience
from clover.core.overlap_spec import ImageSplit, OverlapPair, OverlapSpec
from clover.core.stream import Stream
from clover.core.stream_builder import build_benchmark
from clover.core.stream_spec import RevisitSpec, StreamSpec

__all__ = [
    # Modern stream API
    "build_benchmark",
    "Benchmark",
    "Stream",
    "Experience",
    "StreamSpec",
    "RevisitSpec",
    # Legacy / low-level engine API (v0.1 back-compat)
    "OverlapDataManager",
    "OverlapSpec",
    "OverlapPair",
    "ImageSplit",
]
