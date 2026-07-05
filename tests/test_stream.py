"""Stream/Benchmark stats + end-to-end build_benchmark (SPEC §3-§4)."""

from __future__ import annotations

import pytest

from clover.core.spec import DatasetInfo, RevisitSpec, StreamSpec
from clover.core.stream import Benchmark, Stream, build_benchmark


def _synthetic_class_to_indices(num_classes: int, per_class: int = 8) -> dict[int, list[int]]:
    return {c: list(range(c * per_class, (c + 1) * per_class)) for c in range(num_classes)}


def test_stream_rejects_bad_split():
    with pytest.raises(ValueError, match="split must be"):
        Stream([], split="validation")


def test_disjoint_benchmark_has_no_revisits_and_full_overlap_matrix_is_diagonal():
    spec = StreamSpec(dataset="synthetic", init_cls=5, increment=5)
    info = DatasetInfo("synthetic", 20)
    class_to_indices = _synthetic_class_to_indices(20)

    benchmark = build_benchmark(spec, info, class_to_indices, class_to_indices)

    assert isinstance(benchmark, Benchmark)
    assert benchmark.nb_experiences == 4
    assert benchmark.total_classes == 20
    assert benchmark.train_stream.revisit_density() == 0.0

    matrix = benchmark.overlap_matrix()
    for i in range(benchmark.nb_experiences):
        for j in range(benchmark.nb_experiences):
            if i != j:
                assert matrix[i, j] == 0


def test_exact_replay_benchmark_has_one_revisit_experience():
    spec = StreamSpec(
        dataset="synthetic",
        init_cls=5,
        increment=5,
        revisits=[RevisitSpec(classes="task0", placement="end_of_stream", label="new", images="same")],
    )
    info = DatasetInfo("synthetic", 20)
    class_to_indices = _synthetic_class_to_indices(20)

    benchmark = build_benchmark(spec, info, class_to_indices, class_to_indices)

    # Echo ids are brand-new label ids (never seen before), so they count as
    # first_appearance_of at the identity level -- same as v1's own
    # revisiting_classes/is_revisit_experience semantics. The revisit signal
    # for echoes lives in echo_map, not revisit_density().
    last_experience = benchmark.train_stream[-1]
    assert not last_experience.is_revisit_experience()
    assert set(last_experience.first_appearance_of) == set(last_experience.classes_in_this_experience)
    assert last_experience.echo_map  # the revisit signal for label="new"
    # echo images are identical to task 0's (image_relation="same")
    first_experience = benchmark.train_stream[0]
    for echo_id in last_experience.echo_map:
        source_id = last_experience.echo_map[echo_id].source_id
        assert last_experience.image_indices[echo_id] == first_experience.image_indices[source_id]


def test_average_overlap_and_class_appearance_count():
    spec = StreamSpec(
        dataset="synthetic",
        init_cls=5,
        increment=5,
        revisits=[RevisitSpec(classes=[0, 1, 2], placement="every_task", label="same", images="new")],
    )
    info = DatasetInfo("synthetic", 20)
    class_to_indices = _synthetic_class_to_indices(20)
    benchmark = build_benchmark(spec, info, class_to_indices, class_to_indices)

    counts = benchmark.train_stream.class_appearance_count()
    assert counts[0] == benchmark.nb_experiences  # anchor appears in every task
    assert benchmark.train_stream.average_overlap() > 0
