"""Standard metrics correctness (SPEC §10): hand-computed R-matrix fixtures."""

from __future__ import annotations

import math

import numpy as np

from clover.evaluation.metrics import (
    aggregate_accuracy,
    average_incremental_accuracy,
    backward_transfer,
    forgetting,
    forward_transfer,
)

nan = float("nan")


def _make_R(rows):
    return np.array(rows, dtype=float)


# R[evaluated, trained]: task 0 eval'd after training 0/1/2; task 1 eval'd
# after training 1/2; task 2 eval'd only after training 2.
_R = _make_R(
    [
        [0.9, 0.85, 0.88],
        [nan, 0.8, 0.75],
        [nan, nan, 0.7],
    ]
)


def test_aggregate_accuracy_hand_computed():
    assert math.isclose(aggregate_accuracy(_R, 0), 0.9)
    assert math.isclose(aggregate_accuracy(_R, 1), 0.825)
    assert math.isclose(aggregate_accuracy(_R, 2), 0.7766666666666667)


def test_average_incremental_accuracy_hand_computed():
    assert math.isclose(average_incremental_accuracy(_R, 0), 0.9)
    assert math.isclose(average_incremental_accuracy(_R, 1), 0.8625)
    assert math.isclose(average_incremental_accuracy(_R, 2), 0.8338888888888889)


def test_backward_transfer_is_nan_at_t0_then_hand_computed():
    assert math.isnan(backward_transfer(_R, 0))
    assert math.isclose(backward_transfer(_R, 1), -0.05)
    assert math.isclose(backward_transfer(_R, 2), -0.035)


def test_forgetting_is_nan_at_t0_then_hand_computed():
    assert math.isnan(forgetting(_R, 0))
    assert math.isclose(forgetting(_R, 1), 0.05)
    assert math.isclose(forgetting(_R, 2), 0.035)


def test_forward_transfer_is_nan_at_t0_then_hand_computed():
    assert math.isnan(forward_transfer(_R, 0))
    assert math.isclose(forward_transfer(_R, 1), 0.9)
    assert math.isclose(forward_transfer(_R, 2), 0.85)


def test_backward_transfer_positive_when_revisit_improves_old_task():
    # Task 0's accuracy improves after training task 1 -- BWT should be positive.
    R = _make_R([[0.5, 0.9], [nan, 0.8]])
    assert math.isclose(backward_transfer(R, 1), 0.4)


def test_forgetting_is_zero_when_no_drop_occurred():
    R = _make_R([[0.5, 0.9], [nan, 0.8]])  # task 0's peak IS its value at t=1
    assert math.isclose(forgetting(R, 1), 0.0)


def test_all_nan_column_gives_nan_aggregate_accuracy():
    R = _make_R([[nan, nan], [nan, nan]])
    assert math.isnan(aggregate_accuracy(R, 1))
