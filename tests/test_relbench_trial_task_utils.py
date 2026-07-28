from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from examples.relbench_trial_task_utils import (
    SITE_SUCCESS_FEATURE_FAMILIES,
    _select_evenly_spaced_timestamps,
)
from examples.relbench_catboost_utils import TEMPORAL_FEATURE_FAMILIES


def test_site_success_uses_temporal_feature_families():
    assert SITE_SUCCESS_FEATURE_FAMILIES == TEMPORAL_FEATURE_FAMILIES


def test_trial_training_timestamp_sampling_keeps_full_range():
    timestamps = list(pd.date_range("2001-01-01", periods=19, freq="YS"))

    selected = _select_evenly_spaced_timestamps(timestamps, 5)

    assert len(selected) == 5
    assert selected[0] == timestamps[0]
    assert selected[-1] == timestamps[-1]
    assert selected == [timestamps[index] for index in (0, 4, 9, 14, 18)]


def test_trial_training_timestamp_sampling_does_not_oversample_short_ranges():
    timestamps = list(pd.date_range("2020-01-01", periods=3, freq="D"))

    assert _select_evenly_spaced_timestamps(timestamps, 5) == timestamps
