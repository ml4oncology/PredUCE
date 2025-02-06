import pytest

import numpy as np

from preduce.util import compute_threshold


@pytest.mark.parametrize(
    "pred, desired_alarm_rate, expected_threshold, expected_alarm_rate",
    [
        ([0.1, 0.2, 0.3, 0.4, 0.5], 0.6, 0.2, 0.6),  # Basic case
        ([0.05, 0.1, 0.15, 0.2, 0.25], 0.4, 0.15, 0.4),  # Small values
        ([0.9, 0.8, 0.7, 0.6, 0.5], 0.4, 0.7, 0.4),  # Decreasing order
        ([0.2] * 10 + [0.8] * 10, 0.5, 0.2, 0.5),  # Half above threshold
        ([0.05, 0.1, 0.15, 0.2, 0.25], 0.99, None, None),  # No matching threshold
    ],
)
def test_compute_threshold(
    pred, desired_alarm_rate, expected_threshold, expected_alarm_rate
):
    pred = np.array(pred)
    result = compute_threshold(pred, desired_alarm_rate)
    if expected_threshold is None:
        assert result is None
    else:
        threshold, alarm_rate = result
        assert np.isclose(threshold, expected_threshold, atol=0.005)
        assert np.isclose(alarm_rate, expected_alarm_rate, atol=0.005)
