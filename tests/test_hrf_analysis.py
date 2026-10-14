
import numpy as np
import pytest

from src.hrf_analysis import calculate_pip


def test_two_hard_inflections():
    rr = [0.80, 0.85, 0.90, 0.83, 0.78, 0.86]
    result = calculate_pip(rr)

    assert result["pip"] == pytest.approx(50.0)
    assert result["hard_count"] == 2
    assert result["soft_count"] == 0
    assert result["comparison_count"] == 4


def test_constant_rr():
    rr = [0.8, 0.8, 0.8, 0.8]
    result = calculate_pip(rr)

    assert result["pip"] == 0.0


def test_soft_inflections():
    rr = [0.8, 0.9, 0.9, 1.0]
    result = calculate_pip(rr)

    assert result["pip"] == 100.0
    assert result["soft_count"] == 2


def test_invalid_interval_breaks_sequence():
    rr = [0.8, 0.9, 0.8, 0.9, 0.8]
    mask = [True, True, False, True, True]

    result = calculate_pip(rr, valid_mask=mask)

    assert result["comparison_count"] == 0
    assert result["pip"] is None


def test_insufficient_intervals():
    result = calculate_pip([0.8, 0.9])

    assert result["pip"] is None
    assert result["comparison_count"] == 0


def test_negative_interval_rejected():
    with pytest.raises(ValueError):
        calculate_pip([0.8, -0.9, 0.7])


def test_sampling_resolution():
    rr = [0.8, 0.801, 0.800, 0.81]

    result = calculate_pip(
        rr,
        resolution=1 / 360
    )

    assert result["soft_count"] == 1
    assert result["comparison_count"] == 2
    assert result["pip"] == pytest.approx(50.0)
