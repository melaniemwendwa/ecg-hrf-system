
import pytest

from src.hrf_analysis import (
    calculate_pip,
    reference_nn_mask
)


def test_hard_inflections():
    rr = [0.80, 0.85, 0.90, 0.82, 0.78, 0.86]

    result = calculate_pip(rr, fs=360)

    assert result["pip"] == pytest.approx(50.0)
    assert result["hard_count"] == 2
    assert result["soft_count"] == 0
    assert result["comparison_count"] == 4


def test_constant_intervals():
    rr = [0.8, 0.8, 0.8, 0.8]

    result = calculate_pip(rr, fs=360)

    assert result["pip"] == 0.0
    assert result["inflection_count"] == 0


def test_soft_inflections():
    rr = [0.8, 0.9, 0.9, 1.0]

    result = calculate_pip(rr, fs=360)

    assert result["pip"] == pytest.approx(100.0)
    assert result["soft_count"] == 2


def test_invalid_intervals_break_sequence():
    rr = [0.8, 0.9, 0.8, 0.9, 0.8]
    mask = [True, True, False, True, True]

    result = calculate_pip(rr, fs=360, valid_mask=mask)

    assert result["pip"] is None
    assert result["comparison_count"] == 0


def test_insufficient_data():
    result = calculate_pip([0.8, 0.9], fs=360)

    assert result["pip"] is None


def test_negative_rr_rejected():
    with pytest.raises(ValueError):
        calculate_pip([0.8, -0.9, 0.7], fs=360)


def test_reference_nn_mask():
    symbols = ["N", "N", "V", "N", "N"]

    mask = reference_nn_mask(symbols)

    assert mask.tolist() == [
        True, False, False, True
    ]


def test_sampling_resolution():
    rr = [0.8, 0.801, 0.800, 0.81]

    result = calculate_pip(rr, fs=360)

    assert result["pip"] == pytest.approx(50.0)
    assert result["soft_count"] == 1
