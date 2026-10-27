import numpy as np
import pytest

from src.hrf_analysis import (
    calculate_pip,
    costa_beat_codes,
    costa_hrf,
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


def test_reference_nn_mask_bundle_branch_block():
    symbols = ["L", "L", "R", "A", "N"]

    mask = reference_nn_mask(symbols)

    assert mask.tolist() == [
        True, True, False, False
    ]


def test_sampling_resolution():
    rr = [0.8, 0.801, 0.800, 0.81]

    result = calculate_pip(rr, fs=360)

    assert result["pip"] == pytest.approx(50.0)
    assert result["soft_count"] == 1


def test_pip_excludes_intervals_outside_limits():
    # The 2.2 s gap is treated as a detection error
    # and breaks the sequence, as in HRF.awk.
    rr = [0.80, 0.85, 0.80, 2.20, 0.80, 0.85, 0.80]

    result = calculate_pip(rr, fs=250)

    assert result["comparison_count"] == 2


def beats(rr, non_sinus=()):
    """Beat times and types (1 = sinus) from intervals."""

    times = np.round(np.concatenate([[0], np.cumsum(rr)]), 3)
    types = np.ones(len(times), dtype=int)
    types[list(non_sinus)] = 2

    return times, types


# Expected counts below were produced by Costa's
# reference software (HRF.awk, PhysioNet) on the
# same beats, at its default 250 Hz.
REFERENCE_CASES = [
    (
        "steady then zig-zag",
        [0.80, 0.81, 0.82, 0.83, 0.84, 0.83, 0.84, 0.83, 0.84,
         0.83, 0.84, 0.83, 0.82, 0.81, 0.82, 0.82, 0.82, 0.83],
        (),
        {"inflection_count": 10, "nn_total": 13,
         "short": 7, "long": 3, "n_dnn": 12}
    ),
    (
        "premature beat",
        [0.80, 0.82, 0.81, 0.83, 0.82, 0.84, 0.50, 1.10, 0.83,
         0.82, 0.84, 0.83, 0.85, 0.84, 0.82, 0.83, 0.81],
        (7,),
        {"inflection_count": 10, "nn_total": 11,
         "short": 9, "long": 0, "n_dnn": 9}
    ),
    (
        "2.2 s gap",
        [0.80, 0.82, 0.81, 0.83, 0.82, 0.84, 0.83, 2.20, 0.82,
         0.84, 0.83, 0.85, 0.84, 0.82, 0.83, 0.81, 0.82],
        (),
        {"inflection_count": 11, "nn_total": 12,
         "short": 10, "long": 0, "n_dnn": 10}
    )
]


@pytest.mark.parametrize(
    "rr, non_sinus, expected",
    [case[1:] for case in REFERENCE_CASES],
    ids=[case[0] for case in REFERENCE_CASES]
)
def test_costa_hrf_matches_reference_software(rr, non_sinus, expected):
    result = costa_hrf(*beats(rr, non_sinus), fs=250)

    for key, value in expected.items():
        assert result[key] == value, key


def test_costa_hrf_zig_zag():
    # Direction flips at every beat: every segment
    # is one interval long, all in one alternation.
    result = costa_hrf(*beats([0.800, 0.820] * 6), fs=250)

    assert result["pip"] == pytest.approx(100.0)
    assert result["pnnss"] == pytest.approx(100.0)
    assert result["pnnls"] == pytest.approx(0.0)
    assert result["ials"] == pytest.approx(1.0)
    assert result["pas"] == pytest.approx(100.0)


def test_costa_hrf_one_long_segment():
    # Up 5, down 6, up 3: two inflection points
    # with one 6-interval segment between them.
    rr = [0.800, 0.804, 0.808, 0.812, 0.816, 0.820, 0.816,
          0.812, 0.808, 0.804, 0.800, 0.796, 0.800, 0.804, 0.808]

    result = costa_hrf(*beats(rr), fs=250)

    assert result["pip"] == pytest.approx(200 / 7)
    assert result["pnnss"] == pytest.approx(0.0)
    assert result["pnnls"] == pytest.approx(100.0)
    assert result["ials"] == pytest.approx(1 / 6)
    assert result["pas"] == pytest.approx(0.0)


def test_costa_hrf_non_sinus_beat_splits_alternation():
    # A non-sinus beat splits the zig-zag into
    # runs of 3 and 4 segments. Only the run of 4
    # is long enough to count as alternation.
    result = costa_hrf(
        *beats([0.800, 0.820] * 7 + [0.800], non_sinus=(7,)),
        fs=250
    )

    assert result["inflection_count"] == 9
    assert result["n_dnn"] == 7
    assert result["pas"] == pytest.approx(400 / 7)


def test_costa_hrf_too_few_beats():
    result = costa_hrf(*beats([0.8, 0.9]), fs=250)

    assert result["pip"] is None
    assert result["pas"] is None


def test_costa_beat_codes():
    codes = costa_beat_codes(["N", "L", "V", "A", "R"])

    assert codes.tolist() == [1, 1, 2, 2, 1]
