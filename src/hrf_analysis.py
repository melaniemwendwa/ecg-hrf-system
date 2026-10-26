
import numpy as np


def calculate_pip(rr_intervals, fs, valid_mask=None):
    """
    Calculate comparison-based PIP.

    rr_intervals: Consecutive intervals in seconds.
    fs: ECG sampling frequency in Hz.
    valid_mask: Optional Boolean mask indicating
                eligible RR intervals.

    Invalid intervals break the sequence.
    """

    rr = np.asarray(rr_intervals, dtype=float)

    if rr.ndim != 1:
        raise ValueError("RR intervals must be 1D.")

    if not np.all(np.isfinite(rr)):
        raise ValueError("RR intervals must be finite.")

    if np.any(rr <= 0):
        raise ValueError("RR intervals must be positive.")

    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Invalid sampling frequency.")

    if valid_mask is None:
        valid = np.ones(len(rr), dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)

        if valid.ndim != 1 or len(valid) != len(rr):
            raise ValueError("Invalid interval mask length.")

    differences = np.diff(rr)

    # Classify changes according to ECG resolution.
    # +1 = RR increased
    # -1 = RR decreased
    #  0 = no measurable change

    resolution = 1.0 / fs
    tolerance = resolution * 1e-6

    states = np.zeros(len(differences), dtype=int)

    states[differences >= resolution - tolerance] = 1
    states[differences <= -resolution + tolerance] = -1

    inflection_indices = np.array([], dtype=int)

    if len(rr) < 3:
        return {
            "pip": None,
            "hard_count": 0,
            "soft_count": 0,
            "inflection_count": 0,
            "comparison_count": 0,
            "differences": differences,
            "states": states,
            "inflection_indices": inflection_indices,
            "eligible_comparisons": np.array([], dtype=bool)
        }

    # Three consecutive eligible RR intervals
    # are necessary for one PIP comparison.
    eligible = (
        valid[:-2]
        & valid[1:-1]
        & valid[2:]
    )

    before = states[:-1]
    after = states[1:]

    # Hard: direction reverses
    hard = (before * after == -1) & eligible

    # Soft: change to/from zero
    soft = (
        (before != after)
        & ((before == 0) | (after == 0))
        & eligible
    )

    inflections = hard | soft

    hard_count = int(np.sum(hard))
    soft_count = int(np.sum(soft))

    inflection_count = hard_count + soft_count
    comparison_count = int(np.sum(eligible))

    pip = (
        100.0 * inflection_count / comparison_count
        if comparison_count > 0
        else None
    )

    inflection_indices = (
        np.flatnonzero(inflections) + 1
    )

    return {
        "pip": pip,
        "hard_count": hard_count,
        "soft_count": soft_count,
        "inflection_count": inflection_count,
        "comparison_count": comparison_count,
        "differences": differences,
        "states": states,
        "inflection_indices": inflection_indices,
        "eligible_comparisons": eligible
    }


def reference_nn_mask(beat_types):
    """
    Conservatively flag RR intervals bounded
    by two MIT-BIH 'N' annotations.

    This is an initial NN approximation,
    not a complete sinus-rhythm validation.
    """

    symbols = np.asarray(beat_types)

    if symbols.ndim != 1:
        raise ValueError("Beat types must be 1D.")

    return (
        (symbols[:-1] == "N")
        & (symbols[1:] == "N")
    )
