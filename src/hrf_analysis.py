
import numpy as np


def calculate_pip(
    rr_intervals,
    valid_mask=None,
    resolution=0.0
):
    """
    Calculate Percentage of Inflection Points.

    Parameters
    ----------
    rr_intervals : array-like
        Consecutive RR or NN intervals in seconds.

    valid_mask : array-like of bool, optional
        True for eligible intervals.
        Invalid intervals break the sequence.

    resolution : float
        Changes smaller than this threshold
        are treated as zero.

    Returns
    -------
    dict
        PIP result and intermediate calculations.
    """

    rr = np.asarray(rr_intervals, dtype=float)

    if rr.ndim != 1:
        raise ValueError("RR intervals must be 1D.")

    if not np.all(np.isfinite(rr)):
        raise ValueError("Invalid RR interval values.")

    if np.any(rr <= 0):
        raise ValueError("RR intervals must be positive.")

    if resolution < 0 or not np.isfinite(resolution):
        raise ValueError("Invalid resolution.")

    if valid_mask is None:
        valid = np.ones(len(rr), dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)

        if len(valid) != len(rr) or valid.ndim != 1:
            raise ValueError(
                "Mask must match RR interval length."
            )

    # Need 3 intervals to evaluate one inflection.
    if len(rr) < 3:
        return {
            "pip": None,
            "inflection_count": 0,
            "comparison_count": 0,
            "hard_count": 0,
            "soft_count": 0,
            "differences": np.diff(rr),
            "states": np.array([], dtype=int),
            "inflection_indices": np.array([], dtype=int)
        }

    # Differences between adjacent RR intervals.
    differences = np.diff(rr)

    # +1: RR increases (HR decelerates)
    # -1: RR decreases (HR accelerates)
    #  0: No meaningful change
    states = np.zeros(len(differences), dtype=int)

    states[differences >= resolution] = 1
    states[differences <= -resolution] = -1

    # Special handling of exact zero if resolution=0.
    states[differences == 0] = 0

    # A valid inflection requires three consecutive
    # eligible RR intervals.
    eligible = (
        valid[:-2]
        & valid[1:-1]
        & valid[2:]
    )

    before = states[:-1]
    after = states[1:]

    # A change between -1, 0, +1 is an inflection.
    changed = before != after

    hard = (
        (before * after == -1)
        & eligible
    )

    soft = (
        changed
        & ~hard
        & eligible
    )

    inflections = changed & eligible

    comparison_count = int(np.sum(eligible))
    inflection_count = int(np.sum(inflections))

    pip = (
        100 * inflection_count / comparison_count
        if comparison_count > 0
        else None
    )

    return {
        "pip": pip,
        "inflection_count": inflection_count,
        "comparison_count": comparison_count,
        "hard_count": int(np.sum(hard)),
        "soft_count": int(np.sum(soft)),
        "differences": differences,
        "states": states,
        "inflection_indices": (
            np.where(inflections)[0] + 1
        )
    }
