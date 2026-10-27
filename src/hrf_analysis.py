import numpy as np


# NN intervals outside these limits (seconds) are
# treated as detection errors, as in the reference
# software HRF.awk. Its README states 1.5 s, but the
# code uses 1.8 s; the code is followed here.
NN_MIN = 0.3
NN_MAX = 1.8

# Alternation segments need at least this many
# NN intervals (Costa et al. 2017).
MIN_ALTERNATION = 4


def calculate_pip(
    rr_intervals,
    fs,
    valid_mask=None,
    nn_min=NN_MIN,
    nn_max=NN_MAX
):
    """
    Calculate comparison-based PIP, with per-interval
    detail used for plotting.

    rr_intervals: Consecutive intervals in seconds.
    fs: ECG sampling frequency in Hz.
    valid_mask: Optional Boolean mask indicating
                eligible RR intervals.

    Invalid intervals, and intervals outside
    (nn_min, nn_max), break the sequence.

    The denominator here is every eligible
    comparison. The reference PIP (costa_hrf)
    instead counts the NN intervals between the
    first and last inflection points.
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

    valid = valid & (rr > nn_min) & (rr < nn_max)

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


def _round3(value):
    """Round like the reference software (%.3lf)."""

    return float(f"{value:.3f}")


def costa_hrf(beat_times, beat_types, fs, nn_min=NN_MIN, nn_max=NN_MAX):
    """
    Heart Rate Fragmentation measures, translated
    line by line from Costa's reference software
    (HRF.awk, PhysioNet heart-rate-fragmentation-code
    1.0.0), including its rounding and exclusions.

    beat_times: R-peak times in seconds.
    beat_types: 1 for normal sinus beats, any other
                value for non-sinus beats.

    PIP, PNNSS and PNNLS reproduce the reference
    software. IALS and PAS are not in the software;
    they are computed from the same segments,
    following the definitions in Costa et al.,
    Front Physiol 2017;8:255.

    A segment is the run of NN intervals between
    two consecutive inflection points. Segments
    touching a non-sinus beat or an NN interval
    outside (nn_min, nn_max) are excluded.
    """

    times = np.asarray(beat_times, dtype=float)
    types = np.asarray(beat_types)

    if times.ndim != 1 or len(times) != len(types):
        raise ValueError("Need one beat type per beat time.")

    # Variables start at zero, as in AWK.
    t_1 = t_2 = t_3 = 0.0
    ann_1 = ann_2 = ann_3 = 0
    ip = n_dnn = short = long = cont = nn_total = 0
    non_sinus = outlier = 0

    # Counted segments as (length, flat); None marks
    # an excluded segment, which breaks alternation.
    segments = []

    for nr, (t, ann) in enumerate(zip(times, types), start=1):

        # Snap the beat time to the 1/fs grid.
        aux = int(_round3(t * fs))

        if t - aux / fs > 0.5:
            t = (aux + 1) / fs
        if 0 < t - aux / fs <= 0.5:
            t = aux / fs

        nn = t - t_1
        nn_1 = t_1 - t_2
        nn_2 = t_2 - t_3
        dnn = _round3(t - 2 * t_1 + t_2)
        dnn_1 = _round3(t_1 - 2 * t_2 + t_3)

        # Current beat and the 3 before it must be sinus.
        if ann == 1 and ann_1 == 1 and ann_2 == 1 and ann_3 == 1:
            flag_1 = 0
        else:
            flag_1 = 1
            non_sinus = 1

        if nr <= 4 and (ann < 1 or ann_1 < 1 or ann_2 < 1 or ann_3 < 1):
            non_sinus = 0

        # The 3 latest NN intervals must be in range.
        if (
            nn_min < nn < nn_max
            and nn_min < nn_1 < nn_max
            and nn_min < nn_2 < nn_max
        ):
            flag_2 = 0
        else:
            flag_2 = 1
            outlier = 1

        # The reference compares an undefined NN_3,
        # which is always 0, so this always applies.
        if nr <= 4:
            outlier = 0

        hard = dnn * dnn_1 < 0
        soft = dnn * dnn_1 == 0 and dnn != dnn_1

        if nr > 3 and flag_1 == 0 and flag_2 == 0 and (hard or soft):
            ip += 1

            # Soft inflection after a run of zero changes
            flat = soft and dnn_1 == 0

            if non_sinus != 1 and outlier != 1:
                if ip > 1:
                    n_dnn += cont

                    if cont <= 2 and not flat:
                        short += cont
                    if cont > 2:
                        long += cont

                    segments.append((cont, flat))

                nn_total = 1 if ip == 1 else nn_total + cont

            else:
                non_sinus = 0
                outlier = 0
                nn_total += 1
                segments.append(None)

            cont = 0

        if nr > 1:
            cont += 1

        ann_3, ann_2, ann_1 = ann_2, ann_1, ann
        t_3, t_2, t_1 = t_2, t_1, t

    accel_decel = [
        length for length, flat in filter(None, segments)
        if not flat
    ]

    # Alternation: consecutive one-interval
    # acceleration/deceleration segments, i.e. the
    # sign of the change flips at every beat.
    pas_count = 0
    run = 0

    for segment in [*segments, None]:
        if segment is not None and segment == (1, False):
            run += 1
            continue

        if run >= MIN_ALTERNATION:
            pas_count += run
        run = 0

    return {
        "pip": 100.0 * ip / nn_total if n_dnn > 0 else None,
        "pnnss": (
            100.0 * short / (short + long)
            if short + long > 0 else None
        ),
        "pnnls": 100.0 * long / n_dnn if n_dnn > 0 else None,
        "ials": (
            len(accel_decel) / sum(accel_decel)
            if accel_decel else None
        ),
        "pas": 100.0 * pas_count / n_dnn if n_dnn > 0 else None,
        "inflection_count": ip,
        "nn_total": nn_total,
        "n_dnn": n_dnn,
        "short": short,
        "long": long
    }


def costa_beat_codes(beat_types):
    """Map annotation symbols to 1 (sinus) or 2 (other)."""

    return np.where(np.isin(beat_types, NORMAL_BEATS), 1, 2)


# Beats conducted from the sinus node.
# L and R are normal beats with bundle branch
# block; e and j are atrial/nodal escape beats.
NORMAL_BEATS = ["N", "L", "R", "e", "j"]


def reference_nn_mask(beat_types):
    """
    Flag RR intervals bounded by two normal
    beat annotations (see NORMAL_BEATS).

    This is an NN approximation from beat
    labels, not a sinus-rhythm validation.
    """

    symbols = np.asarray(beat_types)

    if symbols.ndim != 1:
        raise ValueError("Beat types must be 1D.")

    normal = np.isin(symbols, NORMAL_BEATS)

    return normal[:-1] & normal[1:]
