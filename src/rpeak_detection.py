
import numpy as np
from scipy.signal import (
    butter,
    sosfiltfilt,
    find_peaks
)


def preprocess_ecg(signal, fs):
    """Bandpass filter the ECG from 5 to 20 Hz."""
    signal = np.asarray(signal, dtype=float)

    if signal.ndim != 1 or len(signal) < int(fs * 2):
        raise ValueError("Provide at least 2 seconds of ECG data.")

    if not np.all(np.isfinite(signal)):
        raise ValueError("ECG contains invalid values.")

    sos = butter(
        3,
        [5, 20],
        btype="bandpass",
        fs=fs,
        output="sos"
    )

    return sosfiltfilt(sos, signal)


def detect_rpeaks(signal, fs):
    """
    Detect R-peak candidates using a
    Pan-Tompkins-inspired processing pipeline.

    This is a baseline detector and requires
    validation against reference annotations.
    """

    filtered = preprocess_ecg(signal, fs)

    # Emphasize steep QRS changes.
    derivative = np.gradient(filtered)
    squared = derivative ** 2

    # Smooth with a 150 ms moving average.
    window_size = max(1, int(0.15 * fs))
    kernel = np.ones(window_size) / window_size

    integrated = np.convolve(
        squared, kernel, mode="same"
    )

    # Signal-dependent candidate threshold.
    baseline = np.median(integrated)
    high_level = np.percentile(integrated, 90)

    threshold = baseline + 0.4 * (
        high_level - baseline
    )

    # Avoid detecting two beats too close together.
    min_distance = max(1, int(0.25 * fs))

    candidates, _ = find_peaks(
        integrated,
        height=threshold,
        distance=min_distance,
        prominence=max(
            threshold - baseline,
            np.finfo(float).eps
        )
    )

    # Align candidates with the largest absolute
    # filtered ECG deflection nearby.
    search_radius = int(0.10 * fs)
    refined_peaks = []

    for candidate in candidates:
        start = max(0, candidate - search_radius)
        end = min(
            len(filtered),
            candidate + search_radius + 1
        )

        local_signal = np.abs(filtered[start:end])
        peak = start + int(np.argmax(local_signal))
        refined_peaks.append(peak)

    refined_peaks = np.unique(
        np.asarray(refined_peaks, dtype=int)
    )

    # Enforce a final refractory interval.
    accepted = []

    for peak in refined_peaks:
        if not accepted:
            accepted.append(int(peak))
            continue

        if peak - accepted[-1] >= min_distance:
            accepted.append(int(peak))
        elif abs(filtered[peak]) > abs(
            filtered[accepted[-1]]
        ):
            accepted[-1] = int(peak)

    return {
        "filtered_signal": filtered,
        "integrated_signal": integrated,
        "rpeaks": np.asarray(accepted, dtype=int),
        "threshold": threshold
    }


def calculate_rr_intervals(rpeaks, fs):
    """Return consecutive RR intervals in seconds."""
    rpeaks = np.asarray(rpeaks, dtype=int)

    if len(rpeaks) < 2:
        return np.array([], dtype=float)

    return np.diff(rpeaks) / fs


def evaluate_rpeak_detection(
    detected,
    reference,
    fs,
    tolerance_ms=150
):
    """
    Match detected beats with reference beats.

    A reference and detection can each match
    only once. Matches must fall within the
    specified timing tolerance.
    """

    detected = np.sort(np.asarray(detected, dtype=int))
    reference = np.sort(np.asarray(reference, dtype=int))

    tolerance = int(round(tolerance_ms * fs / 1000))

    i = 0
    j = 0
    tp = 0

    while i < len(detected) and j < len(reference):
        difference = detected[i] - reference[j]

        if abs(difference) <= tolerance:
            tp += 1
            i += 1
            j += 1
        elif difference < -tolerance:
            i += 1
        else:
            j += 1

    fp = len(detected) - tp
    fn = len(reference) - tp

    precision = tp / (tp + fp) if tp + fp else 0.0
    sensitivity = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * sensitivity /
        (precision + sensitivity)
        if precision + sensitivity else 0.0
    )

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "sensitivity": sensitivity,
        "f1_score": f1
    }
