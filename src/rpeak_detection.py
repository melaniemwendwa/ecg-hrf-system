
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks


def preprocess_ecg(signal, fs):
    """Filter an ECG signal to emphasize QRS activity."""

    signal = np.asarray(signal, dtype=float)

    if signal.ndim != 1:
        raise ValueError("ECG must be a one-dimensional signal.")

    if fs <= 0:
        raise ValueError("Sampling frequency must be positive.")

    if len(signal) < 2 * fs:
        raise ValueError("At least 2 seconds of ECG are required.")

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
    Pan-Tompkins-inspired baseline R-peak detector.

    Uses filtered ECG, differentiation, squaring,
    moving-window integration, peak candidate
    detection, and local peak refinement.
    """

    filtered = preprocess_ecg(signal, fs)

    # Highlight rapid signal changes.
    derivative = np.gradient(filtered)
    squared = derivative ** 2

    # Integrate using a 150 ms window.
    window_size = max(1, int(round(0.15 * fs)))
    kernel = np.ones(window_size) / window_size

    integrated = np.convolve(
        squared,
        kernel,
        mode="same"
    )

    # If the ECG is flat, there are no detectable peaks.
    if np.max(integrated) <= 1e-12:
        return {
            "rpeaks": np.array([], dtype=int),
            "filtered_signal": filtered,
            "integrated_signal": integrated
        }

    baseline = np.median(integrated)
    high = np.percentile(integrated, 90)

    threshold = baseline + 0.4 * (high - baseline)

    # Candidate QRS complexes must be separated
    # by at least 250 milliseconds.
    min_distance = max(1, int(round(0.25 * fs)))

    candidates, _ = find_peaks(
        integrated,
        height=threshold,
        distance=min_distance,
        prominence=max(
            threshold - baseline,
            np.finfo(float).eps
        )
    )

    # Refine candidate locations using the
    # largest absolute filtered ECG deflection.
    radius = int(round(0.10 * fs))
    refined = []

    for candidate in candidates:
        start = max(0, candidate - radius)
        end = min(len(filtered), candidate + radius + 1)

        local = np.abs(filtered[start:end])
        peak = start + int(np.argmax(local))

        refined.append(peak)

    refined = np.unique(refined)

    # Enforce a refractory period after refinement.
    accepted = []

    for peak in refined:
        peak = int(peak)

        if not accepted:
            accepted.append(peak)

        elif peak - accepted[-1] >= min_distance:
            accepted.append(peak)

        elif abs(filtered[peak]) > abs(
            filtered[accepted[-1]]
        ):
            accepted[-1] = peak

    return {
        "rpeaks": np.asarray(accepted, dtype=int),
        "filtered_signal": filtered,
        "integrated_signal": integrated
    }


def calculate_rr_intervals(peaks, fs):
    """Calculate RR intervals in seconds."""

    peaks = np.sort(np.asarray(peaks, dtype=int))

    if len(peaks) < 2:
        return np.array([], dtype=float)

    return np.diff(peaks) / fs


def evaluate_rpeaks(
    detected,
    reference,
    fs,
    tolerance_ms=150
):
    """
    Compare detected peaks with reference beats.

    Each detected peak and reference beat
    can be matched only once.
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

    precision = (
        tp / (tp + fp)
        if tp + fp > 0 else 0.0
    )

    sensitivity = (
        tp / (tp + fn)
        if tp + fn > 0 else 0.0
    )

    f1 = (
        2 * precision * sensitivity /
        (precision + sensitivity)
        if precision + sensitivity > 0
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "sensitivity": sensitivity,
        "f1": f1
    }
