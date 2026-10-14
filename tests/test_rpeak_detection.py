
import numpy as np

from src.rpeak_detection import (
    detect_rpeaks,
    calculate_rr_intervals,
    evaluate_rpeak_detection
)


def test_rr_intervals():
    peaks = np.array([100, 460, 820])
    rr = calculate_rr_intervals(peaks, 360)

    assert np.allclose(rr, [1.0, 1.0])


def test_detection_metrics():
    reference = [100, 460, 820]
    detected = [102, 458, 1000]

    result = evaluate_rpeak_detection(
        detected, reference, 360
    )

    assert result["true_positives"] == 2
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1


def test_no_signal_peaks():
    signal = np.zeros(3600)

    result = detect_rpeaks(signal, 360)

    assert len(result["rpeaks"]) == 0
