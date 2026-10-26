
import numpy as np

from src.rpeak_detection import (
    detect_rpeaks,
    calculate_rr_intervals,
    evaluate_rpeaks
)


def test_rr_intervals():
    peaks = [100, 460, 820]

    rr = calculate_rr_intervals(peaks, 360)

    assert np.allclose(rr, [1.0, 1.0])


def test_evaluation():
    reference = [100, 460, 820]
    detected = [102, 458, 1000]

    result = evaluate_rpeaks(
        detected,
        reference,
        360
    )

    assert result["tp"] == 2
    assert result["fp"] == 1
    assert result["fn"] == 1


def test_flat_signal():
    signal = np.zeros(3600)

    result = detect_rpeaks(signal, 360)

    assert len(result["rpeaks"]) == 0
