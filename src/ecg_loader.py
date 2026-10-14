
import numpy as np
import wfdb


# MIT-BIH Arrhythmia Database
DATABASE = "mitdb"

# Records available in our initial interface
AVAILABLE_RECORDS = [
    "100", "101", "102",
    "103", "105", "106"
]

# Standard WFDB beat annotation symbols
BEAT_SYMBOLS = {
    "N", "L", "R", "B",
    "A", "a", "J", "S",
    "V", "r", "F", "e",
    "j", "n", "E", "/",
    "f", "Q", "?"
}


def load_ecg_record(record_id="100", duration=10):
    """
    Load an ECG segment and reference annotations
    from the MIT-BIH Arrhythmia Database.

    Returns the waveform, sampling frequency,
    channel names, and annotated beat positions.
    """

    if record_id not in AVAILABLE_RECORDS:
        raise ValueError("Unsupported ECG record ID.")

    if not 5 <= duration <= 60:
        raise ValueError(
            "Duration must be between 5 and 60 seconds."
        )

    # MIT-BIH signals are sampled at 360 Hz.
    fs = 360
    sample_count = int(duration * fs)

    # Load the ECG waveform
    record = wfdb.rdrecord(
        record_id,
        sampfrom=0,
        sampto=sample_count,
        pn_dir=DATABASE
    )

    # Load expert reference annotations
    annotations = wfdb.rdann(
        record_id,
        "atr",
        sampfrom=0,
        sampto=sample_count,
        pn_dir=DATABASE
    )

    signals = np.asarray(
        record.p_signal,
        dtype=float
    )

    actual_fs = float(record.fs)

    time = np.arange(
        signals.shape[0]
    ) / actual_fs

    # Extract only actual beat annotations
    beats = [
        (int(sample), symbol)
        for sample, symbol in zip(
            annotations.sample,
            annotations.symbol
        )
        if (
            symbol in BEAT_SYMBOLS
            and 0 <= sample < len(time)
        )
    ]

    reference_peaks = np.array(
        [sample for sample, symbol in beats],
        dtype=int
    )

    beat_types = [
        symbol for sample, symbol in beats
    ]

    # Time between consecutive annotated beats
    rr_intervals = (
        np.diff(reference_peaks) / actual_fs
    )

    # Estimated average heart rate in BPM
    mean_heart_rate = None

    if len(rr_intervals) > 0:
        mean_rr = np.mean(rr_intervals)
        mean_heart_rate = 60 / mean_rr

    return {
        "record_id": record_id,
        "signals": signals,
        "time": time,
        "fs": actual_fs,
        "channel_names": list(record.sig_name),
        "reference_peaks": reference_peaks,
        "beat_types": beat_types,
        "rr_intervals": rr_intervals,
        "mean_heart_rate": mean_heart_rate,
        "duration": len(time) / actual_fs
    }
