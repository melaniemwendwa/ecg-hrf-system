
import numpy as np
import wfdb


def load_ecg_record(record_name="100", duration=10):
    """
    Load an ECG segment and expert annotations
    from the MIT-BIH Arrhythmia Database.
    """

    # MIT-BIH records are sampled at 360 Hz.
    fs = 360
    sample_count = int(duration * fs)

    record = wfdb.rdrecord(
        record_name,
        sampfrom=0,
        sampto=sample_count,
        pn_dir="mitdb"
    )

    annotation = wfdb.rdann(
        record_name,
        "atr",
        sampfrom=0,
        sampto=sample_count,
        pn_dir="mitdb"
    )

    signal = record.p_signal[:, 0]
    actual_fs = record.fs

    time = np.arange(len(signal)) / actual_fs

    # Keep annotations that represent actual beats.
    beat_symbols = {
        "N", "L", "R", "B", "A", "a", "J", "S",
        "V", "r", "F", "e", "j", "n", "E",
        "/", "f", "Q", "?"
    }

    peaks = np.array(
        [
            sample
            for sample, symbol in zip(
                annotation.sample,
                annotation.symbol
            )
            if symbol in beat_symbols
        ],
        dtype=int
    )

    peaks = peaks[
        (peaks >= 0) & (peaks < len(signal))
    ]

    rr_intervals = np.diff(peaks) / actual_fs

    heart_rate = (
        60 / np.mean(rr_intervals)
        if len(rr_intervals) > 0
        else None
    )

    return {
        "record_name": record_name,
        "signal": signal,
        "time": time,
        "fs": actual_fs,
        "peaks": peaks,
        "rr_intervals": rr_intervals,
        "heart_rate": heart_rate,
        "channel": record.sig_name[0]
    }
