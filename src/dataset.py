import argparse
import bisect
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from src.ecg_loader import BEAT_SYMBOLS
from src.hrf_analysis import (
    NORMAL_BEATS,
    costa_beat_codes,
    costa_hrf
)


RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")

# All 48 MIT-BIH Arrhythmia Database records
MITDB_RECORDS = [
    "100", "101", "102", "103", "104", "105",
    "106", "107", "108", "109", "111", "112",
    "113", "114", "115", "116", "117", "118",
    "119", "121", "122", "123", "124", "200",
    "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215",
    "217", "219", "220", "221", "222", "223",
    "228", "230", "231", "232", "233", "234"
]

# Records with paced beats are commonly excluded
# from MIT-BIH rhythm studies.
PACED_RECORDS = {"102", "104", "107", "217"}

# Supported PhysioNet databases.
#   records: fixed record list, or None to use
#            every record downloaded locally
#   patient_of_record: records that share a patient
DATABASES = {
    "mitdb": {
        "records": MITDB_RECORDS,
        "excluded": PACED_RECORDS,
        # Records 201 and 202 come from the same patient.
        "patient_of_record": {"202": "201"}
    },
    "ltafdb": {
        "records": None,
        "excluded": set(),
        "patient_of_record": {}
    }
}

NORMAL_RHYTHM = "(N"

# Beat times from every database are rounded to
# this rate, so a "no change" in PIP means the
# same time difference everywhere. 128 Hz is the
# Long-Term AF Database rate (MIT-BIH is 360 Hz).
COMMON_FS = 128

# Each patient contributes at most this much
# data per class, so 24-hour recordings do not
# outweigh 30-minute ones.
MAX_MINUTES_PER_CLASS = 30

RANDOM_STATE = 42


def patient_id(database, record_id):
    """
    Identify the patient behind a record.

    Prefixed with the database name, because
    different databases reuse record names.
    """

    mapping = DATABASES[database]["patient_of_record"]

    return f"{database}:{mapping.get(record_id, record_id)}"


def to_common_rate(samples, fs, common_fs=COMMON_FS):
    """Convert sample positions to another sampling rate."""

    samples = np.asarray(samples, dtype=float)

    return np.round(samples * common_fs / fs).astype(int)


def rhythm_spans(rhythm_samples, rhythm_labels, signal_length):
    """
    Convert rhythm change annotations into
    (start, end, label) sample spans.

    Each rhythm lasts until the next rhythm
    annotation or the end of the record.
    """

    samples = np.asarray(rhythm_samples, dtype=int)

    if len(samples) != len(rhythm_labels):
        raise ValueError("Each rhythm needs one label.")

    order = np.argsort(samples, kind="stable")
    samples = samples[order]
    labels = [rhythm_labels[i] for i in order]

    spans = []

    for i, (start, label) in enumerate(zip(samples, labels)):
        end = (
            samples[i + 1]
            if i + 1 < len(samples)
            else signal_length
        )

        if end > start:
            spans.append((int(start), int(end), label))

    return spans


def window_rhythm(spans, start, end):
    """
    Return the single rhythm covering a window,
    or None if the window has no rhythm or
    contains a rhythm change.

    Spans must be sorted and non-overlapping,
    as returned by rhythm_spans.
    """

    starts = [span_start for span_start, _, _ in spans]

    # Last span starting at or before the window
    i = bisect.bisect_right(starts, start) - 1

    if i < 0:
        return None

    _, span_end, label = spans[i]

    return label if end <= span_end else None


# HRF measures taken from costa_hrf for NN intervals
HRF_MEASURES = ["pip", "pnnss", "pnnls", "ials", "pas"]


def window_features(beat_samples, beat_types, fs, start, end):
    """
    Calculate HRF features for beats inside
    one window [start, end).

    HRF measures use NN intervals only (normal
    beats, see NORMAL_BEATS), following Costa's
    reference software. PIP is also calculated
    treating every beat as normal ("all RR").

    comparisons_nn is the PIP denominator: the
    NN intervals between the first and last
    inflection points.

    Beat samples must be sorted.
    """

    beat_samples = np.asarray(beat_samples, dtype=int)
    beat_types = np.asarray(beat_types)

    first, last = np.searchsorted(beat_samples, [start, end])
    samples = beat_samples[first:last]
    types = beat_types[first:last]

    features = {
        "n_beats": int(len(samples)),
        "frac_non_normal_beats": (
            float(np.mean(~np.isin(types, NORMAL_BEATS)))
            if len(types) > 0 else None
        ),
        "mean_rr": None,
        "pip_all": None,
        "comparisons_all": 0,
        **{f"{m}_nn": None for m in HRF_MEASURES},
        "comparisons_nn": 0
    }

    if len(samples) < 2:
        return features

    features["mean_rr"] = float(np.mean(np.diff(samples) / fs))

    times = (samples - start) / fs

    nn_result = costa_hrf(times, costa_beat_codes(types), fs)
    all_result = costa_hrf(times, np.ones(len(times), dtype=int), fs)

    for measure in HRF_MEASURES:
        features[f"{measure}_nn"] = nn_result[measure]

    features["comparisons_nn"] = nn_result["nn_total"]
    features["pip_all"] = all_result["pip"]
    features["comparisons_all"] = all_result["nn_total"]

    return features


def segment_record(
    record_id,
    beat_samples,
    beat_types,
    spans,
    fs,
    signal_length,
    window_seconds,
    patient=None
):
    """
    Split one record into non-overlapping windows
    and return one feature row per window.

    Windows that cross a rhythm change are kept
    with rhythm None so they can be counted,
    but they are not labelled.
    """

    window = int(round(window_seconds * fs))
    rows = []

    # Convert once, not once per window.
    beat_samples = np.asarray(beat_samples, dtype=int)
    beat_types = np.asarray(beat_types)

    for start in range(0, signal_length - window + 1, window):
        end = start + window
        rhythm = window_rhythm(spans, start, end)

        label = None

        if rhythm is not None:
            label = 0 if rhythm == NORMAL_RHYTHM else 1

        rows.append({
            "record": record_id,
            "patient": patient or record_id,
            "start_s": start / fs,
            "end_s": end / fs,
            "rhythm": rhythm,
            "label": label,
            **window_features(
                beat_samples,
                beat_types,
                fs,
                start,
                end
            )
        })

    return rows


def parse_annotations(samples, symbols, notes):
    """
    Separate WFDB annotations into beats
    and rhythm changes.

    Rhythm labels start with '('. Other notes,
    such as signal-quality remarks, are ignored.
    """

    beat_samples = []
    beat_types = []
    rhythm_samples = []
    rhythm_labels = []

    for sample, symbol, note in zip(samples, symbols, notes):
        note = (note or "").strip("\x00 ")

        if symbol in BEAT_SYMBOLS:
            beat_samples.append(int(sample))
            beat_types.append(symbol)

        elif symbol == "+" and note.startswith("("):
            rhythm_samples.append(int(sample))
            rhythm_labels.append(note)

    return (
        np.asarray(beat_samples, dtype=int),
        beat_types,
        rhythm_samples,
        rhythm_labels
    )


def load_record_annotations(
    record_id,
    database="mitdb",
    common_fs=COMMON_FS
):
    """
    Read beat and rhythm annotations for one record,
    with sample positions converted to common_fs
    (or left at the native rate if it is None).

    Uses local files in data/raw/<database> when
    present, otherwise reads directly from PhysioNet.
    """

    data_dir = RAW_DIR / database

    if (data_dir / f"{record_id}.atr").exists():
        path = str(data_dir / record_id)
        header = wfdb.rdheader(path)
        annotations = wfdb.rdann(path, "atr")
    else:
        header = wfdb.rdheader(record_id, pn_dir=database)
        annotations = wfdb.rdann(record_id, "atr", pn_dir=database)

    beat_samples, beat_types, rhythm_samples, rhythm_labels = (
        parse_annotations(
            annotations.sample,
            annotations.symbol,
            annotations.aux_note
        )
    )

    fs = float(header.fs)

    if common_fs is None:
        common_fs = fs

    order = np.argsort(beat_samples, kind="stable")
    beat_samples = to_common_rate(beat_samples[order], fs, common_fs)
    beat_types = [beat_types[i] for i in order]
    rhythm_samples = to_common_rate(rhythm_samples, fs, common_fs)
    signal_length = int(to_common_rate(header.sig_len, fs, common_fs))

    return {
        "fs": float(common_fs),
        "original_fs": fs,
        "signal_length": signal_length,
        "beat_samples": beat_samples,
        "beat_types": beat_types,
        "spans": rhythm_spans(
            rhythm_samples,
            rhythm_labels,
            signal_length
        )
    }


def database_records(database):
    """
    Records to use from a database: its fixed list,
    or every record downloaded locally.
    """

    config = DATABASES[database]

    records = config["records"]

    if records is None:
        records = sorted(
            p.stem for p in (RAW_DIR / database).glob("*.atr")
        )

    return [r for r in records if r not in config["excluded"]]


def cap_windows(dataset, max_windows, random_state=RANDOM_STATE):
    """
    Keep at most max_windows labelled windows per
    patient and class, chosen at random.

    Unlabelled windows are kept for counting.
    """

    labelled = dataset["label"].notna()

    # Shuffle, then keep the first windows
    # of each patient and class.
    capped = (
        dataset[labelled]
        .sample(frac=1, random_state=random_state)
        .groupby(["patient", "label"])
        .head(max_windows)
    )

    return (
        pd.concat([capped, dataset[~labelled]])
        .sort_values(["patient", "record", "start_s"])
        .reset_index(drop=True)
    )


def build_dataset(
    databases=("mitdb",),
    window_seconds=30,
    max_minutes_per_class=MAX_MINUTES_PER_CLASS,
    common_fs=COMMON_FS
):
    """Build a window-level HRF feature table."""

    rows = []

    for database in databases:
        records = database_records(database)
        print(f"{database}: {len(records)} records")

        for record_id in records:
            ann = load_record_annotations(record_id, database, common_fs)

            for row in segment_record(
                record_id,
                ann["beat_samples"],
                ann["beat_types"],
                ann["spans"],
                ann["fs"],
                ann["signal_length"],
                window_seconds,
                patient=patient_id(database, record_id)
            ):
                rows.append({"database": database, **row})

    dataset = pd.DataFrame(rows)

    if max_minutes_per_class is not None:
        max_windows = int(max_minutes_per_class * 60 // window_seconds)
        dataset = cap_windows(dataset, max_windows)

    return dataset


def summarize(dataset):
    """Print window counts by database, label and rhythm."""

    labelled = dataset.dropna(subset=["label"])

    print(f"\nWindows: {len(dataset)}")
    print(f"Mixed-rhythm windows (unlabelled): "
          f"{len(dataset) - len(labelled)}")
    print(f"Labelled windows: {len(labelled)}")
    print()
    print(labelled.groupby(["database", "label"]).agg(
        windows=("record", "size"),
        patients=("patient", "nunique")
    ))
    print()
    print(labelled.groupby("rhythm").agg(
        windows=("record", "size"),
        patients=("patient", "nunique"),
        median_pip_nn=("pip_nn", "median"),
        median_pip_all=("pip_all", "median")
    ))


def main():
    parser = argparse.ArgumentParser(
        description="Build an HRF window dataset."
    )
    parser.add_argument("--databases", nargs="+", default=["mitdb"],
                        choices=sorted(DATABASES))
    parser.add_argument("--window", type=float, default=30,
                        help="Window length in seconds.")
    parser.add_argument("--native-rate", action="store_true",
                        help="Keep each database's own sampling "
                             f"rate instead of {COMMON_FS} Hz.")
    args = parser.parse_args()

    dataset = build_dataset(
        databases=args.databases,
        window_seconds=args.window,
        common_fs=None if args.native_rate else COMMON_FS
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    name = "+".join(args.databases)
    rate = "_native" if args.native_rate else ""
    output = OUTPUT_DIR / f"{name}_hrf_{args.window:g}s{rate}.csv"
    dataset.to_csv(output, index=False)

    summarize(dataset)
    print(f"\nSaved {output}")


if __name__ == "__main__":
    main()
