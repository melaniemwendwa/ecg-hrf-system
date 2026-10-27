import numpy as np
import pandas as pd
import pytest

from src.dataset import (
    cap_windows,
    patient_id,
    to_common_rate,
    rhythm_spans,
    window_rhythm,
    window_features,
    segment_record,
    parse_annotations
)


def test_rhythm_spans():
    spans = rhythm_spans([0, 500], ["(N", "(AFIB"], 1000)

    assert spans == [(0, 500, "(N"), (500, 1000, "(AFIB")]


def test_rhythm_spans_unsorted():
    spans = rhythm_spans([500, 0], ["(AFIB", "(N"], 1000)

    assert spans[0] == (0, 500, "(N")


def test_window_rhythm():
    spans = [(0, 500, "(N"), (500, 1000, "(AFIB")]

    assert window_rhythm(spans, 0, 500) == "(N"
    assert window_rhythm(spans, 600, 900) == "(AFIB"

    # Window crosses a rhythm change
    assert window_rhythm(spans, 400, 600) is None


def test_window_rhythm_before_first_annotation():
    spans = [(100, 1000, "(N")]

    assert window_rhythm(spans, 0, 200) is None


def test_window_features():
    fs = 250
    rr = [0.80, 0.82] * 6
    samples = np.round(
        np.concatenate([[0], np.cumsum(rr)]) * fs
    ).astype(int) + 500
    types = ["N"] * len(samples)
    types[6] = "V"

    features = window_features(samples, types, fs, 500, 10_000)

    assert features["n_beats"] == 13
    assert features["frac_non_normal_beats"] == pytest.approx(1 / 13)
    assert features["mean_rr"] == pytest.approx(0.81)

    # Every beat counted as normal: one unbroken zig-zag.
    assert features["pip_all"] == pytest.approx(100.0)
    assert features["comparisons_all"] == 10

    # The V beat excludes the segments around it.
    assert features["comparisons_nn"] < features["comparisons_all"]
    assert features["pas_nn"] is not None


def test_window_features_too_few_beats():
    features = window_features([0, 80], ["N", "N"], 100, 0, 1000)

    assert features["pip_all"] is None
    assert features["mean_rr"] == pytest.approx(0.8)


def test_segment_record_labels():
    fs = 10
    spans = [(0, 100, "(N"), (100, 250, "(AFIB")]
    samples = list(range(0, 250, 8))
    types = ["N"] * len(samples)

    rows = segment_record("999", samples, types, spans, fs, 250, 5)

    # 250 samples / 50-sample windows = 5 windows
    assert [r["label"] for r in rows] == [0, 0, 1, 1, 1]
    assert rows[2]["rhythm"] == "(AFIB"


def test_segment_record_mixed_window_unlabelled():
    fs = 10
    spans = [(0, 70, "(N"), (70, 100, "(AFIB")]

    rows = segment_record("999", [], [], spans, fs, 100, 5)

    assert rows[0]["label"] == 0
    assert rows[1]["label"] is None


def test_patient_id():
    # Records 201 and 202 share a patient.
    assert patient_id("mitdb", "202") == "mitdb:201"

    # Same record name in different databases
    # must be different patients.
    assert patient_id("mitdb", "100") != patient_id("ltafdb", "100")


def test_to_common_rate():
    # 360 Hz samples to 128 Hz
    assert to_common_rate([0, 360, 720], 360, 128).tolist() == [
        0, 128, 256
    ]


def test_cap_windows():
    dataset = pd.DataFrame({
        "patient": ["a"] * 5 + ["b"] * 2 + ["a"],
        "record": ["1"] * 8,
        "start_s": range(8),
        "label": [0, 0, 0, 0, 1, 0, 0, None]
    })

    capped = cap_windows(dataset, max_windows=2)

    counts = capped.groupby(
        ["patient", "label"], dropna=False
    ).size().to_dict()

    assert counts[("a", 0)] == 2
    assert counts[("a", 1)] == 1
    assert counts[("b", 0)] == 2

    # Unlabelled windows are kept.
    assert capped["label"].isna().sum() == 1


def test_parse_annotations_ignores_non_rhythm_notes():
    samples = [0, 10, 50, 60, 90, 120]
    symbols = ["+", "N", "+", '"', "L", "+"]
    notes = ["(N\x00", "", "MISSB", "PSE", "", "(AFIB"]

    beats, types, rhythm_samples, rhythm_labels = (
        parse_annotations(samples, symbols, notes)
    )

    assert beats.tolist() == [10, 90]
    assert types == ["N", "L"]
    assert rhythm_samples == [0, 120]
    assert rhythm_labels == ["(N", "(AFIB"]
