import numpy as np
import pandas as pd

from src.train import (
    FEATURE_SETS,
    confidence_intervals,
    cross_validate,
    patient_bootstrap,
    make_splits,
    select_rows
)


def synthetic_dataset(n_patients=20, windows=10, seed=0):
    """Abnormal patients have higher PIP."""

    rng = np.random.default_rng(seed)
    rows = []

    for p in range(n_patients):
        label = p % 4 == 0

        for _ in range(windows):
            rows.append({
                "patient": str(p),
                "label": float(label),
                "pip_nn": rng.normal(75 if label else 55, 5),
                "pip_all": rng.normal(75 if label else 55, 5),
                "ials_nn": rng.normal(0.5, 0.1),
                "pnnss_nn": rng.normal(50, 10),
                "pnnls_nn": rng.normal(30, 10),
                "pas_nn": rng.normal(10, 5),
                "mean_rr": rng.normal(0.8, 0.1),
                "comparisons_nn": 30
            })

    return pd.DataFrame(rows)


def test_select_rows_drops_unusable_windows():
    dataset = synthetic_dataset(n_patients=4, windows=1)

    dataset.loc[0, "label"] = None
    dataset.loc[1, "pip_nn"] = None
    dataset.loc[2, "comparisons_nn"] = 3

    selected = select_rows(dataset)

    assert len(selected) == 1
    assert selected["label"].dtype == int


def test_splits_never_share_patients():
    dataset = synthetic_dataset()

    splits = make_splits(
        dataset["label"].to_numpy(),
        dataset["patient"].to_numpy()
    )

    for train, test in splits:
        train_patients = set(dataset["patient"].iloc[train])
        test_patients = set(dataset["patient"].iloc[test])

        assert train_patients.isdisjoint(test_patients)


def test_cross_validate_learns_separable_data():
    selected = select_rows(synthetic_dataset())

    results, predictions = cross_validate(selected)

    assert len(results) == len(FEATURE_SETS) * 4
    assert "pip_distance" in selected.columns

    pip_lr = next(
        r for r in results
        if r["feature_set"] == "PIP (NN)"
        and r["model"] == "Logistic Regression"
    )
    assert pip_lr["balanced_accuracy"] > 0.9

    # Every window is predicted exactly once.
    cm = pip_lr["confusion_matrix"]
    assert sum(cm.values()) == len(selected)


def test_patient_bootstrap_keeps_patients_together():
    patients = np.array(["a", "a", "b", "c", "c", "c"])

    for idx in patient_bootstrap(patients, n_bootstrap=20):
        drawn = patients[idx]

        # Each drawn patient brings all their windows.
        for p in set(drawn):
            assert (drawn == p).sum() % (patients == p).sum() == 0


def test_confidence_intervals_contain_estimate():
    selected = select_rows(synthetic_dataset(n_patients=40))

    results, predictions = cross_validate(selected)
    intervals = confidence_intervals(predictions, n_bootstrap=200)

    for r in results:
        low, high = intervals["roc_auc"][f"{r['feature_set']} | {r['model']}"]
        assert low <= r["roc_auc"] + 1e-9
        assert r["roc_auc"] - 1e-9 <= high

    # Mean RR is noise in the synthetic data, so
    # adding PIP to it should clearly help.
    low, _ = intervals["roc_auc_difference"][
        "PIP (NN) + Mean RR vs Mean RR (baseline) | Logistic Regression"
    ]
    assert low > 0
