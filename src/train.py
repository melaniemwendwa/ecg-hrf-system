import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


MODELS_DIR = Path("models")

RANDOM_STATE = 42
N_SPLITS = 5

# Bootstrap resamples for confidence intervals
N_BOOTSTRAP = 1000

# A window needs enough NN comparisons for
# its PIP value to be meaningful.
MIN_COMPARISONS = 10

# PIP of a random interval sequence: two of
# every three points are peaks or troughs.
# AF windows sit close to this value.
RANDOM_PIP = 200 / 3

# PIP, PNNSS and PNNLS follow Costa's reference
# software; IALS and PAS follow Costa et al. 2017.
HRF_FEATURES = ["pip_nn", "pnnss_nn", "pnnls_nn", "ials_nn", "pas_nn"]

# Feature sets compared in the experiment.
# Mean RR is a baseline: it shows how much
# HRF adds beyond plain heart rate.
FEATURE_SETS = {
    "PIP (NN)": ["pip_nn"],
    "PIP distance from random": ["pip_distance"],
    "HRF (all 5 measures)": HRF_FEATURES,
    "Mean RR (baseline)": ["mean_rr"],
    "PIP (NN) + Mean RR": ["pip_nn", "mean_rr"],
    "HRF + Mean RR": [*HRF_FEATURES, "mean_rr"]
}

# Pairs of feature sets whose difference in
# ROC AUC is tested: (richer set, simpler set).
COMPARISONS = [
    ("PIP (NN) + Mean RR", "Mean RR (baseline)"),
    ("HRF + Mean RR", "Mean RR (baseline)"),
    ("HRF + Mean RR", "PIP (NN) + Mean RR"),
    ("PIP distance from random", "PIP (NN)")
]


def add_derived_features(dataset):
    """Add features calculated from dataset columns."""

    dataset = dataset.copy()
    dataset["pip_distance"] = (dataset["pip_nn"] - RANDOM_PIP).abs()

    return dataset


def make_models():
    """
    Return the classifiers to compare.

    Classes are imbalanced, so models weight
    abnormal windows more heavily. The majority
    classifier is the score to beat.
    """

    return {
        "Majority class": DummyClassifier(
            strategy="most_frequent"
        ),
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced")
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "SVM": make_pipeline(
            StandardScaler(),
            SVC(class_weight="balanced")
        )
    }


def select_rows(dataset, min_comparisons=MIN_COMPARISONS):
    """
    Keep labelled windows where every feature
    is available, so all feature sets are
    compared on the same windows.
    """

    dataset = add_derived_features(dataset)

    features = sorted({
        f for fs in FEATURE_SETS.values() for f in fs
    })

    keep = (
        dataset["label"].notna()
        & dataset[features].notna().all(axis=1)
        & (dataset["comparisons_nn"] >= min_comparisons)
    )

    selected = dataset[keep].copy()
    selected["label"] = selected["label"].astype(int)
    selected["patient"] = selected["patient"].astype(str)

    return selected


def make_splits(labels, groups, n_splits=N_SPLITS):
    """
    Split windows into folds by patient, so no
    patient appears in both training and testing.
    """

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    return list(splitter.split(
        np.zeros(len(labels)),
        labels,
        groups
    ))


def abnormal_scores(model, X):
    """
    Scores where higher means more likely
    abnormal. ROC AUC only needs the ranking,
    so SVM decision values work as well as
    probabilities.
    """

    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    return model.decision_function(X)


def score(labels, predictions, scores):
    """Classification metrics for the abnormal class."""

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1]
    ).ravel()

    return {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            labels,
            predictions
        ),
        "precision": precision_score(
            labels,
            predictions,
            zero_division=0
        ),
        "recall": recall_score(
            labels,
            predictions,
            zero_division=0
        ),
        "f1": f1_score(labels, predictions, zero_division=0),
        "roc_auc": roc_auc_score(labels, scores),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }
    }


def cross_validate(selected, n_splits=N_SPLITS):
    """
    Evaluate every model on every feature set
    using patient-grouped cross-validation.

    Each window is predicted once, by a model
    that never saw that window's patient.

    Returns the metrics and a table of these
    out-of-fold predictions.
    """

    labels = selected["label"].to_numpy()
    splits = make_splits(
        labels,
        selected["patient"].to_numpy(),
        n_splits
    )

    results = []

    id_columns = [
        c for c in ["database", "record", "patient", "start_s", "label"]
        if c in selected.columns
    ]
    predictions_table = selected[id_columns].reset_index(drop=True)

    for set_name, features in FEATURE_SETS.items():
        X = selected[features].to_numpy()

        for model_name, model in make_models().items():
            predictions = np.zeros(len(labels), dtype=int)
            scores = np.zeros(len(labels))

            for train, test in splits:
                model.fit(X[train], labels[train])
                predictions[test] = model.predict(X[test])
                scores[test] = abnormal_scores(model, X[test])

            result = {
                "feature_set": set_name,
                "features": features,
                "model": model_name,
                **score(labels, predictions, scores)
            }

            if "database" in selected.columns:
                result["by_database"] = score_by_group(
                    labels,
                    predictions,
                    scores,
                    selected["database"].to_numpy()
                )

            results.append(result)

            key = f"{set_name} | {model_name}"
            predictions_table[f"{key} | prediction"] = predictions
            predictions_table[f"{key} | score"] = scores

    return results, predictions_table


def score_by_group(labels, predictions, scores, groups):
    """
    Metrics separately for each group, such as
    each database. Groups with one class only
    cannot be scored and are skipped.
    """

    scored = {}

    for group in np.unique(groups):
        mask = groups == group

        if len(np.unique(labels[mask])) == 2:
            scored[str(group)] = score(
                labels[mask],
                predictions[mask],
                scores[mask]
            )

    return scored


def patient_bootstrap(patients, n_bootstrap=N_BOOTSTRAP):
    """
    Yield window indices for bootstrap samples
    drawn by patient, not by window, because
    windows from one patient are not independent.
    """

    rng = np.random.default_rng(RANDOM_STATE)

    unique = np.unique(patients)
    indices = [np.flatnonzero(patients == p) for p in unique]

    for _ in range(n_bootstrap):
        chosen = rng.integers(0, len(unique), len(unique))
        yield np.concatenate([indices[i] for i in chosen])


def confidence_intervals(
    predictions_table,
    n_bootstrap=N_BOOTSTRAP
):
    """
    95% bootstrap intervals for each model's
    ROC AUC, and for the AUC gained by adding
    PIP (see COMPARISONS).

    An improvement is supported when its
    interval lies entirely above zero.
    """

    labels = predictions_table["label"].to_numpy()
    patients = predictions_table["patient"].to_numpy()

    score_columns = {
        c[:-len(" | score")]: predictions_table[c].to_numpy()
        for c in predictions_table.columns
        if c.endswith(" | score")
    }

    # The majority classifier ignores features,
    # so comparing its feature sets is pointless.
    models = sorted(
        {key.split(" | ")[1] for key in score_columns}
        - {"Majority class"}
    )

    differences = {
        f"{with_pip} vs {without} | {model}": (
            f"{with_pip} | {model}",
            f"{without} | {model}"
        )
        for with_pip, without in COMPARISONS
        for model in models
    }

    samples = {key: [] for key in [*score_columns, *differences]}

    for idx in patient_bootstrap(patients, n_bootstrap):
        if len(np.unique(labels[idx])) < 2:
            continue

        auc = {
            key: roc_auc_score(labels[idx], scores[idx])
            for key, scores in score_columns.items()
        }

        for key in score_columns:
            samples[key].append(auc[key])

        for key, (a, b) in differences.items():
            samples[key].append(auc[a] - auc[b])

    return {
        "roc_auc": {
            key: list(np.percentile(samples[key], [2.5, 97.5]))
            for key in score_columns
        },
        "roc_auc_difference": {
            key: list(np.percentile(samples[key], [2.5, 97.5]))
            for key in differences
        }
    }


def cross_database(selected):
    """
    Train on one database and test on another.

    This checks that models learn rhythm, not
    differences between the databases.
    """

    if "database" not in selected.columns:
        return []

    databases = sorted(selected["database"].unique())
    results = []

    for train_db in databases:
        for test_db in databases:
            if train_db == test_db:
                continue

            train = selected[selected["database"] == train_db]
            test = selected[selected["database"] == test_db]

            if train["label"].nunique() < 2 or test["label"].nunique() < 2:
                continue

            for set_name, features in FEATURE_SETS.items():
                for model_name, model in make_models().items():
                    model.fit(train[features].to_numpy(), train["label"])

                    X = test[features].to_numpy()

                    results.append({
                        "train": train_db,
                        "test": test_db,
                        "feature_set": set_name,
                        "model": model_name,
                        **score(
                            test["label"].to_numpy(),
                            model.predict(X),
                            abnormal_scores(model, X)
                        )
                    })

    return results


def train_final_models(selected):
    """Fit every model on all selected windows."""

    labels = selected["label"].to_numpy()
    fitted = {}

    for set_name, features in FEATURE_SETS.items():
        X = selected[features].to_numpy()
        fitted[set_name] = {}

        for model_name, model in make_models().items():
            fitted[set_name][model_name] = model.fit(X, labels)

    return fitted


def run_experiment(data_path, n_splits=N_SPLITS):
    """
    Cross-validate, fit final models, and save
    both to MODELS_DIR, named after the dataset.
    """

    data_path = Path(data_path)
    dataset = pd.read_csv(data_path, dtype={"record": str, "patient": str})
    selected = select_rows(dataset)

    labelled = dataset[dataset["label"].notna()]

    summary = {
        "dataset": data_path.name,
        "n_splits": n_splits,
        "min_comparisons": MIN_COMPARISONS,
        "labelled_windows": int(len(labelled)),
        "used_windows": int(len(selected)),
        "used_by_class": {
            str(k): int(v)
            for k, v in selected["label"].value_counts().items()
        },
        "excluded_by_class": {
            str(int(k)): int(v)
            for k, v in (
                labelled["label"].value_counts()
                - selected["label"].value_counts()
            ).fillna(0).items()
        },
        "patients": int(selected["patient"].nunique()),
        "patients_with_abnormal": int(
            selected.loc[selected["label"] == 1, "patient"].nunique()
        )
    }

    results, predictions_table = cross_validate(selected, n_splits)

    summary["results"] = results
    summary["confidence_intervals"] = confidence_intervals(
        predictions_table
    )
    summary["cross_database"] = cross_database(selected)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    results_path = MODELS_DIR / f"{data_path.stem}_results.json"
    results_path.write_text(json.dumps(summary, indent=2))

    predictions_table.to_csv(
        MODELS_DIR / f"{data_path.stem}_predictions.csv",
        index=False
    )

    joblib.dump(
        {
            "feature_sets": FEATURE_SETS,
            "models": train_final_models(selected)
        },
        MODELS_DIR / f"{data_path.stem}_models.joblib"
    )

    return summary, results_path


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate PIP classifiers."
    )
    parser.add_argument(
        "--data",
        default="data/processed/mitdb+ltafdb_hrf_120s.csv"
    )
    args = parser.parse_args()

    summary, results_path = run_experiment(args.data)

    print(f"Windows used: {summary['used_windows']} "
          f"of {summary['labelled_windows']} labelled")
    print(f"By class: {summary['used_by_class']}")
    print(f"Excluded by class: {summary['excluded_by_class']}")
    print(f"Patients: {summary['patients']} "
          f"({summary['patients_with_abnormal']} with abnormal windows)")
    print()

    intervals = summary["confidence_intervals"]

    table = pd.DataFrame(summary["results"])

    table["roc_auc_95ci"] = [
        "{:.3f}-{:.3f}".format(
            *intervals["roc_auc"][f"{r.feature_set} | {r.model}"]
        )
        for r in table.itertuples()
    ]

    columns = [
        "feature_set", "model", "balanced_accuracy",
        "precision", "recall", "f1", "roc_auc", "roc_auc_95ci"
    ]

    with pd.option_context("display.width", 160):
        print(table[columns].round(3).to_string(index=False))

        print("\nROC AUC gained by adding PIP (95% CI):")

        for key, (low, high) in intervals["roc_auc_difference"].items():
            verdict = "supported" if low > 0 else "not supported"
            print(f"  {key}: {low:+.3f} to {high:+.3f} ({verdict})")

        if "by_database" in table.columns:
            print("\nROC AUC per database:")
            per_db = pd.DataFrame([
                {
                    "feature_set": r["feature_set"],
                    "model": r["model"],
                    **{db: m["roc_auc"] for db, m in r["by_database"].items()}
                }
                for r in summary["results"]
            ])
            print(per_db.round(3).to_string(index=False))

        if summary["cross_database"]:
            print("\nCross-database ROC AUC (train -> test):")
            cross = pd.DataFrame(summary["cross_database"])
            cross["direction"] = cross["train"] + " -> " + cross["test"]
            print(
                cross.pivot_table(
                    index=["feature_set", "model"],
                    columns="direction",
                    values="roc_auc",
                    sort=False
                ).round(3).to_string()
            )

    print(f"\nSaved {results_path}")


if __name__ == "__main__":
    main()
