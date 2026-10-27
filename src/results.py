import json
import re
from pathlib import Path

import pandas as pd

from src.dataset import OUTPUT_DIR
from src.train import MODELS_DIR, add_derived_features


# Experiment shown by default in the app
MAIN_EXPERIMENT = "mitdb+ltafdb_hrf_120s"

DATABASE_NAMES = {
    "mitdb": "MIT-BIH Arrhythmia",
    "ltafdb": "Long-Term AF"
}


def list_experiments():
    """Names of experiments with saved results, main first."""

    names = sorted(
        p.name[:-len("_results.json")]
        for p in MODELS_DIR.glob("*_results.json")
    )

    # Main experiment, then combined databases,
    # each ordered by window length.
    return sorted(names, key=lambda n: (
        n != MAIN_EXPERIMENT,
        "+" not in n,
        n.split("_hrf_")[0],
        window_seconds(n) or 0,
        n
    ))


def describe_experiment(name):
    """Readable label, e.g. 'Both databases, 2 min windows'."""

    match = re.fullmatch(r"(.+)_hrf_(\d+(?:\.\d+)?)s(_native)?", name)

    if not match:
        return name

    databases, seconds, native = match.groups()
    seconds = float(seconds)

    source = (
        "Both databases" if "+" in databases
        else DATABASE_NAMES.get(databases, databases)
    )
    window = (
        f"{seconds / 60:g} min" if seconds >= 60
        else f"{seconds:g} s"
    )
    rate = ", native sampling rate" if native else ""

    return f"{source}, {window} windows{rate}"


def window_seconds(name):
    """Window length of an experiment, in seconds."""

    match = re.search(r"_hrf_(\d+(?:\.\d+)?)s", name)

    return float(match.group(1)) if match else None


def load_results(name):
    return json.loads((MODELS_DIR / f"{name}_results.json").read_text())


def load_predictions(name):
    """Out-of-fold predictions, one row per window."""

    return pd.read_csv(
        MODELS_DIR / f"{name}_predictions.csv",
        dtype={"record": str, "patient": str}
    )


def load_windows(name):
    """Window features from the dataset the experiment used."""

    dataset = pd.read_csv(
        OUTPUT_DIR / f"{name}.csv",
        dtype={"record": str, "patient": str}
    )

    return add_derived_features(dataset)
