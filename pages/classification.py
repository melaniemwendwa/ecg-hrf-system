import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dataset import load_record_annotations
from src.hrf_analysis import NORMAL_BEATS
from src.results import (
    DATABASE_NAMES,
    describe_experiment,
    list_experiments,
    load_predictions,
    load_results,
    load_windows
)


NORMAL_COLOR = "#0F9F9A"
ABNORMAL_COLOR = "#E16C71"

LABELS = {0: "Normal", 1: "Abnormal"}


@st.cache_data(show_spinner=False)
def get_results(name):
    return load_results(name)


@st.cache_data(show_spinner=False)
def get_predictions(name):
    return load_predictions(name)


@st.cache_data(show_spinner=False)
def get_windows(name):
    return load_windows(name)


@st.cache_data(show_spinner=False)
def get_annotations(record_id, database):
    return load_record_annotations(record_id, database)


# ==========================================
# PAGE HEADER
# ==========================================

st.title("Rhythm Classification")

st.write(
    "See how the trained models classify each window "
    "of a recording as normal or abnormal rhythm, "
    "compared with the expert labels."
)

st.info(
    "Every prediction on this page comes from patient-grouped "
    "cross-validation: the model that classified a window "
    "was trained without any data from that patient."
)

experiments = list_experiments()

if not experiments:
    st.warning(
        "No experiment results found. Build a dataset and "
        "train the models first:\n\n"
        "`python -m src.dataset --databases mitdb ltafdb --window 120`\n\n"
        "`python -m src.train`"
    )
    st.stop()

st.divider()


# ==========================================
# SETTINGS
# ==========================================

st.subheader("Classification Settings")

with st.container(border=True):

    c1, c2, c3 = st.columns(3)

    with c1:
        experiment = st.selectbox(
            "Experiment",
            experiments,
            format_func=describe_experiment
        )

    results = get_results(experiment)
    predictions = get_predictions(experiment)

    feature_sets = list(dict.fromkeys(
        r["feature_set"] for r in results["results"]
    ))
    models = [
        m for m in dict.fromkeys(r["model"] for r in results["results"])
        if m != "Majority class"
    ]

    with c2:
        feature_set = st.selectbox(
            "Features",
            feature_sets,
            index=feature_sets.index("HRF + Mean RR")
            if "HRF + Mean RR" in feature_sets else 0
        )

    with c3:
        model = st.selectbox(
            "Model",
            models,
            index=models.index("SVM") if "SVM" in models else 0
        )

    c4, c5 = st.columns(2)

    databases = sorted(predictions["database"].unique())

    with c4:
        database = st.selectbox(
            "Database",
            databases,
            format_func=lambda d: DATABASE_NAMES.get(d, d)
        )

    in_database = predictions[predictions["database"] == database]

    record_counts = (
        in_database.groupby("record")["label"]
        .agg(normal=lambda s: int((s == 0).sum()),
             abnormal=lambda s: int((s == 1).sum()))
    )

    # Records with both classes first: the most
    # interesting ones to look at.
    both = (record_counts["normal"] > 0) & (record_counts["abnormal"] > 0)
    record_order = sorted(
        record_counts.index,
        key=lambda r: (not both[r], -record_counts.loc[r, "abnormal"], r)
    )

    with c5:
        record_id = st.selectbox(
            "Recording",
            record_order,
            format_func=lambda r: (
                f"{r}  ({record_counts.loc[r, 'normal']} normal, "
                f"{record_counts.loc[r, 'abnormal']} abnormal windows)"
            )
        )


# ==========================================
# RECORD PREDICTIONS
# ==========================================

key = f"{feature_set} | {model}"

record = (
    in_database[in_database["record"] == record_id]
    .sort_values("start_s")
    .reset_index(drop=True)
)

record["prediction"] = record[f"{key} | prediction"]
record["score"] = record[f"{key} | score"]
record["correct"] = record["prediction"] == record["label"]

# SVM scores are distances from the decision
# boundary; the other models give probabilities.
threshold = 0.0 if model == "SVM" else 0.5

st.divider()
st.subheader(f"Recording {record_id}")

st.caption(
    f"{DATABASE_NAMES.get(database, database)} | "
    f"{describe_experiment(experiment)} | "
    f"{feature_set} | {model}"
)

m1, m2, m3, m4 = st.columns(4)

m1.metric("Windows Classified", len(record))
m2.metric("Agreement with Experts", f"{100 * record['correct'].mean():.1f}%")
m3.metric("Expert: Abnormal", int((record["label"] == 1).sum()))
m4.metric("Model: Abnormal", int((record["prediction"] == 1).sum()))


# ==========================================
# TIMELINE
# ==========================================

fig = go.Figure()

for label, color in [(0, NORMAL_COLOR), (1, ABNORMAL_COLOR)]:
    for correct in [True, False]:
        rows = record[
            (record["label"] == label) & (record["correct"] == correct)
        ]

        if rows.empty:
            continue

        fig.add_trace(
            go.Scatter(
                x=rows["start_s"] / 3600,
                y=rows["score"],
                mode="markers",
                name=(
                    f"Expert: {LABELS[label]} "
                    f"({'correct' if correct else 'misclassified'})"
                ),
                marker=dict(
                    color=color,
                    size=11 if correct else 14,
                    symbol="circle" if correct else "x",
                    line=dict(width=1, color="white")
                ),
                hovertemplate=(
                    "Start: %{x:.2f} h<br>"
                    "Score: %{y:.2f}<extra></extra>"
                )
            )
        )

fig.add_hline(
    y=threshold,
    line_dash="dash",
    line_color="gray",
    annotation_text="Decision threshold",
    annotation_position="top left"
)

fig.update_layout(
    title="Model Score per Window",
    xaxis_title="Window Start (hours into recording)",
    yaxis_title="Score (higher = more likely abnormal)",
    height=450,
    template="plotly_white",
    legend=dict(orientation="h", y=-0.25)
)

st.plotly_chart(fig, width="stretch")

st.caption(
    "Colour shows the expert label; crosses mark windows the "
    "model got wrong. Windows above the threshold are "
    "classified as abnormal. Only windows with a single "
    "expert rhythm label are shown, and long recordings "
    "contribute at most 30 minutes per class."
)


# ==========================================
# WINDOW TABLE
# ==========================================

windows = get_windows(experiment)

details = record.merge(
    windows[
        (windows["database"] == database)
        & (windows["record"] == record_id)
    ].drop(columns=["patient", "label"]),
    on=["database", "record", "start_s"],
    how="left"
)

table = pd.DataFrame({
    "Start (min)": (details["start_s"] / 60).round(1),
    "Expert Rhythm": details["rhythm"],
    "Expert Label": details["label"].map(LABELS),
    "Model Label": details["prediction"].map(LABELS),
    "Correct": details["correct"].map({True: "Yes", False: "No"}),
    "PIP (%)": details["pip_nn"].round(1),
    "PAS (%)": details["pas_nn"].round(1),
    "Mean RR (ms)": (details["mean_rr"] * 1000).round(0)
})

st.subheader("Window Results")

st.dataframe(table, hide_index=True, width="stretch")


# ==========================================
# WINDOW DETAIL
# ==========================================

st.subheader("Inspect a Window")

choice = st.selectbox(
    "Window",
    details.index,
    format_func=lambda i: (
        f"{details.loc[i, 'start_s'] / 60:.1f} min | "
        f"expert {LABELS[details.loc[i, 'label']].lower()}, "
        f"model {LABELS[details.loc[i, 'prediction']].lower()}"
    )
)

window = details.loc[choice]

try:
    ann = get_annotations(record_id, database)
except Exception as error:
    st.error(f"Could not load annotations: {error}")
    st.stop()

fs = ann["fs"]
start = int(round(window["start_s"] * fs))
end = int(round(window["end_s"] * fs))

first, last = np.searchsorted(ann["beat_samples"], [start, end])
samples = ann["beat_samples"][first:last]
types = np.asarray(ann["beat_types"])[first:last]

rr_ms = np.diff(samples) / fs * 1000
normal_pair = (
    np.isin(types[:-1], NORMAL_BEATS) & np.isin(types[1:], NORMAL_BEATS)
)

fig_rr = go.Figure()

fig_rr.add_trace(
    go.Scatter(
        x=np.arange(1, len(rr_ms) + 1),
        y=rr_ms,
        mode="lines",
        line=dict(color="#2563EB", width=1.5),
        name="RR Intervals",
        hoverinfo="skip"
    )
)

for is_normal, name, color in [
    (True, "Between normal beats", "#2563EB"),
    (False, "Involving a non-normal beat", "#F59E0B")
]:
    idx = np.flatnonzero(normal_pair == is_normal)

    if len(idx) == 0:
        continue

    fig_rr.add_trace(
        go.Scatter(
            x=idx + 1,
            y=rr_ms[idx],
            mode="markers",
            name=name,
            marker=dict(color=color, size=6 if is_normal else 9)
        )
    )

fig_rr.update_layout(
    title=(
        f"RR Intervals, {window['start_s'] / 60:.1f}–"
        f"{window['end_s'] / 60:.1f} min ({window['rhythm']})"
    ),
    xaxis_title="Interval Number",
    yaxis_title="RR Interval (ms)",
    height=400,
    template="plotly_white",
    legend=dict(orientation="h", y=-0.25)
)

st.plotly_chart(fig_rr, width="stretch")


def show(value, fmt):
    return fmt.format(value) if pd.notna(value) else "N/A"


h1, h2, h3, h4, h5, h6 = st.columns(6)

h1.metric("PIP", show(window["pip_nn"], "{:.1f}%"))
h2.metric("PNNSS", show(window["pnnss_nn"], "{:.1f}%"))
h3.metric("PNNLS", show(window["pnnls_nn"], "{:.1f}%"))
h4.metric("IALS", show(window["ials_nn"], "{:.3f}"))
h5.metric("PAS", show(window["pas_nn"], "{:.1f}%"))
h6.metric("Mean RR", show(window["mean_rr"] * 1000, "{:.0f} ms"))

st.caption(
    "HRF measures use intervals between normal beats only. "
    "During atrial fibrillation intervals vary almost at "
    "random: PIP sits near 67% and PAS rises."
)

st.divider()

st.caption(
    "ECG Insight | Classification | "
    "Academic research only. Not for clinical diagnosis."
)
