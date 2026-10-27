import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve

from src.results import (
    DATABASE_NAMES,
    describe_experiment,
    list_experiments,
    load_predictions,
    load_results,
    load_windows,
    window_seconds
)
from src.train import RANDOM_PIP, select_rows


NORMAL_COLOR = "#0F9F9A"
ABNORMAL_COLOR = "#E16C71"

MEASURE_NAMES = {
    "pip_nn": "PIP (%)",
    "pas_nn": "PAS (%)",
    "pnnss_nn": "PNNSS (%)",
    "pnnls_nn": "PNNLS (%)",
    "ials_nn": "IALS",
    "mean_rr": "Mean RR (s)"
}

MODEL_COLORS = {
    "Logistic Regression": "#8B5CF6",
    "Random Forest": "#0F9F9A",
    "SVM": "#2563EB"
}


@st.cache_data(show_spinner=False)
def get_results(name):
    return load_results(name)


@st.cache_data(show_spinner=False)
def get_predictions(name):
    return load_predictions(name)


@st.cache_data(show_spinner=False)
def get_used_windows(name):
    return select_rows(load_windows(name))


# ==========================================
# PAGE HEADER
# ==========================================

st.title("Model Performance")

st.write(
    "Compare how well each model separates normal from "
    "abnormal rhythm windows, using Heart Rate Fragmentation "
    "(HRF) measures, heart rate, or both."
)

st.info(
    "All scores come from 5-fold cross-validation grouped by "
    "patient: no patient appears in both training and testing. "
    "ROC AUC is 0.5 for guessing and 1.0 for perfect separation."
)

experiments = list_experiments()

if not experiments:
    st.warning(
        "No experiment results found. Train the models first: "
        "`python -m src.train`"
    )
    st.stop()

experiment = st.selectbox(
    "Experiment",
    experiments,
    format_func=describe_experiment
)

results = get_results(experiment)
predictions = get_predictions(experiment)

scores = pd.DataFrame(results["results"])
scores = scores[scores["model"] != "Majority class"]

intervals = results["confidence_intervals"]

st.divider()


# ==========================================
# DATA SUMMARY
# ==========================================

st.subheader("Data Used")

d1, d2, d3, d4 = st.columns(4)

d1.metric("Windows", results["used_windows"])
d2.metric("Patients", results["patients"])
d3.metric("Patients with Abnormal Rhythm", results["patients_with_abnormal"])
d4.metric(
    "Normal / Abnormal Windows",
    f"{results['used_by_class'].get('0', 0)} / "
    f"{results['used_by_class'].get('1', 0)}"
)


# ==========================================
# AUC COMPARISON
# ==========================================

st.subheader("ROC AUC by Features and Model")

feature_sets = list(dict.fromkeys(scores["feature_set"]))
models = list(dict.fromkeys(scores["model"]))

fig = go.Figure()

for model in models:
    rows = scores[scores["model"] == model].set_index("feature_set")
    rows = rows.loc[feature_sets]

    low, high = np.array([
        intervals["roc_auc"][f"{fs} | {model}"] for fs in feature_sets
    ]).T

    fig.add_trace(
        go.Bar(
            name=model,
            x=feature_sets,
            y=rows["roc_auc"],
            marker_color=MODEL_COLORS.get(model),
            error_y=dict(
                type="data",
                symmetric=False,
                array=high - rows["roc_auc"],
                arrayminus=rows["roc_auc"] - low
            ),
            hovertemplate="%{x}<br>AUC %{y:.3f}<extra>" + model + "</extra>"
        )
    )

fig.add_hline(
    y=0.5,
    line_dash="dash",
    line_color="gray",
    annotation_text="Guessing",
    annotation_position="bottom right"
)

fig.update_layout(
    barmode="group",
    yaxis=dict(title="ROC AUC", range=[0.4, 1.0]),
    height=450,
    template="plotly_white",
    legend=dict(orientation="h", y=-0.25)
)

st.plotly_chart(fig, width="stretch")

st.caption("Error bars are 95% confidence intervals from a patient-level bootstrap.")


# ==========================================
# RESULTS TABLE
# ==========================================

table = pd.DataFrame({
    "Features": scores["feature_set"],
    "Model": scores["model"],
    "Balanced Accuracy": scores["balanced_accuracy"],
    "Precision": scores["precision"],
    "Recall": scores["recall"],
    "F1": scores["f1"],
    "ROC AUC": scores["roc_auc"],
    "AUC 95% CI": [
        "{:.3f} – {:.3f}".format(
            *intervals["roc_auc"][f"{r.feature_set} | {r.model}"]
        )
        for r in scores.itertuples()
    ]
})

st.dataframe(
    table.style.format(precision=3).highlight_max(
        subset=["Balanced Accuracy", "F1", "ROC AUC"],
        color="#D1FAE5"
    ),
    hide_index=True,
    width="stretch"
)

with st.expander("Does adding features really help?"):

    rows = []

    for key, (low, high) in intervals["roc_auc_difference"].items():
        comparison, model = key.split(" | ")
        richer, simpler = comparison.split(" vs ")

        rows.append({
            "Model": model,
            "Compared": f"{richer}  vs  {simpler}",
            "AUC Gain, 95% CI": f"{low:+.3f} to {high:+.3f}",
            "Verdict": (
                "Improvement" if low > 0
                else "Worse" if high < 0
                else "No clear difference"
            )
        })

    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.caption(
        "A difference is supported when its whole interval "
        "lies above zero."
    )


# ==========================================
# CONFUSION MATRIX AND ROC
# ==========================================

st.divider()
st.subheader("Detailed View")

c1, c2 = st.columns(2)

with c1:
    feature_set = st.selectbox(
        "Features",
        feature_sets,
        index=feature_sets.index("HRF + Mean RR")
        if "HRF + Mean RR" in feature_sets else 0
    )

with c2:
    model = st.selectbox(
        "Model",
        models,
        index=models.index("SVM") if "SVM" in models else 0
    )

selected = scores[
    (scores["feature_set"] == feature_set) & (scores["model"] == model)
].iloc[0]

cm = selected["confusion_matrix"]

left, right = st.columns(2)

with left:

    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])

    fig_cm = go.Figure(
        go.Heatmap(
            z=matrix,
            x=["Predicted Normal", "Predicted Abnormal"],
            y=["Expert Normal", "Expert Abnormal"],
            text=matrix,
            texttemplate="%{text}",
            textfont=dict(size=18),
            colorscale="Blues",
            showscale=False
        )
    )

    fig_cm.update_layout(
        title="Confusion Matrix",
        yaxis=dict(autorange="reversed"),
        height=400,
        template="plotly_white"
    )

    st.plotly_chart(fig_cm, width="stretch")

with right:

    fig_roc = go.Figure()

    for fs in feature_sets:
        fpr, tpr, _ = roc_curve(
            predictions["label"],
            predictions[f"{fs} | {model} | score"]
        )
        auc = scores[
            (scores["feature_set"] == fs) & (scores["model"] == model)
        ]["roc_auc"].iloc[0]

        fig_roc.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{fs} ({auc:.2f})",
                line=dict(width=3 if fs == feature_set else 1.5)
            )
        )

    fig_roc.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="gray"),
            showlegend=False
        )
    )

    fig_roc.update_layout(
        title=f"ROC Curves, {model}",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=400,
        template="plotly_white",
        legend=dict(font=dict(size=10))
    )

    st.plotly_chart(fig_roc, width="stretch")

if "by_database" in selected and isinstance(selected["by_database"], dict):

    per_db = pd.DataFrame([
        {
            "Database": DATABASE_NAMES.get(db, db),
            "Balanced Accuracy": m["balanced_accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1": m["f1"],
            "ROC AUC": m["roc_auc"]
        }
        for db, m in selected["by_database"].items()
    ])

    st.markdown("**Results within each database**")
    st.dataframe(
        per_db.style.format(precision=3),
        hide_index=True,
        width="stretch"
    )


# ==========================================
# CROSS-DATABASE
# ==========================================

if results.get("cross_database"):

    st.divider()
    st.subheader("Testing on an Unseen Database")

    st.write(
        "Models trained on one database and tested on the other. "
        "Good scores here show the models learned rhythm, "
        "not differences between the databases."
    )

    cross = pd.DataFrame(results["cross_database"])
    cross = cross[cross["model"] != "Majority class"]
    cross["Direction"] = (
        cross["train"].map(DATABASE_NAMES) + "  →  "
        + cross["test"].map(DATABASE_NAMES)
    )

    pivot = cross.pivot_table(
        index=["feature_set", "model"],
        columns="Direction",
        values="roc_auc",
        sort=False
    ).reset_index().rename(
        columns={"feature_set": "Features", "model": "Model"}
    )

    st.dataframe(
        pivot.style.format(precision=3, subset=pivot.columns[2:]),
        hide_index=True,
        width="stretch"
    )


# ==========================================
# WINDOW LENGTH
# ==========================================

combined = [
    name for name in experiments
    if "+" in name and window_seconds(name)
]

if len(combined) > 1:

    st.divider()
    st.subheader("Effect of Window Length")

    rows = []

    for name in combined:
        for r in get_results(name)["results"]:
            if r["model"] == model:
                rows.append({
                    "minutes": window_seconds(name) / 60,
                    "feature_set": r["feature_set"],
                    "roc_auc": r["roc_auc"]
                })

    lengths = pd.DataFrame(rows).sort_values("minutes")

    fig_len = go.Figure()

    for fs in feature_sets:
        rows = lengths[lengths["feature_set"] == fs]

        fig_len.add_trace(
            go.Scatter(
                x=rows["minutes"],
                y=rows["roc_auc"],
                mode="lines+markers",
                name=fs
            )
        )

    fig_len.update_layout(
        title=f"ROC AUC by Window Length, {model}",
        xaxis=dict(
            title="Window Length (minutes)",
            type="log",
            tickvals=sorted(lengths["minutes"].unique()),
            ticktext=[f"{m:g}" for m in sorted(lengths["minutes"].unique())]
        ),
        yaxis_title="ROC AUC",
        height=420,
        template="plotly_white",
        legend=dict(orientation="h", y=-0.3)
    )

    st.plotly_chart(fig_len, width="stretch")

    st.caption(
        "Longer windows contain more heartbeats, so HRF "
        "measures are estimated more reliably."
    )


# ==========================================
# HRF BY RHYTHM
# ==========================================

st.divider()
st.subheader("HRF Measures by Rhythm")

windows = get_used_windows(experiment)
windows["Class"] = windows["label"].map({0: "Normal", 1: "Abnormal"})
windows["Database"] = windows["database"].map(DATABASE_NAMES)

measure = st.radio(
    "Measure",
    list(MEASURE_NAMES),
    format_func=lambda m: MEASURE_NAMES[m].split(" (")[0],
    horizontal=True
)

fig_box = go.Figure()

for label, color in [("Normal", NORMAL_COLOR), ("Abnormal", ABNORMAL_COLOR)]:
    rows = windows[windows["Class"] == label]

    fig_box.add_trace(
        go.Box(
            x=rows["Database"],
            y=rows[measure],
            name=label,
            marker_color=color,
            boxpoints=False
        )
    )

if measure == "pip_nn":
    fig_box.add_hline(
        y=RANDOM_PIP,
        line_dash="dash",
        line_color="gray",
        annotation_text="Random sequence (66.7%)",
        annotation_position="top left"
    )

fig_box.update_layout(
    boxmode="group",
    yaxis_title=MEASURE_NAMES[measure],
    height=420,
    template="plotly_white"
)

st.plotly_chart(fig_box, width="stretch")

st.caption(
    "In atrial fibrillation, intervals vary almost at random, "
    "so PIP approaches the random-sequence value of 66.7% and "
    "PAS (zig-zag runs) roughly triples. Normal PIP lies below "
    "this value in MIT-BIH and above it in the Long-Term AF "
    "Database, which is why a single PIP threshold works poorly."
)

st.divider()

st.caption(
    "ECG Insight | Model Performance | "
    "Academic research only. Not for clinical diagnosis."
)
