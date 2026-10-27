
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ecg_loader import (
    load_ecg_record,
    AVAILABLE_RECORDS
)
from src.rpeak_detection import (
    detect_rpeaks,
    calculate_rr_intervals
)
from src.hrf_analysis import (
    calculate_pip,
    costa_beat_codes,
    costa_hrf,
    reference_nn_mask
)


@st.cache_data(show_spinner=False)
def load_record(record_id, duration):
    return load_ecg_record(record_id, duration)


# ==========================================
# PAGE HEADER
# ==========================================

st.title("Heart Rate Fragmentation Analysis")

st.write(
    "Study changes in heartbeat intervals and "
    "calculate Percentage of Inflection Points (PIP)."
)

st.info(
    "PIP measures how often successive heartbeat "
    "interval changes switch direction or change "
    "to/from zero."
)

st.divider()


# ==========================================
# ANALYSIS SETTINGS
# ==========================================

st.subheader("HRF Analysis Settings")

with st.container(border=True):

    c1, c2 = st.columns(2)

    with c1:
        record_id = st.selectbox(
            "Select MIT-BIH Record",
            AVAILABLE_RECORDS
        )

    with c2:
        duration = st.slider(
            "Recording Duration (seconds)",
            min_value=10,
            max_value=60,
            value=60,
            step=10
        )

    source = st.radio(
        "Choose Interval Source",
        [
            "Reference NN intervals",
            "Reference all RR intervals",
            "Automatically detected RR intervals"
        ]
    )

    run_analysis = st.button(
        "Calculate HRF",
        type="primary",
        use_container_width=True
    )


# ==========================================
# RUN CALCULATION
# ==========================================

if run_analysis:

    try:
        with st.spinner("Calculating HRF..."):

            ecg = load_record(record_id, duration)

            fs = ecg["fs"]

            if source == "Reference NN intervals":

                rr = ecg["rr_intervals"]

                mask = reference_nn_mask(
                    ecg["beat_types"]
                )

                peaks = ecg["reference_peaks"]
                codes = costa_beat_codes(ecg["beat_types"])

            elif source == "Reference all RR intervals":

                rr = ecg["rr_intervals"]
                mask = None

                peaks = ecg["reference_peaks"]
                codes = np.ones(len(peaks), dtype=int)

            else:

                channel = ecg["signals"][:, 0]

                detection = detect_rpeaks(
                    channel,
                    fs
                )

                rr = calculate_rr_intervals(
                    detection["rpeaks"],
                    fs
                )

                mask = None

                peaks = detection["rpeaks"]
                codes = np.ones(len(peaks), dtype=int)

            result = calculate_pip(
                rr,
                fs,
                valid_mask=mask
            )

            # Reference HRF measures (Costa's method)
            hrf = costa_hrf(
                np.asarray(peaks) / fs,
                codes,
                fs
            )

            st.session_state["hrf_result"] = {
                "record_id": record_id,
                "duration": duration,
                "source": source,
                "rr": rr,
                "mask": mask,
                "result": result,
                "hrf": hrf,
                "fs": fs
            }

    except Exception as error:
        st.session_state.pop("hrf_result", None)
        st.error(f"Analysis failed: {error}")


# ==========================================
# DISPLAY RESULTS
# ==========================================

if "hrf_result" in st.session_state:

    saved = st.session_state["hrf_result"]

    rr = saved["rr"]
    mask = saved["mask"]
    result = saved["result"]
    hrf = saved["hrf"]

    st.divider()
    st.subheader("HRF Results")

    st.caption(
        f"Record {saved['record_id']} | "
        f"{saved['duration']} seconds | "
        f"{saved['source']}"
    )

    if saved["source"] != "Reference NN intervals":
        st.warning(
            "Exploratory RR analysis: abnormal beats "
            "or detection errors may affect PIP. "
            "These are not confirmed NN intervals."
        )

    if result["comparison_count"] < 10:
        st.warning(
            "There are fewer than 10 eligible comparisons. "
            "The result may be unstable. "
            "A longer recording is recommended."
        )

    # ======================================
    # SUMMARY METRICS
    # ======================================

    def percent(value):
        return f"{value:.2f}%" if value is not None else "N/A"

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("PIP", percent(hrf["pip"]))
    c2.metric("Hard Inflections", result["hard_count"])
    c3.metric("Soft Inflections", result["soft_count"])
    c4.metric(
        "Eligible Comparisons",
        result["comparison_count"]
    )

    c5, c6, c7, c8 = st.columns(4)

    c5.metric("PNNSS", percent(hrf["pnnss"]))
    c6.metric("PNNLS", percent(hrf["pnnls"]))
    c7.metric(
        "IALS",
        f"{hrf['ials']:.3f}" if hrf["ials"] is not None else "N/A"
    )
    c8.metric("PAS", percent(hrf["pas"]))

    st.caption(
        "PIP, PNNSS and PNNLS are calculated as in Costa's "
        "reference software (HRF.awk, PhysioNet). IALS and "
        "PAS follow Costa et al. (2017). Hard/soft counts "
        "and the charts below show every eligible comparison."
    )

    # ======================================
    # RR INTERVAL GRAPH
    # ======================================

    st.subheader("RR Interval Trend")

    rr_ms = np.asarray(rr) * 1000
    indices = result["inflection_indices"]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=np.arange(1, len(rr_ms) + 1),
            y=rr_ms,
            mode="lines+markers",
            name="RR Intervals",
            line=dict(
                color="#2563EB",
                width=2
            ),
            marker=dict(size=5)
        )
    )

    if len(indices) > 0:

        fig.add_trace(
            go.Scatter(
                x=indices + 1,
                y=rr_ms[indices],
                mode="markers",
                name="Inflection Points",
                marker=dict(
                    color="#E16C71",
                    size=12,
                    symbol="circle"
                )
            )
        )

    fig.update_layout(
        title="RR Intervals and Inflection Points",
        xaxis_title="Interval Number",
        yaxis_title="RR Interval (ms)",
        height=450,
        template="plotly_white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ======================================
    # INTERVAL DIFFERENCES
    # ======================================

    st.subheader("Changes Between RR Intervals")

    differences_ms = result["differences"] * 1000

    fig_diff = go.Figure()

    fig_diff.add_trace(
        go.Bar(
            x=np.arange(
                1,
                len(differences_ms) + 1
            ),
            y=differences_ms,
            marker_color=[
                "#0F9F9A" if d >= 0 else "#E16C71"
                for d in differences_ms
            ],
            name="RR Change"
        )
    )

    fig_diff.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray"
    )

    fig_diff.update_layout(
        title="Successive RR Interval Differences",
        xaxis_title="Difference Number",
        yaxis_title="Change (ms)",
        height=350,
        template="plotly_white"
    )

    st.plotly_chart(
        fig_diff,
        use_container_width=True
    )

    # ======================================
    # DATA TABLE
    # ======================================

    st.subheader("HRF Data")

    flags = np.zeros(len(rr), dtype=bool)
    flags[indices] = True

    if mask is None:
        eligible_intervals = np.ones(
            len(rr),
            dtype=bool
        )
    else:
        eligible_intervals = mask

    df = pd.DataFrame({
        "Interval": np.arange(1, len(rr) + 1),
        "RR (ms)": np.round(rr_ms, 2),
        "Eligible": eligible_intervals,
        "Inflection Point": flags
    })

    with st.expander("View HRF Data Table"):

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

    st.download_button(
        "Download HRF Results CSV",
        data=df.to_csv(index=False),
        file_name=(
            f"hrf_record_{saved['record_id']}.csv"
        ),
        mime="text/csv"
    )

    # ======================================
    # METHODOLOGY
    # ======================================

    with st.expander("How is PIP calculated?"):

        st.markdown("""
        1. Obtain RR intervals from heartbeat locations.
        2. Keep intervals between normal beats that lie
           between 0.3 and 1.8 seconds.
        3. Calculate differences between consecutive intervals.
        4. Convert differences into increasing, decreasing,
           or unchanged states.
        5. Mark inflection points: a change of direction
           (hard) or a change to or from no change (soft).
        6. Divide the number of inflection points by the
           number of intervals between the first and last
           inflection point, and multiply by 100.

        This follows Costa's reference software, and the
        implementation is checked against it in the tests.
        """)

    st.divider()

    st.caption(
        "ECG Insight | HRF Analysis | "
        "Academic research only. Not for clinical diagnosis."
    )
