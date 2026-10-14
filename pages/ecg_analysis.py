
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ecg_loader import (
    load_ecg_record,
    AVAILABLE_RECORDS
)


# ==========================================
# LOAD ECG DATA
# ==========================================

@st.cache_data(show_spinner=False)
def get_ecg_data(record_id, duration):
    return load_ecg_record(record_id, duration)


# ==========================================
# PAGE HEADER
# ==========================================

st.title("ECG Signal Analysis")

st.write(
    "Load and explore ECG recordings from the "
    "MIT-BIH Arrhythmia Database."
)

st.info(
    "This phase uses expert reference annotations. "
    "Automatic R-peak detection will be added next."
)

st.divider()


# ==========================================
# ECG DATA SOURCE
# ==========================================

st.subheader("ECG Data Source")

with st.container(border=True):

    source = st.radio(
        "Choose ECG source",
        [
            "MIT-BIH Dataset",
            "Upload ECG File"
        ],
        horizontal=True
    )

    if source == "Upload ECG File":
        st.info(
            "Uploading your own ECG recording "
            "will be implemented in a later phase. "
            "For now, select an MIT-BIH record."
        )

    else:
        col1, col2 = st.columns(2)

        with col1:
            record_id = st.selectbox(
                "Select Record ID",
                AVAILABLE_RECORDS,
                index=0
            )

        with col2:
            duration = st.slider(
                "Recording Duration (seconds)",
                min_value=5,
                max_value=60,
                value=10,
                step=5
            )

        analyze = st.button(
            "Analyze ECG",
            type="primary",
            use_container_width=True
        )


# ==========================================
# ECG ANALYSIS
# ==========================================

if source == "MIT-BIH Dataset":

    if analyze:

        with st.spinner(
            "Loading ECG recording from PhysioNet..."
        ):
            try:
                ecg = get_ecg_data(
                    record_id,
                    duration
                )

                st.session_state["ecg_data"] = ecg

            except Exception as error:
                st.error(
                    f"Unable to load ECG: {error}"
                )

                st.info(
                    "Check your internet connection "
                    "and try again."
                )

                st.session_state.pop(
                    "ecg_data", None
                )

    # Display saved analysis from session
    if "ecg_data" in st.session_state:

        ecg = st.session_state["ecg_data"]

        signals = ecg["signals"]
        time = ecg["time"]
        fs = ecg["fs"]

        reference_peaks = ecg["reference_peaks"]
        rr_intervals = ecg["rr_intervals"]

        st.divider()

        # ==================================
        # CHANNEL SELECTION
        # ==================================

        st.subheader("Recording Overview")

        channel_names = ecg["channel_names"]

        channel_index = st.selectbox(
            "Select ECG Channel",
            options=list(range(len(channel_names))),
            format_func=lambda i: (
                f"Channel {i + 1}: {channel_names[i]}"
            )
        )

        signal = signals[:, channel_index]

        # ==================================
        # SUMMARY METRICS
        # ==================================

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Record ID",
            ecg["record_id"]
        )

        c2.metric(
            "Sampling Rate",
            f"{fs:g} Hz"
        )

        c3.metric(
            "Annotated Beats",
            len(reference_peaks)
        )

        heart_rate = ecg["mean_heart_rate"]

        c4.metric(
            "Estimated Heart Rate",
            (
                f"{heart_rate:.1f} BPM"
                if heart_rate is not None
                else "N/A"
            )
        )

        # ==================================
        # ECG WAVEFORM
        # ==================================

        st.subheader("ECG Waveform")

        show_annotations = st.checkbox(
            "Show reference beat annotations",
            value=True
        )

        fig = go.Figure()

        # ECG waveform
        fig.add_trace(
            go.Scatter(
                x=time,
                y=signal,
                mode="lines",
                name="ECG Signal",
                line=dict(
                    color="#2563EB",
                    width=1.5
                )
            )
        )

        # Expert reference beat locations
        if (
            show_annotations
            and len(reference_peaks) > 0
        ):
            fig.add_trace(
                go.Scatter(
                    x=time[reference_peaks],
                    y=signal[reference_peaks],
                    mode="markers",
                    name="Reference Beats",
                    marker=dict(
                        color="#E16C71",
                        size=8
                    ),
                    text=ecg["beat_types"],
                    hovertemplate=(
                        "Time: %{x:.3f} s"
                        "<br>Amplitude: %{y:.3f} mV"
                        "<br>Beat type: %{text}"
                        "<extra></extra>"
                    )
                )
            )

        fig.update_layout(
            title=(
                f"MIT-BIH Record {ecg['record_id']} "
                f"- {channel_names[channel_index]}"
            ),
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude (mV)",
            height=460,
            template="plotly_white",
            hovermode="closest",
            legend=dict(
                orientation="h",
                y=1.12,
                x=0
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.caption(
            "Red markers indicate expert-annotated "
            "heartbeat locations, not automatically "
            "detected peaks."
        )

        # ==================================
        # RR INTERVAL ANALYSIS
        # ==================================

        st.subheader("RR Interval Analysis")

        if len(rr_intervals) > 0:

            rr_ms = rr_intervals * 1000

            m1, m2, m3 = st.columns(3)

            m1.metric(
                "Mean RR Interval",
                f"{np.mean(rr_ms):.1f} ms"
            )

            m2.metric(
                "Minimum RR",
                f"{np.min(rr_ms):.1f} ms"
            )

            m3.metric(
                "Maximum RR",
                f"{np.max(rr_ms):.1f} ms"
            )

            fig_rr = go.Figure()

            fig_rr.add_trace(
                go.Scatter(
                    x=np.arange(
                        1, len(rr_ms) + 1
                    ),
                    y=rr_ms,
                    mode="lines+markers",
                    name="RR Intervals",
                    line=dict(
                        color="#0F9F9A",
                        width=2
                    )
                )
            )

            fig_rr.update_layout(
                title="Beat-to-Beat RR Intervals",
                xaxis_title="Interval Number",
                yaxis_title="RR Interval (ms)",
                height=350,
                template="plotly_white"
            )

            st.plotly_chart(
                fig_rr,
                use_container_width=True
            )

            # ==================================
            # RR INTERVAL DATA TABLE
            # ==================================

            with st.expander(
                "View RR Interval Data"
            ):

                rr_df = pd.DataFrame({
                    "Interval Number": np.arange(
                        1, len(rr_ms) + 1
                    ),
                    "RR Interval (ms)": np.round(
                        rr_ms, 2
                    )
                })

                st.dataframe(
                    rr_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    "Download RR Intervals CSV",
                    data=rr_df.to_csv(index=False),
                    file_name=(
                        f"record_{ecg['record_id']}"
                        "_rr_intervals.csv"
                    ),
                    mime="text/csv"
                )

        else:
            st.warning(
                "Not enough annotated beats "
                "to calculate RR intervals."
            )

        # ==================================
        # RECORD INFORMATION
        # ==================================

        with st.expander(
            "About this ECG recording"
        ):
            st.write(
                f"**Record:** {ecg['record_id']}"
            )

            st.write(
                f"**Duration:** "
                f"{ecg['duration']:.1f} seconds"
            )

            st.write(
                f"**Sampling frequency:** {fs:g} Hz"
            )

            st.write(
                f"**Channels:** "
                f"{', '.join(channel_names)}"
            )

            st.write(
                "**Dataset:** MIT-BIH "
                "Arrhythmia Database"
            )

            st.write(
                "**Source:** PhysioNet"
            )

        st.divider()

        st.caption(
            "ECG Insight | ECG Signal Analysis | "
            "Academic research only. "
            "Not intended for clinical diagnosis."
        )
