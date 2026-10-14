
import streamlit as st
import plotly.graph_objects as go
import numpy as np

from src.ecg_loader import load_ecg_record


st.set_page_config(
    page_title="ECG HRF Analysis",
    page_icon="❤️",
    layout="wide"
)


@st.cache_data
def get_ecg(record_name, duration):
    return load_ecg_record(record_name, duration)


st.title("ECG Heart Rate Fragmentation Analysis")
st.caption(
    "ECG signal processing and machine learning "
    "classification research system"
)

st.divider()

st.sidebar.header("ECG Settings")

record_name = st.sidebar.selectbox(
    "Select MIT-BIH Record",
    ["100", "101", "102", "103", "105", "106"]
)

duration = st.sidebar.slider(
    "Signal Duration (seconds)",
    min_value=5,
    max_value=60,
    value=10,
    step=5
)

st.sidebar.info(
    "Data Source: MIT-BIH Arrhythmia Database"
)

st.subheader("ECG Signal Analysis")

try:
    with st.spinner("Loading ECG recording..."):
        ecg = get_ecg(record_name, duration)

    signal = ecg["signal"]
    time = ecg["time"]
    peaks = ecg["peaks"]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Record", record_name)
    col2.metric(
        "Sampling Rate",
        f"{ecg['fs']} Hz"
    )
    col3.metric(
        "Annotated Beats",
        len(peaks)
    )
    col4.metric(
        "Estimated Heart Rate",
        (
            f"{ecg['heart_rate']:.1f} BPM"
            if ecg["heart_rate"] is not None
            else "N/A"
        )
    )

    st.subheader("ECG Waveform")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=time,
            y=signal,
            mode="lines",
            name="ECG Signal",
            line=dict(
                color="#2563eb",
                width=1.5
            )
        )
    )

    if len(peaks) > 0:
        fig.add_trace(
            go.Scatter(
                x=time[peaks],
                y=signal[peaks],
                mode="markers",
                name="Annotated Beats",
                marker=dict(
                    color="red",
                    size=7
                )
            )
        )

    fig.update_layout(
        title=f"MIT-BIH Record {record_name}",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude (mV)",
        height=450,
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("RR Interval Analysis")

    rr = ecg["rr_intervals"]

    if len(rr) > 0:
        rr_ms = rr * 1000

        fig_rr = go.Figure()

        fig_rr.add_trace(
            go.Scatter(
                x=np.arange(1, len(rr_ms) + 1),
                y=rr_ms,
                mode="lines+markers",
                name="RR Intervals"
            )
        )

        fig_rr.update_layout(
            xaxis_title="Interval Number",
            yaxis_title="RR Interval (ms)",
            height=350
        )

        st.plotly_chart(
            fig_rr,
            use_container_width=True
        )

        st.metric(
            "Mean RR Interval",
            f"{np.mean(rr_ms):.1f} ms"
        )
    else:
        st.warning(
            "Not enough beats to calculate RR intervals."
        )

    st.info(
        "Beat locations are reference annotations, "
        "not predictions. Rhythm classification "
        "has not been implemented yet."
    )

except Exception as error:
    st.error(f"Unable to load ECG: {error}")
    st.info(
        "Check your internet connection and "
        "confirm that WFDB is installed."
    )
