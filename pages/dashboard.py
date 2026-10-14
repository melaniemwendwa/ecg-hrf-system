
import streamlit as st
import plotly.graph_objects as go
import numpy as np


# ==========================================
# DASHBOARD HEADER
# ==========================================

st.title("Dashboard")
st.markdown("### Welcome to ECG Insight")

st.write(
    "A web-based system for ECG signal analysis, "
    "Heart Rate Fragmentation (HRF), and "
    "machine learning-based rhythm classification."
)

st.info(
    "Research Prototype | Developed for academic "
    "and educational purposes."
)

st.write("")


# ==========================================
# FEATURE CARDS
# ==========================================

st.subheader("Explore the System")

col1, col2, col3, col4 = st.columns(4)


with col1:
    with st.container(border=True):
        st.markdown("### 💓 ECG Analysis")

        st.write(
            "Load ECG recordings, visualize heart "
            "signals, and detect R-peaks."
        )

        st.write("")

        if st.button(
            "Open ECG Analysis →",
            use_container_width=True,
            key="open_ecg"
        ):
            st.switch_page("pages/ecg_analysis.py")


with col2:
    with st.container(border=True):
        st.markdown("### 📊 HRF Analysis")

        st.write(
            "Analyze RR intervals and calculate "
            "Percentage of Inflection Points."
        )

        st.write("")

        if st.button(
            "Open HRF Analysis →",
            use_container_width=True,
            key="open_hrf"
        ):
            st.switch_page("pages/hrf_analysis.py")


with col3:
    with st.container(border=True):
        st.markdown("### 🧠 Classification")

        st.write(
            "Classify ECG rhythms as normal "
            "or abnormal using machine learning."
        )

        st.write("")

        if st.button(
            "Open Classification →",
            use_container_width=True,
            key="open_classification"
        ):
            st.switch_page("pages/classification.py")


with col4:
    with st.container(border=True):
        st.markdown("### 📈 Model Performance")

        st.write(
            "Evaluate and compare machine "
            "learning classification models."
        )

        st.write("")

        if st.button(
            "View Performance →",
            use_container_width=True,
            key="open_performance"
        ):
            st.switch_page("pages/performance.py")


# ==========================================
# ECG WAVEFORM PREVIEW
# ==========================================

st.write("")

left, right = st.columns([2, 1])


with left:
    with st.container(border=True):

        st.subheader("Sample ECG Signal")

        st.caption(
            "Illustrative ECG waveform for "
            "dashboard demonstration."
        )

        fs = 360
        time = np.arange(0, 6, 1 / fs)

        signal = np.zeros_like(time)

        # Generate a synthetic ECG-like waveform.
        for beat_time in np.arange(0.5, 6, 1.0):

            # P wave
            signal += 0.12 * np.exp(
                -((time - (beat_time - 0.18)) / 0.04) ** 2
            )

            # Q wave
            signal -= 0.15 * np.exp(
                -((time - (beat_time - 0.03)) / 0.012) ** 2
            )

            # R wave
            signal += 1.10 * np.exp(
                -((time - beat_time) / 0.01) ** 2
            )

            # S wave
            signal -= 0.25 * np.exp(
                -((time - (beat_time + 0.03)) / 0.015) ** 2
            )

            # T wave
            signal += 0.25 * np.exp(
                -((time - (beat_time + 0.24)) / 0.08) ** 2
            )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=time,
                y=signal,
                mode="lines",
                name="Synthetic ECG",
                line=dict(
                    color="#0F9F9A",
                    width=2
                )
            )
        )

        fig.update_layout(
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude (arbitrary units)",
            height=350,
            template="plotly_white",
            showlegend=False,
            margin=dict(
                l=20, r=20, t=20, b=20
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ==========================================
# QUICK START PANEL
# ==========================================

with right:
    with st.container(border=True):

        st.subheader("Quick Start")

        st.markdown("""
        **1. Select ECG Analysis**

        Choose a recording from the MIT-BIH dataset
        or upload your own ECG file once supported.

        **2. Analyze the Signal**

        View ECG waveforms and automatically
        detected R-peaks.

        **3. Explore HRF**

        Calculate RR intervals and PIP.

        **4. Classify the Rhythm**

        Run machine learning models once trained.

        **5. Review Results**

        Examine predictions and model performance.
        """)


# ==========================================
# SYSTEM OVERVIEW
# ==========================================

st.write("")

st.subheader("System Overview")

c1, c2, c3, c4 = st.columns(4)

with c1:
    with st.container(border=True):
        st.caption("Dataset")
        st.markdown("### MIT-BIH")
        st.write("48 ECG recordings")

with c2:
    with st.container(border=True):
        st.caption("Planned File Support")
        st.markdown("### CSV / WFDB")
        st.write("ECG uploads")

with c3:
    with st.container(border=True):
        st.caption("Core HRF Feature")
        st.markdown("### PIP")
        st.write("Inflection point analysis")

with c4:
    with st.container(border=True):
        st.caption("Classification")
        st.markdown("### Binary")
        st.write("Normal / Abnormal")


# ==========================================
# FOOTER
# ==========================================

st.write("")
st.divider()

st.caption(
    "ECG Insight | Heart Rate Fragmentation "
    "Analysis System | Academic Research Prototype"
)

st.caption(
    "For educational and research purposes only. "
    "Not intended for clinical diagnosis."
)
