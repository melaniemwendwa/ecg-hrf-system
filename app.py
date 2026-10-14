
import streamlit as st

st.set_page_config(
    page_title="ECG Insight",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
.stApp {
    background-color: #F4F7FB;
}

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
}

h1, h2, h3 {
    color: #10243A;
}
</style>
""", unsafe_allow_html=True)

# Website navigation
dashboard = st.Page(
    "pages/dashboard.py",
    title="Dashboard",
    icon=":material/dashboard:",
    default=True
)

ecg = st.Page(
    "pages/ecg_analysis.py",
    title="ECG Analysis",
    icon=":material/monitor_heart:"
)

hrf = st.Page(
    "pages/hrf_analysis.py",
    title="HRF Analysis",
    icon=":material/analytics:"
)

classification = st.Page(
    "pages/classification.py",
    title="Classification",
    icon=":material/psychology:"
)

performance = st.Page(
    "pages/performance.py",
    title="Model Performance",
    icon=":material/bar_chart:"
)

about = st.Page(
    "pages/about.py",
    title="About",
    icon=":material/info:"
)

pg = st.navigation(
    [
        dashboard,
        ecg,
        hrf,
        classification,
        performance,
        about
    ],
    position="top"
)

st.markdown("""
<div style="
    background: #10243A;
    color: white;
    padding: 20px 25px;
    border-radius: 12px;
    margin-bottom: 25px;
">
    <div style="
        font-size: 27px;
        font-weight: bold;
        color: white;
    ">
        💚 ECG Insight
    </div>
    <div style="
        font-size: 14px;
        color: #B8D1DE;
    ">
        Heart Rate Fragmentation Analysis System
    </div>
</div>
""", unsafe_allow_html=True)

pg.run()
