
import streamlit as st

from src.ui import apply_theme

st.set_page_config(
    page_title="ECG Insight",
    page_icon=":material/monitor_heart:",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.logo("assets/logo.svg", icon_image="assets/icon.svg", size="large")

apply_theme()

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

pg.run()
