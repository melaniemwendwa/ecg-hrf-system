"""Shared look and feel for the Streamlit pages."""

from html import escape

import numpy as np
import streamlit as st


# Colours by role. Series colours follow the validated
# categorical palette in .streamlit/config.toml, in order.
INK = "#14213d"
MUTED = "#5b6478"
GRID = "#e6eaf1"
SURFACE = "#ffffff"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

NORMAL_COLOR = SERIES[0]
ABNORMAL_COLOR = SERIES[1]

FONT = "Inter, Segoe UI, Helvetica, Arial, sans-serif"


CSS = """
<style>
.block-container {
    max-width: 1280px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}

/* Hero banner */
.ui-hero {
    display: flex;
    align-items: center;
    gap: 2rem;
    flex-wrap: wrap;
    background: linear-gradient(135deg, #0f2a4a 0%, #1d4f8f 100%);
    color: #ffffff;
    border-radius: 1.25rem;
    padding: 2.25rem 2.5rem;
    margin-bottom: 1.75rem;
}
.ui-hero-text { flex: 1 1 420px; min-width: 0; }
.ui-hero-art { flex: 0 1 360px; min-width: 240px; }
.ui-hero-art svg { width: 100%; height: auto; display: block; }
.ui-eyebrow {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #a9c8f0;
    margin-bottom: 0.5rem;
}
.ui-hero h1 {
    color: #ffffff;
    font-size: 2.4rem;
    line-height: 1.15;
    margin: 0 0 0.75rem 0;
    padding: 0;
}
.ui-hero p {
    color: #d7e4f5;
    font-size: 1.05rem;
    line-height: 1.6;
    margin: 0;
}
.ui-chips { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 1.25rem; }
.ui-chip {
    background: rgba(255, 255, 255, 0.12);
    border: 1px solid rgba(255, 255, 255, 0.22);
    color: #ffffff;
    border-radius: 999px;
    padding: 0.3rem 0.8rem;
    font-size: 0.82rem;
}

/* Stat tiles */
.ui-tiles {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-bottom: 1.75rem;
}
.ui-tile {
    background: #ffffff;
    border: 1px solid #e3e8f0;
    border-radius: 1rem;
    padding: 1.1rem 1.25rem;
}
.ui-tile-label {
    font-size: 0.82rem;
    font-weight: 500;
    color: #5b6478;
}
.ui-tile-value {
    font-size: 2rem;
    font-weight: 700;
    color: #14213d;
    line-height: 1.2;
    margin: 0.25rem 0;
}
.ui-tile-note { font-size: 0.82rem; color: #5b6478; }

/* Section headings */
.ui-section { margin: 2rem 0 1rem 0; }
.ui-section h2 {
    font-size: 1.35rem;
    margin: 0;
    padding: 0;
}
.ui-section p { color: #5b6478; margin: 0.25rem 0 0 0; }

/* Step cards */
.ui-step-number {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 2rem;
    height: 2rem;
    border-radius: 0.6rem;
    background: #e8f1fc;
    color: #2a78d6;
    font-weight: 700;
    margin-bottom: 0.75rem;
}
.ui-step-title { font-weight: 600; font-size: 1.05rem; color: #14213d; }
.ui-step-text {
    color: #5b6478;
    font-size: 0.92rem;
    line-height: 1.5;
    margin: 0.35rem 0 0.75rem 0;
    min-height: 4.2rem;
}

/* Finding list */
.ui-finding {
    display: flex;
    gap: 0.75rem;
    padding: 0.65rem 0;
    border-bottom: 1px solid #eef1f6;
}
.ui-finding:last-child { border-bottom: none; }
.ui-finding-mark {
    flex: 0 0 auto;
    width: 0.5rem;
    border-radius: 999px;
    background: #2a78d6;
}
.ui-finding-text { color: #14213d; font-size: 0.95rem; line-height: 1.5; }

/* Footer */
.ui-footer {
    margin-top: 3rem;
    padding-top: 1.25rem;
    border-top: 1px solid #e3e8f0;
    color: #5b6478;
    font-size: 0.85rem;
}
</style>
"""


def apply_theme():
    """Inject the shared CSS. Call once per page run."""

    st.markdown(CSS, unsafe_allow_html=True)


def ecg_trace_svg(width=360, height=120, beats=4):
    """A synthetic ECG trace as inline SVG, for decoration."""

    t = np.linspace(0, beats, 600)
    phase = t % 1.0

    def bump(centre, scale, amplitude):
        return amplitude * np.exp(-((phase - centre) / scale) ** 2)

    signal = (
        bump(0.20, 0.040, 0.12)     # P wave
        - bump(0.33, 0.012, 0.15)   # Q
        + bump(0.36, 0.012, 1.00)   # R
        - bump(0.39, 0.014, 0.25)   # S
        + bump(0.62, 0.070, 0.28)   # T wave
    )

    x = t / beats * width
    y = height * 0.72 - signal * height * 0.6

    points = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(x, y))

    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="ECG trace">'
        f'<polyline points="{points}" fill="none" stroke="#7fb3f0" '
        f'stroke-width="2.5" stroke-linejoin="round" '
        f'stroke-linecap="round"/></svg>'
    )


def hero(eyebrow, title, text, chips=(), art=True):
    """Page banner with title, description and tags."""

    chips_html = "".join(
        f'<span class="ui-chip">{escape(c)}</span>' for c in chips
    )

    art_html = (
        f'<div class="ui-hero-art">{ecg_trace_svg()}</div>' if art else ""
    )

    st.markdown(
        f"""
        <div class="ui-hero">
          <div class="ui-hero-text">
            <div class="ui-eyebrow">{escape(eyebrow)}</div>
            <h1>{escape(title)}</h1>
            <p>{escape(text)}</p>
            <div class="ui-chips">{chips_html}</div>
          </div>
          {art_html}
        </div>
        """,
        unsafe_allow_html=True
    )


def stat_tiles(tiles):
    """Row of headline numbers: [(label, value, note), ...]."""

    html = "".join(
        f"""
        <div class="ui-tile">
          <div class="ui-tile-label">{escape(label)}</div>
          <div class="ui-tile-value">{escape(value)}</div>
          <div class="ui-tile-note">{escape(note)}</div>
        </div>
        """
        for label, value, note in tiles
    )

    st.markdown(f'<div class="ui-tiles">{html}</div>', unsafe_allow_html=True)


def section(title, text=None):
    """Section heading with an optional one-line description."""

    text_html = f"<p>{escape(text)}</p>" if text else ""

    st.markdown(
        f'<div class="ui-section"><h2>{escape(title)}</h2>{text_html}</div>',
        unsafe_allow_html=True
    )


def step_card(number, title, text):
    """Numbered step inside a bordered container."""

    st.markdown(
        f"""
        <div class="ui-step-number">{number}</div>
        <div class="ui-step-title">{escape(title)}</div>
        <div class="ui-step-text">{escape(text)}</div>
        """,
        unsafe_allow_html=True
    )


def findings(items):
    """Short list of key points."""

    html = "".join(
        f"""
        <div class="ui-finding">
          <div class="ui-finding-mark"></div>
          <div class="ui-finding-text">{escape(item)}</div>
        </div>
        """
        for item in items
    )

    st.markdown(html, unsafe_allow_html=True)


def footer(page):
    st.markdown(
        f"""
        <div class="ui-footer">
          ECG Insight · {escape(page)} · Academic research prototype.
          Not intended for clinical diagnosis.
        </div>
        """,
        unsafe_allow_html=True
    )


def style_figure(fig, height=400):
    """Shared chart styling: quiet grid, readable type."""

    fig.update_layout(
        template="plotly_white",
        height=height,
        font=dict(family=FONT, size=13, color=INK),
        title_font=dict(size=15, color=INK),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        margin=dict(l=16, r=16, t=48, b=16),
        hoverlabel=dict(font=dict(family=FONT)),
        legend=dict(font=dict(color=MUTED))
    )
    fig.update_xaxes(
        gridcolor=GRID,
        zerolinecolor=GRID,
        linecolor=GRID,
        tickfont=dict(color=MUTED)
    )
    fig.update_yaxes(
        gridcolor=GRID,
        zerolinecolor=GRID,
        linecolor=GRID,
        tickfont=dict(color=MUTED)
    )

    return fig
