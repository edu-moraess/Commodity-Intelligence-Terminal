"""
Design System — Commodity Intelligence Terminal
================================================
CSS + componentes visuais de nivel institucional (Bloomberg / FactSet style).

Uso em app.py (uma vez, apos set_page_config):
    from utils.design_system import inject_institutional_css, render_sidebar_brand
    inject_institutional_css()
    render_sidebar_brand()

NAO altera logica de negocio — apenas apresentacao.
"""

from __future__ import annotations

from typing import Optional, List, Tuple
from datetime import datetime, timezone, timedelta

import streamlit as st

from config.settings import APP_NAME, THEME


def inject_institutional_css() -> None:
    """Injeta o Design System completo (tipografia, sidebar, cards, tabelas, tabs)."""
    t = THEME
    bg = t["background"]
    surface = t["surface"]
    surface_alt = t.get("surface_alt", surface)
    border = t["border"]
    text = t["text"]
    text_muted = t["text_muted"]
    accent = t["accent"]
    positive = t["positive"]
    negative = t["negative"]

    def _hex_rgb(hx: str) -> str:
        hx = hx.lstrip("#")
        return "{},{},{}".format(int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16))

    accent_rgb = _hex_rgb(accent)

    css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}
.stApp {
    background-color: __BG__ !important;
}
.block-container {
    padding-top: 0.5rem !important;
    padding-bottom: 2rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    max-width: 1600px !important;
}

@media (max-width: 991px) {
    .block-container {
        padding-top: 0.4rem !important;
        padding-bottom: 1.25rem !important;
        padding-left: 0.7rem !important;
        padding-right: 0.7rem !important;
        max-width: 100% !important;
    }
    section[data-testid="stSidebar"] {
        min-width: 0 !important;
    }
    section[data-testid="stSidebar"][aria-expanded="false"] {
        width: 0 !important;
        min-width: 0 !important;
        margin: 0 !important;
    }
    section.main {
        margin-left: 0 !important;
        padding-left: 0 !important;
    }
    div[data-testid="stAppViewContainer"] > .main {
        margin-left: 0 !important;
    }
    [data-testid="collapsedControl"] {
        background-color: __SURFACE__ !important;
        border: 1px solid __BORDER__ !important;
        border-radius: 8px !important;
        color: __TEXT__ !important;
    }
    div[data-testid="stMetric"] {
        padding: 10px 12px !important;
    }
    div[data-testid="stMetric"] > div:nth-child(2) {
        font-size: 1.15rem !important;
    }
    h1 {
        font-size: 1.35rem !important;
    }
}

@media (max-width: 600px) {
    .block-container {
        padding-left: 0.6rem !important;
        padding-right: 0.6rem !important;
    }
}

h1 {
    color: __TEXT__ !important;
    font-weight: 700 !important;
    font-size: 1.75rem !important;
    letter-spacing: -0.03em !important;
    line-height: 1.25 !important;
    margin-bottom: 0.35rem !important;
}
h2 {
    color: __TEXT__ !important;
    font-weight: 600 !important;
    font-size: 1.2rem !important;
    letter-spacing: -0.02em !important;
    margin-top: 0.75rem !important;
}
h3 {
    color: __TEXT__ !important;
    font-weight: 600 !important;
    font-size: 1.0rem !important;
    letter-spacing: -0.01em !important;
}
p, .stMarkdown, label {
    color: __TEXT__ !important;
    font-size: 0.92rem !important;
}
[data-testid="stCaptionContainer"], .stCaption {
    color: __MUTED__ !important;
    font-size: 0.78rem !important;
}

div[data-testid="stMetricValue"],
div[data-testid="stDataFrame"],
.cit-mono {
    font-family: 'JetBrains Mono', 'SF Mono', 'Courier New', monospace !important;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, __SURFACE__ 0%, #0e0e12 100%) !important;
    border-right: 1px solid __BORDER__ !important;
    min-width: 260px !important;
}
section[data-testid="stSidebar"] > div {
    padding-top: 0.15rem !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {
    padding-top: 0 !important;
    margin-top: 0 !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarNavItems"] {
    padding-top: 0.15rem !important;
}

.cit-sidebar-brand {
    padding: 0.35rem 0.75rem 0.5rem 0.75rem !important;
    margin-bottom: 0.1rem !important;
}
.cit-sidebar-brand .cit-logo-row {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    margin-bottom: 0.35rem;
}
.cit-sidebar-brand .cit-logo-mark {
    width: 28px;
    height: 28px;
    border-radius: 7px;
    background: linear-gradient(135deg, __ACCENT__ 0%, #a8841c 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
    box-shadow: 0 0 12px rgba(__ACCENT_RGB__,0.28);
}
.cit-sidebar-brand .cit-logo-title {
    font-size: 0.82rem;
    font-weight: 700;
    color: __TEXT__;
    letter-spacing: -0.02em;
    line-height: 1.2;
}
.cit-sidebar-brand .cit-logo-sub {
    font-size: 0.65rem;
    color: __MUTED__;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-top: 1px;
}
.cit-sidebar-meta {
    padding: 0.5rem 0.85rem;
    font-size: 0.7rem;
    color: __MUTED__;
    border-top: 1px solid __BORDER__;
    margin-top: 0.4rem;
}
.cit-status-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: rgba(31,179,122,0.12);
    color: __POSITIVE__;
    border: 1px solid rgba(31,179,122,0.25);
    border-radius: 999px;
    padding: 0.15rem 0.55rem;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}
.cit-status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: __POSITIVE__;
    box-shadow: 0 0 6px __POSITIVE__;
}

section[data-testid="stSidebar"] [data-testid*="stSidebarNav"] a,
section[data-testid="stSidebar"] nav a,
section[data-testid="stSidebar"] [data-testid*="NavLink"] {
    border-left: 3px solid transparent !important;
    border-radius: 6px !important;
    margin: 2px 8px !important;
    padding: 0.5rem 0.7rem !important;
    color: __MUTED__ !important;
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    transition: background-color 0.12s ease, color 0.12s ease, border-color 0.12s ease !important;
}
section[data-testid="stSidebar"] [data-testid*="stSidebarNav"] a:hover,
section[data-testid="stSidebar"] nav a:hover,
section[data-testid="stSidebar"] [data-testid*="NavLink"]:hover {
    background-color: rgba(255,255,255,0.04) !important;
    color: __TEXT__ !important;
}
section[data-testid="stSidebar"] [data-testid*="stSidebarNav"] a[aria-current="page"],
section[data-testid="stSidebar"] nav a[aria-current="page"],
section[data-testid="stSidebar"] [data-testid*="NavLink"][aria-current="page"],
section[data-testid="stSidebar"] li[aria-current="page"] a {
    background-color: rgba(__ACCENT_RGB__,0.12) !important;
    border-left: 3px solid __ACCENT__ !important;
    color: __ACCENT__ !important;
    font-weight: 600 !important;
}

div[data-testid="stMetric"] {
    background: linear-gradient(180deg, __SURFACE__ 0%, __SURFACE_ALT__ 100%);
    border: 1px solid __BORDER__;
    border-radius: 10px;
    padding: 14px 16px 12px 16px;
    box-shadow: 0 1px 0 rgba(255,255,255,0.03) inset, 0 4px 16px rgba(0,0,0,0.18);
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}
div[data-testid="stMetric"]:hover {
    border-color: rgba(__ACCENT_RGB__,0.50);
    box-shadow: 0 1px 0 rgba(255,255,255,0.04) inset, 0 6px 20px rgba(0,0,0,0.25);
}
div[data-testid="stMetricLabel"] {
    color: __MUTED__ !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
div[data-testid="stMetric"] > div:nth-child(2) {
    font-size: 1.45rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    color: __TEXT__ !important;
}
div[data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
}
div[data-testid="stMetricDelta"] svg[data-testid="stMetricDeltaIcon-Up"] {
    fill: __POSITIVE__ !important;
}
div[data-testid="stMetricDelta"] svg[data-testid="stMetricDeltaIcon-Down"] {
    fill: __NEGATIVE__ !important;
}
div[data-testid="stMetricDelta"] [data-testid="stMetricDeltaIcon-Up"] \~ div,
div[data-testid="stMetricDelta"]:has(svg[data-testid="stMetricDeltaIcon-Up"]) {
    color: __POSITIVE__ !important;
}
div[data-testid="stMetricDelta"] [data-testid="stMetricDeltaIcon-Down"] \~ div,
div[data-testid="stMetricDelta"]:has(svg[data-testid="stMetricDeltaIcon-Down"]) {
    color: __NEGATIVE__ !important;
}

button[data-baseweb="tab"] {
    color: __MUTED__ !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.01em;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: __ACCENT__ !important;
    font-weight: 600 !important;
    border-bottom: 2px solid __ACCENT__ !important;
}
div[data-baseweb="tab-highlight"] {
    background-color: __ACCENT__ !important;
}

div.stButton > button {
    border-radius: 8px !important;
    border: 1px solid __BORDER__ !important;
    background-color: __SURFACE__ !important;
    color: __TEXT__ !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    transition: border-color 0.12s ease, color 0.12s ease, background 0.12s ease !important;
    box-shadow: none !important;
}
div.stButton > button:hover {
    border-color: __ACCENT__ !important;
    color: __ACCENT__ !important;
    background-color: rgba(__ACCENT_RGB__,0.08) !important;
}
div.stButton > button[kind="primary"],
div.stButton > button[data-testid="baseButton-primary"] {
    background: linear-gradient(180deg, __ACCENT__ 0%, #a8841c 100%) !important;
    color: #0a0a0c !important;
    border: none !important;
    font-weight: 600 !important;
}
div.stButton > button[kind="primary"]:hover {
    filter: brightness(1.08);
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="base-input"] {
    background-color: __SURFACE__ !important;
    border-color: __BORDER__ !important;
    border-radius: 8px !important;
}
div[data-baseweb="select"]:hover > div,
div[data-baseweb="input"]:hover > div {
    border-color: rgba(__ACCENT_RGB__,0.45) !important;
}

div[data-testid="stDataFrame"] {
    border: 1px solid __BORDER__ !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.15);
}
div[data-testid="stDataFrame"] thead tr th {
    background-color: __SURFACE_ALT__ !important;
    color: __MUTED__ !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
    border-bottom: 1px solid __BORDER__ !important;
}
div[data-testid="stDataFrame"] tbody tr:hover td {
    background-color: rgba(255,255,255,0.03) !important;
}

div[data-testid="stExpander"] {
    border: 1px solid __BORDER__ !important;
    border-radius: 10px !important;
    background-color: __SURFACE__ !important;
}
div[data-testid="stExpander"] summary {
    font-weight: 500 !important;
    color: __TEXT__ !important;
}

div[data-testid="stAlert"] {
    border-radius: 10px !important;
    border: 1px solid __BORDER__ !important;
}

hr {
    border: none !important;
    border-top: 1px solid __BORDER__ !important;
    margin: 1.1rem 0 !important;
    opacity: 0.85;
}

div[data-testid="stPlotlyChart"] {
    border: 1px solid #1a1a1a;
    border-radius: 10px;
    padding: 2px;
    background-color: #000000;
    box-shadow: 0 4px 20px rgba(0,0,0,0.35);
}

div[data-testid="stDownloadButton"] > button {
    border-radius: 8px !important;
    border: 1px solid __BORDER__ !important;
    background-color: transparent !important;
    color: __MUTED__ !important;
    font-size: 0.8rem !important;
}
div[data-testid="stDownloadButton"] > button:hover {
    border-color: __ACCENT__ !important;
    color: __ACCENT__ !important;
}

.cit-page-header {
    display: flex;
    flex-wrap: wrap;
    align-items: flex-start;
    justify-content: space-between;
    gap: 0.75rem;
    margin-bottom: 1.1rem;
    padding-bottom: 0.85rem;
    border-bottom: 1px solid __BORDER__;
}
.cit-page-header .cit-title {
    font-size: 1.55rem;
    font-weight: 700;
    color: __TEXT__;
    letter-spacing: -0.03em;
    margin: 0;
    line-height: 1.2;
}
.cit-page-header .cit-subtitle {
    font-size: 0.82rem;
    color: __MUTED__;
    margin-top: 0.25rem;
    max-width: 720px;
    line-height: 1.4;
}
.cit-page-header .cit-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    align-items: center;
}
.cit-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    background: __SURFACE__;
    border: 1px solid __BORDER__;
    border-radius: 6px;
    padding: 0.22rem 0.55rem;
    font-size: 0.68rem;
    color: __MUTED__;
    font-weight: 500;
    letter-spacing: 0.02em;
}
.cit-chip strong {
    color: __TEXT__;
    font-weight: 600;
}

.cit-section-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: __MUTED__;
    margin: 1.25rem 0 0.55rem 0;
}

::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: __BG__; }
::-webkit-scrollbar-thumb {
    background: __BORDER__;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover { background: __MUTED__; }

/* MainMenu (3 pontos) permanece visivel para print / settings */
footer { visibility: hidden; }
header[data-testid="stHeader"] {
    background-color: __BG__ !important;
    border-bottom: 1px solid __BORDER__ !important;
}
div[data-testid="stAppViewContainer"] {
    background-color: __BG__ !important;
}
section.main > div:first-child {
    padding-top: 0 !important;
}
div[data-testid="stToolbar"] {
    background-color: __BG__ !important;
}
.stApp > header {
    min-height: 2.5rem !important;
}
</style>
"""
    css = (
        css.replace("__BG__", bg)
        .replace("__SURFACE__", surface)
        .replace("__SURFACE_ALT__", surface_alt)
        .replace("__BORDER__", border)
        .replace("__TEXT__", text)
        .replace("__MUTED__", text_muted)
        .replace("__ACCENT__", accent)
        .replace("__ACCENT_RGB__", accent_rgb)
        .replace("__POSITIVE__", positive)
        .replace("__NEGATIVE__", negative)
    )
    st.markdown(css, unsafe_allow_html=True)


def render_sidebar_brand() -> None:
    """Branding + status + refresh no sidebar (visivel em todas as paginas)."""
    brasilia_tz = timezone(timedelta(hours=-3))
    now_brasilia = datetime.now(brasilia_tz)
    stamp = now_brasilia.strftime("%d/%m/%Y %H:%M") + " BRT"

    with st.sidebar:
        st.markdown(
            "<div class='cit-sidebar-brand'>"
            "<div class='cit-logo-row'>"
            "<div class='cit-logo-mark'>&#9670;</div>"
            "<div>"
            "<div class='cit-logo-title'>" + APP_NAME + "</div>"
            "</div>"
            "</div>"
            "<div class='cit-status-chip'>"
            "<span class='cit-status-dot'></span> Markets Online"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        if st.button("Atualizar dados", use_container_width=True, key="cit_refresh_cache"):
            st.cache_data.clear()
            st.rerun()
        st.markdown(
            "<div class='cit-sidebar-meta'><div>" + stamp + "</div></div>",
            unsafe_allow_html=True,
        )


def page_header(
    title: str,
    subtitle: str = "",
    chips: Optional[List[Tuple[str, str]]] = None,
) -> None:
    """Header institucional de pagina."""
    chips = chips or []
    chips_html = "".join(
        "<span class='cit-chip'><strong>" + lab + "</strong>&nbsp;" + val + "</span>"
        for lab, val in chips
    )
    sub_html = ("<div class='cit-subtitle'>" + subtitle + "</div>") if subtitle else ""
    st.markdown(
        "<div class='cit-page-header'>"
        "<div>"
        "<div class='cit-title'>" + title + "</div>"
        + sub_html
        + "</div>"
        "<div class='cit-meta'>" + chips_html + "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def section_label(text: str) -> None:
    st.markdown(
        "<div class='cit-section-label'>" + text + "</div>",
        unsafe_allow_html=True,
    )