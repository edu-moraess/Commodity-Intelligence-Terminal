"""
Design System — Commodity Intelligence Terminal
================================================
CSS + componentes visuais de nível institucional (Bloomberg / FactSet style).

Uso em app.py (uma vez, após set_page_config):
    from utils.design_system import inject_institutional_css, render_sidebar_brand
    inject_institutional_css()
    render_sidebar_brand()

NÃO altera lógica de negócio — apenas apresentação.
"""

from __future__ import annotations
from datetime import datetime, timezone, timedelta

import streamlit as st

from config.settings import APP_NAME, APP_ICON, THEME


def inject_institutional_css() -> None:
    """Injeta o Design System completo (tipografia, sidebar, cards, tabelas, tabs)."""
    t = THEME
    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Base ───────────────────────────────────────── */
html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}}
.stApp {{
    background-color: {t['background']} !important;
}}
.block-container {{
    padding-top: 1.25rem !important;
    padding-bottom: 2.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1600px !important;
}}

/* ── Tipografia ─────────────────────────────────── */
h1 {{
    color: {t['text']} !important;
    font-weight: 700 !important;
    font-size: 1.75rem !important;
    letter-spacing: -0.03em !important;
    line-height: 1.25 !important;
    margin-bottom: 0.35rem !important;
}}
h2 {{
    color: {t['text']} !important;
    font-weight: 600 !important;
    font-size: 1.2rem !important;
    letter-spacing: -0.02em !important;
    margin-top: 0.75rem !important;
}}
h3 {{
    color: {t['text']} !important;
    font-weight: 600 !important;
    font-size: 1.0rem !important;
    letter-spacing: -0.01em !important;
}}
p, .stMarkdown, label {{
    color: {t['text']} !important;
    font-size: 0.92rem !important;
}}
[data-testid="stCaptionContainer"], .stCaption {{
    color: {t['text_muted']} !important;
    font-size: 0.78rem !important;
}}

/* Números monoespaçados */
div[data-testid="stMetricValue"],
div[data-testid="stDataFrame"],
.cit-mono {{
    font-family: 'JetBrains Mono', 'SF Mono', 'Courier New', monospace !important;
}}

/* ── Sidebar ────────────────────────────────────── */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {t['surface']} 0%, #0e0e12 100%) !important;
    border-right: 1px solid {t['border']} !important;
    min-width: 260px !important;
}}
section[data-testid="stSidebar"] > div {{
    padding-top: 0.75rem !important;
}}

/* Brand block */
.cit-sidebar-brand {{
    padding: 0.5rem 0.85rem 0.85rem 0.85rem;
    margin-bottom: 0.25rem;
}}
.cit-sidebar-brand .cit-logo-row {{
    display: flex;
    align-items: center;
    gap: 0.55rem;
    margin-bottom: 0.35rem;
}}
.cit-sidebar-brand .cit-logo-mark {{
    width: 28px;
    height: 28px;
    border-radius: 7px;
    background: linear-gradient(135deg, {t['accent']} 0%, #a8841c 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
    box-shadow: 0 0 12px rgba(201,162,39,0.25);
}}
.cit-sidebar-brand .cit-logo-title {{
    font-size: 0.82rem;
    font-weight: 700;
    color: {t['text']};
    letter-spacing: -0.02em;
    line-height: 1.2;
}}
.cit-sidebar-brand .cit-logo-sub {{
    font-size: 0.65rem;
    color: {t['text_muted']};
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-top: 1px;
}}
.cit-sidebar-meta {{
    padding: 0.6rem 0.85rem;
    font-size: 0.7rem;
    color: {t['text_muted']};
    border-top: 1px solid {t['border']};
    margin-top: 0.5rem;
}}
.cit-status-chip {{
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: rgba(31,179,122,0.12);
    color: {t['positive']};
    border: 1px solid rgba(31,179,122,0.25);
    border-radius: 999px;
    padding: 0.15rem 0.55rem;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}}
.cit-status-dot {{
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: {t['positive']};
    box-shadow: 0 0 6px {t['positive']};
}}

/* Nav links */
section[data-testid="stSidebar"] [data-testid*="stSidebarNav"] a,
section[data-testid="stSidebar"] nav a,
section[data-testid="stSidebar"] [data-testid*="NavLink"] {{
    border-left: 3px solid transparent !important;
    border-radius: 6px !important;
    margin: 2px 8px !important;
    padding: 0.5rem 0.7rem !important;
    color: {t['text_muted']} !important;
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    transition: background-color 0.12s ease, color 0.12s ease, border-color 0.12s ease !important;
}}
section[data-testid="stSidebar"] [data-testid*="stSidebarNav"] a:hover,
section[data-testid="stSidebar"] nav a:hover,
section[data-testid="stSidebar"] [data-testid*="NavLink"]:hover {{
    background-color: rgba(255,255,255,0.04) !important;
    color: {t['text']} !important;
}}
section[data-testid="stSidebar"] [data-testid*="stSidebarNav"] a[aria-current="page"],
section[data-testid="stSidebar"] nav a[aria-current="page"],
section[data-testid="stSidebar"] [data-testid*="NavLink"][aria-current="page"],
section[data-testid="stSidebar"] li[aria-current="page"] a {{
    background-color: rgba(201,162,39,0.10) !important;
    border-left: 3px solid {t['accent']} !important;
    color: {t['accent']} !important;
    font-weight: 600 !important;
}}

/* ── KPI / Metric cards ─────────────────────────── */
div[data-testid="stMetric"] {{
    background: linear-gradient(180deg, {t['surface']} 0%, {t.get('surface_alt', t['surface'])} 100%);
    border: 1px solid {t['border']};
    border-radius: 10px;
    padding: 14px 16px 12px 16px;
    box-shadow: 0 1px 0 rgba(255,255,255,0.03) inset, 0 4px 16px rgba(0,0,0,0.18);
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}}
div[data-testid="stMetric"]:hover {{
    border-color: rgba(201,162,39,0.45);
    box-shadow: 0 1px 0 rgba(255,255,255,0.04) inset, 0 6px 20px rgba(0,0,0,0.25);
}}
div[data-testid="stMetricLabel"] {{
    color: {t['text_muted']} !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}}
div[data-testid="stMetric"] > div:nth-child(2) {{
    font-size: 1.45rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    color: {t['text']} !important;
}}
div[data-testid="stMetricDelta"] {{
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
}}
div[data-testid="stMetricDelta"] svg[data-testid="stMetricDeltaIcon-Up"] {{
    fill: {t['positive']} !important;
}}
div[data-testid="stMetricDelta"] svg[data-testid="stMetricDeltaIcon-Down"] {{
    fill: {t['negative']} !important;
}}
div[data-testid="stMetricDelta"] [data-testid="stMetricDeltaIcon-Up"] \~ div,
div[data-testid="stMetricDelta"]:has(svg[data-testid="stMetricDeltaIcon-Up"]) {{
    color: {t['positive']} !important;
}}
div[data-testid="stMetricDelta"] [data-testid="stMetricDeltaIcon-Down"] \~ div,
div[data-testid="stMetricDelta"]:has(svg[data-testid="stMetricDeltaIcon-Down"]) {{
    color: {t['negative']} !important;
}}

/* ── Tabs ───────────────────────────────────────── */
button[data-baseweb="tab"] {{
    color: {t['text_muted']} !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.01em;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: {t['accent']} !important;
    font-weight: 600 !important;
    border-bottom: 2px solid {t['accent']} !important;
}}
div[data-baseweb="tab-highlight"] {{
    background-color: {t['accent']} !important;
}}

/* ── Buttons ────────────────────────────────────── */
div.stButton > button {{
    border-radius: 8px !important;
    border: 1px solid {t['border']} !important;
    background-color: {t['surface']} !important;
    color: {t['text']} !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    transition: border-color 0.12s ease, color 0.12s ease, background 0.12s ease !important;
    box-shadow: none !important;
}}
div.stButton > button:hover {{
    border-color: {t['accent']} !important;
    color: {t['accent']} !important;
    background-color: rgba(201,162,39,0.06) !important;
}}
div.stButton > button[kind="primary"],
div.stButton > button[data-testid="baseButton-primary"] {{
    background: linear-gradient(180deg, {t['accent']} 0%, #a8841c 100%) !important;
    color: #0a0a0c !important;
    border: none !important;
    font-weight: 600 !important;
}}
div.stButton > button[kind="primary"]:hover {{
    filter: brightness(1.08);
}}

/* ── Inputs / selects ───────────────────────────── */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="base-input"] {{
    background-color: {t['surface']} !important;
    border-color: {t['border']} !important;
    border-radius: 8px !important;
}}
div[data-baseweb="select"]:hover > div,
div[data-baseweb="input"]:hover > div {{
    border-color: rgba(201,162,39,0.4) !important;
}}

/* ── Dataframes / tables ────────────────────────── */
div[data-testid="stDataFrame"] {{
    border: 1px solid {t['border']} !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.15);
}}
div[data-testid="stDataFrame"] thead tr th {{
    background-color: {t.get('surface_alt', t['surface'])} !important;
    color: {t['text_muted']} !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
    border-bottom: 1px solid {t['border']} !important;
}}
div[data-testid="stDataFrame"] tbody tr:hover td {{
    background-color: rgba(255,255,255,0.03) !important;
}}

/* ── Expander ───────────────────────────────────── */
div[data-testid="stExpander"] {{
    border: 1px solid {t['border']} !important;
    border-radius: 10px !important;
    background-color: {t['surface']} !important;
}}
div[data-testid="stExpander"] summary {{
    font-weight: 500 !important;
    color: {t['text']} !important;
}}

/* ── Alerts ─────────────────────────────────────── */
div[data-testid="stAlert"] {{
    border-radius: 10px !important;
    border: 1px solid {t['border']} !important;
}}

/* ── Dividers ───────────────────────────────────── */
hr {{
    border: none !important;
    border-top: 1px solid {t['border']} !important;
    margin: 1.1rem 0 !important;
    opacity: 0.85;
}}

/* ── Plotly container ───────────────────────────── */
div[data-testid="stPlotlyChart"] {{
    border: 1px solid {t['border']};
    border-radius: 10px;
    padding: 4px;
    background-color: {t['surface']};
    box-shadow: 0 4px 16px rgba(0,0,0,0.12);
}}

/* ── Download buttons ───────────────────────────── */
div[data-testid="stDownloadButton"] > button {{
    border-radius: 8px !important;
    border: 1px solid {t['border']} !important;
    background-color: transparent !important;
    color: {t['text_muted']} !important;
    font-size: 0.8rem !important;
}}
div[data-testid="stDownloadButton"] > button:hover {{
    border-color: {t['accent']} !important;
    color: {t['accent']} !important;
}}

/* ── Header institucional (página) ───────────────── */
.cit-page-header {{
    display: flex;
    flex-wrap: wrap;
    align-items: flex-start;
    justify-content: space-between;
    gap: 0.75rem;
    margin-bottom: 1.1rem;
    padding-bottom: 0.85rem;
    border-bottom: 1px solid {t['border']};
}}
.cit-page-header .cit-title {{
    font-size: 1.55rem;
    font-weight: 700;
    color: {t['text']};
    letter-spacing: -0.03em;
    margin: 0;
    line-height: 1.2;
}}
.cit-page-header .cit-subtitle {{
    font-size: 0.82rem;
    color: {t['text_muted']};
    margin-top: 0.25rem;
    max-width: 720px;
    line-height: 1.4;
}}
.cit-page-header .cit-meta {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    align-items: center;
}}
.cit-chip {{
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    background: {t['surface']};
    border: 1px solid {t['border']};
    border-radius: 6px;
    padding: 0.22rem 0.55rem;
    font-size: 0.68rem;
    color: {t['text_muted']};
    font-weight: 500;
    letter-spacing: 0.02em;
}}
.cit-chip strong {{
    color: {t['text']};
    font-weight: 600;
}}

/* ── Section label ──────────────────────────────── */
.cit-section-label {{
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: {t['text_muted']};
    margin: 1.25rem 0 0.55rem 0;
}}

/* ── Scrollbar discreta ─────────────────────────── */
::-webkit-scrollbar {{ width: 8px; height: 8px; }}
::-webkit-scrollbar-track {{ background: {t['background']}; }}
::-webkit-scrollbar-thumb {{
    background: {t['border']};
    border-radius: 4px;
}}
::-webkit-scrollbar-thumb:hover {{ background: {t['text_muted']}; }}

/* ── Hide Streamlit chrome ──────────────────────── */
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{
    background: transparent !important;
}}
</style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_brand() -> None:
    """Branding + status + refresh no sidebar (visível em todas as páginas)."""
    brasilia_tz = timezone(timedelta(hours=-3))
    now_brasilia = datetime.now(brasilia_tz)

    with st.sidebar:
        st.markdown(
            f"""
<div class="cit-sidebar-brand">
  <div class="cit-logo-row">
    <div class="cit-logo-mark">◈</div>
    <div>
      <div class="cit-logo-title">{APP_NAME}</div>
      <div class="cit-logo-sub">Institutional Quant Platform</div>
    </div>
  </div>
  <div class="cit-status-chip"><span class="cit-status-dot"></span> Markets Online</div>
</div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Fontes: Yahoo Finance · FRED · fallback sintético")
        if st.button("↻ Atualizar dados", use_container_width=True, key="cit_refresh_cache"):
            st.cache_data.clear()
            st.rerun()
        st.markdown(
            f"""
<div class="cit-sidebar-meta">
  <div>{now_brasilia.strftime('%d/%m/%Y · %H:%M')} BRT</div>
  <div style="margin-top:4px;opacity:0.7">v4.5 · CIT Terminal</div>
</div>
            """,
            unsafe_allow_html=True,
        )


def page_header(
    title: str,
    subtitle: str = "",
    chips: list[tuple[str, str]] | None = None,
) -> None:
    """
    Header institucional de página.

    chips: lista de (label, value), ex. [("Fonte", "Yahoo"), ("Janela", "252d")]
    """
    chips = chips or []
    chips_html = "".join(
        f'<span class="cit-chip"><strong>{lab}</strong>&nbsp;{val}</span>'
        for lab, val in chips
    )
    st.markdown(
        f"""
<div class="cit-page-header">
  <div>
    <div class="cit-title">{title}</div>
    {f'<div class="cit-subtitle">{subtitle}</div>' if subtitle else ''}
  </div>
  <div class="cit-meta">{chips_html}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def section_label(text: str) -> None:
    st.markdown(f'<div class="cit-section-label">{text}</div>', unsafe_allow_html=True)