"""
Commodity Intelligence Terminal — Entry Point
=================================================
Página inicial: overview do mercado, KPIs de topo e navegação para os
módulos (Streamlit multipage — cada arquivo em pages/ vira uma página
automaticamente, listada na sidebar).
"""

import streamlit as st
import pandas as pd

from config.settings import APP_NAME, APP_ICON, ENERGY_ASSETS, METALS_ASSETS, AGRI_ASSETS, THEME
from data.data_manager import load_price_history_bulk
from analytics import metrics
from charts import plotly_charts as charts

st.set_page_config(page_title=APP_NAME, page_icon=APP_ICON, layout="wide", initial_sidebar_state="expanded")

# --------------------------------------------------------------------------
# TEMA ESCURO INSTITUCIONAL (CSS)
# --------------------------------------------------------------------------
st.markdown(f"""
<style>
    .stApp {{ background-color: {THEME['background']}; }}
    section[data-testid="stSidebar"] {{
        background-color: {THEME['surface']};
        border-right: 1px solid {THEME['border']};
    }}
    div[data-testid="stMetric"] {{
        background-color: {THEME['surface']};
        border: 1px solid {THEME['border']};
        border-radius: 10px;
        padding: 14px 16px;
    }}
    div[data-testid="stMetricLabel"] {{ color: {THEME['text_muted']}; }}
    h1, h2, h3 {{ color: {THEME['text']}; }}
    .stTabs [data-baseweb="tab"] {{ color: {THEME['text_muted']}; }}
    .stTabs [aria-selected="true"] {{ color: {THEME['accent']} !important; }}
    div[data-testid="stDataFrame"] {{ border: 1px solid {THEME['border']}; border-radius: 8px; }}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"## {APP_ICON} {APP_NAME}")
    st.caption("Institutional Quant Research Platform")
    st.divider()
    st.markdown(
        "**Módulos:**\n"
        "- 🌍 Dashboard Global\n"
        "- 🛢️ Energy Analytics\n"
        "- ⚙️ Metals Analytics\n"
        "- 🌾 Agriculture Analytics\n"
        "- 🇧🇷 Commodities Brasileiras\n"
        "- 🔗 Macro & Correlações\n"
        "- ⚠️ Risk Analytics\n"
        "- 📈 Forecast\n"
        "- 🧮 Quant Research\n"
    )
    st.divider()
    st.caption("Fontes: Yahoo Finance · FRED · fallback sintético automático")
    if st.button("🔄 Atualizar dados (limpar cache)"):
        st.cache_data.clear()
        st.rerun()

# --------------------------------------------------------------------------
# CONTEÚDO PRINCIPAL
# --------------------------------------------------------------------------
st.title(f"{APP_ICON} {APP_NAME}")
st.caption("Monitoramento institucional do mercado global de commodities — Energia · Metais · Agricultura")

quick_assets = [ENERGY_ASSETS[0], ENERGY_ASSETS[1], METALS_ASSETS[0], METALS_ASSETS[2], AGRI_ASSETS[0], AGRI_ASSETS[2]]
with st.spinner("Carregando cotações..."):
    price_data = load_price_history_bulk(quick_assets)

cols = st.columns(len(quick_assets))
for col, asset in zip(cols, quick_assets):
    pdat = price_data[asset.ticker]
    close = pdat.df["Close"]
    chg = metrics.pct_change_over(close, 1)
    with col:
        st.metric(f"{asset.name}", f"{close.iloc[-1]:.2f}", f"{chg:+.2%}" if chg is not None else None)
        if pdat.is_synthetic:
            st.caption("🔸 simulado")

st.divider()

col_left, col_right = st.columns([2, 1])
with col_left:
    st.subheader("Visão Geral — Retorno Acumulado (90 dias)")
    cum_returns = {a.name: metrics.cumulative_return_series(price_data[a.ticker].df["Close"]).tail(90)
                   for a in quick_assets}
    st.plotly_chart(charts.line_chart(cum_returns, title="", y_title="Retorno acumulado"), use_container_width=True)

with col_right:
    st.subheader("Sobre a Plataforma")
    st.markdown(
        "Este terminal reúne preços, analytics quantitativos, macroeconomia, risco e "
        "forecasting probabilístico para os principais mercados de commodities. "
        "Navegue pelos módulos na barra lateral.\n\n"
        "**Status de dados:** quando uma fonte ao vivo falha, o sistema faz fallback "
        "automático para dados simulados (badge 🔸), garantindo que o terminal nunca "
        "quebre em produção."
    )
    st.info(
        "Módulos avançados (NLP de notícias, ESG, Supply Chain/Sankey, geopolítica em mapa, "
        "modelos VAR/VECM/HMM/deep learning) estão no roadmap — ver README.",
        icon="🧭",
    )