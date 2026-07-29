"""
Commodity Intelligence Terminal — Entry Point
=================================================
Página inicial com KPIs, gráfico de retorno acumulado.
A navegação é feita automaticamente pelo Streamlit via pasta pages/.
"""

import streamlit as st
import pandas as pd
import datetime

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
    /* Ajuste para cards não truncarem números */
    div[data-testid="stMetric"] > div:first-child {{ font-size: 0.9rem !important; }}
    div[data-testid="stMetric"] > div:last-child {{ font-size: 1.5rem !important; font-weight: 600; }}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# SIDEBAR – Personalização (sem lista de módulos)
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"## {APP_ICON} {APP_NAME}")
    st.caption("Institutional Quant Research Platform")
    st.divider()
    # NÃO CRIE UMA LISTA DE MÓDULOS AQUI! O Streamlit já a cria automaticamente.
    # Basta adicionar os controles extras:
    st.caption("Fontes: Yahoo Finance · FRED · fallback sintético automático")
    if st.button("🔄 Atualizar dados (limpar cache)"):
        st.cache_data.clear()
        st.rerun()
    st.caption(f"📅 {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}")

# --------------------------------------------------------------------------
# CONTEÚDO PRINCIPAL
# --------------------------------------------------------------------------
st.title(f"{APP_ICON} {APP_NAME}")
st.caption("Monitoramento institucional do mercado global de commodities — Energia · Metais · Agricultura")

# Ativos para os cards rápidos (Brent, WTI, Ouro, Cobre, Soja, Trigo)
quick_assets = [
    ENERGY_ASSETS[0],  # Brent
    ENERGY_ASSETS[1],  # WTI
    METALS_ASSETS[0],  # Ouro
    METALS_ASSETS[2],  # Cobre
    AGRI_ASSETS[0],    # Soja
    AGRI_ASSETS[2]     # Trigo
]

with st.spinner("Carregando cotações..."):
    price_data = load_price_history_bulk(quick_assets)

# Exibe cards com tratamento de NaN e formatação melhorada
cols = st.columns(len(quick_assets))
for col, asset in zip(cols, quick_assets):
    pdat = price_data[asset.ticker]
    close = pdat.df["Close"]
    
    if not close.empty and pd.notna(close.iloc[-1]):
        last_price = close.iloc[-1]
        chg = metrics.pct_change_over(close, 1)
        # Formata com 2 decimais para valores pequenos, ou 2 decimais para grandes
        if abs(last_price) < 100:
            display_price = f"{last_price:.2f}"
        else:
            display_price = f"{last_price:.2f}"
        display_delta = f"{chg:+.2%}" if pd.notna(chg) else None
    else:
        display_price = "N/D"
        display_delta = None
    
    with col:
        st.metric(
            label=f"{asset.name}",
            value=display_price,
            delta=display_delta,
        )
        # Exibe badge de simulação, se for o caso
        if pdat.is_synthetic:
            st.caption("🔸 simulado")
        else:
            st.caption("")  # placeholder para alinhamento

st.divider()

# --------------------------------------------------------------------------
# GRÁFICO DE RETORNO ACUMULADO
# --------------------------------------------------------------------------
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Visão Geral — Retorno Acumulado")
    
    period_days = st.selectbox("Período", [30, 60, 90, 180, 365], index=2, key="period_selector")
    
    cum_returns = {}
    for a in quick_assets:
        ser = price_data[a.ticker].df["Close"]
        if not ser.empty and len(ser) > 1:
            cum = metrics.cumulative_return_series(ser).tail(period_days)
            cum_returns[a.name] = cum
        else:
            cum_returns[a.name] = pd.Series(dtype=float)
    
    # Aviso se algum ativo estiver em fallback
    any_synth = any(pdat.is_synthetic for pdat in price_data.values())
    if any_synth:
        st.caption("⚠️ Alguns ativos podem estar exibindo dados simulados (🔸)")
    
    st.plotly_chart(
        charts.line_chart(cum_returns, title=f"Retorno Acumulado ({period_days} dias)", y_title="Retorno"),
        use_container_width=True,
    )

# --------------------------------------------------------------------------
# SOBRE A PLATAFORMA
# --------------------------------------------------------------------------
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