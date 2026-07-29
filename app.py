"""
Commodity Intelligence Terminal — Entry Point (Home)
=====================================================
Ponto de entrada único do app. Usa st.navigation/st.Page (API nativa do
Streamlit >= 1.36) para declarar explicitamente o nome e o ícone de cada
página, em vez de deixar o Streamlit inferir a partir do nome do arquivo.

CHANGELOG v4.5.0 (fix de navegação):
- Antes, a lista de páginas era 100% automática (pasta pages/), o que
  fazia o item de entrada aparecer com o rótulo cru "app" (nome do
  arquivo app.py sem tratamento) — sem ícone, sem nome amigável,
  parecendo um item "extra"/duplicado ao lado dos outros módulos.
- Agora cada página é registrada explicitamente via st.Page(...) com
  título e ícone próprios. "app" não aparece mais em lugar nenhum.
- O bloco de branding do sidebar (título, botão "Atualizar dados", data)
  antes só era renderizado na Home, porque vivia dentro do script
  app.py e as outras páginas rodavam isoladas. Agora ele roda ANTES de
  st.navigation(...).run(), então aparece de forma consistente em
  TODAS as páginas.
- st.set_page_config() só pode ser chamado 1x por sessão — por isso foi
  removido do topo de cada arquivo em pages/ (ver nota lá).
"""

import datetime

import streamlit as st
import pandas as pd

from config.settings import APP_NAME, APP_ICON, ENERGY_ASSETS, METALS_ASSETS, AGRI_ASSETS, THEME
from data.data_manager import load_price_history_bulk
from analytics import metrics
from charts import plotly_charts as charts

st.set_page_config(page_title=APP_NAME, page_icon=APP_ICON, layout="wide", initial_sidebar_state="expanded")

# --------------------------------------------------------------------------
# TEMA ESCURO INSTITUCIONAL (CSS) — aplicado uma vez, vale para toda navegação
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
    div[data-testid="stMetric"] > div:first-child {{ font-size: 0.9rem !important; }}
    div[data-testid="stMetric"] > div:last-child {{ font-size: 1.5rem !important; font-weight: 600; }}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# PÁGINA HOME — vira uma função, registrada abaixo via st.Page(render_home)
# --------------------------------------------------------------------------
def render_home() -> None:
    st.title(f"{APP_ICON} {APP_NAME}")
    st.caption("Monitoramento institucional do mercado global de commodities — Energia · Metais · Agricultura")

    quick_assets = [
        ENERGY_ASSETS[0],   # Brent
        ENERGY_ASSETS[1],   # WTI
        METALS_ASSETS[0],   # Ouro
        METALS_ASSETS[2],   # Cobre
        AGRI_ASSETS[0],     # Soja
        AGRI_ASSETS[2],     # Trigo
    ]

    with st.spinner("Carregando cotações..."):
        price_data = load_price_history_bulk(quick_assets)

    cols = st.columns(len(quick_assets))
    for col, asset in zip(cols, quick_assets):
        pdat = price_data.get(asset.ticker)
        if pdat is None or pdat.df.empty:
            with col:
                st.metric(label=asset.name, value="N/D", delta=None)
            continue

        close = pdat.df["Close"]
        if not close.empty and pd.notna(close.iloc[-1]):
            last_price = close.iloc[-1]
            chg = metrics.pct_change_over(close, 1)
            display_price = f"{last_price:.2f}"
            display_delta = f"{chg:+.2%}" if chg is not None else None
        else:
            display_price = "N/D"
            display_delta = None

        with col:
            st.metric(label=asset.name, value=display_price, delta=display_delta)
            st.caption("🔸 simulado" if pdat.is_synthetic else "")

    st.divider()

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Visão Geral — Retorno Acumulado")
        period_days = st.selectbox("Período", [30, 60, 90, 180, 365], index=2, key="period_selector")

        cum_returns = {}
        for a in quick_assets:
            pdat = price_data.get(a.ticker)
            if pdat is None or pdat.df.empty:
                continue
            ser = pdat.df["Close"]
            if not ser.empty and len(ser) > 1:
                cum = metrics.cumulative_return_series(ser).tail(period_days)
                cum_returns[a.name] = cum

        any_synth = any(pdat.is_synthetic for pdat in price_data.values() if pdat is not None)
        if any_synth:
            st.caption("⚠️ Alguns ativos podem estar exibindo dados simulados (🔸)")

        if cum_returns:
            st.plotly_chart(
                charts.line_chart(cum_returns, title=f"Retorno Acumulado ({period_days} dias)", y_title="Retorno"),
                use_container_width=True,
            )
        else:
            st.warning("⚠️ Nenhum dado de retorno disponível para o período selecionado.")

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


# --------------------------------------------------------------------------
# SIDEBAR — branding persistente, roda em TODA página (antes de pg.run())
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"## {APP_ICON} {APP_NAME}")
    st.caption("Institutional Quant Research Platform")
    st.divider()
    st.caption("Fontes: Yahoo Finance · FRED · fallback sintético automático")
    if st.button("🔄 Atualizar dados (limpar cache)"):
        st.cache_data.clear()
        st.rerun()
    st.caption(f"📅 {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}")
    st.divider()

# --------------------------------------------------------------------------
# NAVEGAÇÃO EXPLÍCITA — cada página com título e ícone próprios
# --------------------------------------------------------------------------
pg = st.navigation([
    st.Page(render_home, title="Visão Geral", icon="🏠", default=True, url_path="home"),
    st.Page("pages/1_🌍_Dashboard_Global.py", title="Dashboard Global", icon="🌍"),
    st.Page("pages/2_🛢️_Energy.py", title="Energy Analytics", icon="🛢️"),
    st.Page("pages/3_⚙️_Metals.py", title="Metals Analytics", icon="⚙️"),
    st.Page("pages/4_🌾_Agriculture.py", title="Agriculture Analytics", icon="🌾"),
    st.Page("pages/5_🇧🇷_Brazil.py", title="Commodities Brasileiras", icon="🇧🇷"),
    st.Page("pages/6_🔗_Macro_Correlations.py", title="Macro & Correlações", icon="🔗"),
    st.Page("pages/7_⚠️_Risk_Analytics.py", title="Risk Analytics", icon="⚠️"),
    st.Page("pages/8_📈_Forecast.py", title="Forecast", icon="📈"),
    st.Page("pages/9_🧮_Quant_Research.py", title="Quant Research", icon="🧮"),
])
pg.run()