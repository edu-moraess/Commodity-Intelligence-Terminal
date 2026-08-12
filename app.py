"""
Commodity Intelligence Terminal — Entry Point (Home)
=====================================================
Ponto de entrada único do app. Usa st.navigation/st.Page (API nativa do
Streamlit >= 1.36) para declarar explicitamente o nome e o ícone de cada
página.

"""

import streamlit as st
import pandas as pd

from config.settings import (
    APP_NAME, APP_ICON,
    ENERGY_ASSETS, METALS_ASSETS, AGRI_ASSETS, ALL_ASSETS, THEME,
)
from data.data_manager import load_price_history_bulk
from analytics import metrics
from charts import plotly_charts as charts
from utils.ticker_tape import render_ticker_tape
from utils.design_system import inject_institutional_css, render_sidebar_brand
from utils.data_status import render_price_data_disclosure

st.set_page_config(page_title=APP_NAME, page_icon=APP_ICON, layout="wide", initial_sidebar_state="expanded")

inject_institutional_css()

with st.spinner("Carregando fita de cotações..."):
    _ticker_price_data = load_price_history_bulk(ALL_ASSETS)
render_ticker_tape(_ticker_price_data, ALL_ASSETS)


def render_home() -> None:
    st.title(f"{APP_ICON} {APP_NAME}")
    st.caption(
        "Monitoramento institucional de commodities — Energia · Metais · Agricultura. "
        "As séries de preço são diárias; consulte a referência de cada ativo antes da análise."
    )

    quick_assets = [
        ENERGY_ASSETS[0],
        ENERGY_ASSETS[1],
        METALS_ASSETS[0],
        METALS_ASSETS[2],
        AGRI_ASSETS[0],
        AGRI_ASSETS[2],
    ]

    with st.spinner("Carregando cotações..."):
        price_data = load_price_history_bulk(quick_assets)

    render_price_data_disclosure(
        price_data,
        title="Fonte e data das séries da visão geral",
    )

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
                width="stretch",
            )
        else:
            st.warning("⚠️ Nenhum dado de retorno disponível para o período selecionado.")

    with col_right:
        st.subheader("Sobre a Plataforma")
        st.markdown(
            "Este terminal reúne preços, analytics quantitativos, macroeconomia, risco e "
            "forecasting probabilístico para os principais mercados de commodities. "
            "Navegue pelos módulos na barra lateral.\n\n"
            "**Status de dados:** os preços são séries diárias de fechamento. Quando um "
            "provedor não responde, a interface identifica explicitamente qualquer série "
            "simulada (badge 🔸); ela não representa uma cotação de mercado."
        )
        st.info(
            "Monte Carlo com cache de cenários, density backtest (CRPS/PIT) e "
            "health checks operacionais.",
            icon="🧭",
        )


# --------------------------------------------------------------------------
# SIDEBAR BRAND + HEALTH (visível em todas as páginas)
# --------------------------------------------------------------------------
render_sidebar_brand()

with st.sidebar:
    with st.expander("System health", expanded=False):
        try:
            from utils.health import run_health_checks
            report = run_health_checks()
            if report.overall_ok:
                st.success("Core systems OK")
            else:
                st.error("Core systems degraded")
            for c in report.checks:
                icon = "✅" if c.ok else "⚠️"
                lat = f" ({c.latency_ms:.0f} ms)" if c.latency_ms is not None else ""
                st.caption(f"{icon} **{c.name}**{lat} — {c.detail}")
        except Exception as exc:
            st.warning(f"Health check indisponível: {exc}")

# --------------------------------------------------------------------------
# NAVEGAÇÃO EXPLÍCITA
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
    st.Page("pages/10_📊_Portfolio.py", title="Portfolio", icon="📊"),
])
pg.run()
