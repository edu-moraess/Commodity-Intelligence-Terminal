import streamlit as st
import pandas as pd
import numpy as np

from config.settings import ENERGY_ASSETS, METALS_ASSETS, AGRI_ASSETS, APP_NAME, RISK_FREE_RATE_ANNUAL
from data.data_manager import load_price_history_bulk
from analytics import metrics, signals
from charts import plotly_charts as charts
from utils.export import download_dataframe

# --------------------------------------------------------------------------
# CABEÇALHO
# --------------------------------------------------------------------------
st.title("🌍 Dashboard Global de Commodities")
st.caption(
    "Visão consolidada — Energia, Metais e Agricultura, com métricas de retorno, risco e tendência. "
    "Abaixo, você encontra a **metodologia** de cada indicador e sua interpretação prática."
)

with st.expander("📘 Sobre este Dashboard (Metodologia Geral)", expanded=False):
    st.markdown(r"""
    **Objetivo:** Fornecer uma visão panorâmica dos principais mercados de commodities.
    **Métricas:** Retornos (1D, 1S, 1M, YTD), Volatilidade Anualizada, Sharpe, Sortino,
    Máximo Drawdown, Calmar, Momentum Composto e Tendência.
    """)

st.divider()

# --------------------------------------------------------------------------
# CARREGAMENTO DE DADOS
# --------------------------------------------------------------------------
ALL_SECTORS = {"Energia": ENERGY_ASSETS, "Metais": METALS_ASSETS, "Agricultura": AGRI_ASSETS}
ALL_ASSETS = ENERGY_ASSETS + METALS_ASSETS + AGRI_ASSETS

with st.spinner("Carregando universo global de ativos..."):
    price_data = load_price_history_bulk(ALL_ASSETS)

n_synthetic = sum(1 for p in price_data.values() if p.is_synthetic)
if n_synthetic:
    st.warning(
        f"⚠️ {n_synthetic}/{len(price_data)} ativos exibindo **dados simulados** "
        "(fonte ao vivo indisponível neste ambiente). Badge 🔸 identifica cada caso.",
        icon="⚠️",
    )

# --------------------------------------------------------------------------
# KPIs DE TOPO (4 ativos principais)
# --------------------------------------------------------------------------
kpi_tickers = {"Brent (USD/bbl)": "BZ=F", "Ouro (USD/oz)": "GC=F",
               "Cobre (USD/lb)": "HG=F", "Soja (USd/bu)": "ZS=F"}

kpi_cols = st.columns(4)
for col, (label, ticker) in zip(kpi_cols, kpi_tickers.items()):
    pdat = price_data.get(ticker)
    if pdat is None or pdat.df.empty:
        with col:
            st.metric(label, "N/D", delta=None)
        continue
    close = pdat.df["Close"].dropna()
    if close.empty:
        with col:
            st.metric(label, "N/D", delta=None)
        continue
    last_price = close.iloc[-1]
    chg = metrics.pct_change_over(close, 1)
    display_price = f"{last_price:.2f}" if pd.notna(last_price) else "N/D"
    display_delta = f"{chg:+.2%}" if (chg is not None and pd.notna(chg)) else None
    with col:
        st.metric(label, display_price, display_delta)

st.divider()

# --------------------------------------------------------------------------
# TABELA MESTRE POR SETOR
# --------------------------------------------------------------------------
def build_table(assets):
    rows = []
    for a in assets:
        pdat = price_data.get(a.ticker)
        if pdat is None or pdat.df.empty:
            rows.append({
                "Ativo": a.name + " 🔸 (sem dados)", "Unidade": a.unit,
                "Último": None, "1D": None, "1S": None, "1M": None, "YTD": None,
                "Vol.Anual": None, "Sharpe": None, "Sortino": None,
                "Max DD": None, "Calmar": None, "Momentum": None, "Tendência": "N/D",
            })
            continue
        close = pdat.df["Close"].dropna()
        if len(close) < 2:
            rows.append({
                "Ativo": a.name + (" 🔸" if pdat.is_synthetic else ""),
                "Unidade": a.unit,
                "Último": close.iloc[-1] if not close.empty else None,
                "1D": None, "1S": None, "1M": None, "YTD": None,
                "Vol.Anual": None, "Sharpe": None, "Sortino": None,
                "Max DD": None, "Calmar": None, "Momentum": None, "Tendência": "N/D",
            })
            continue
        try:
            row = metrics.summary_row(close, risk_free_annual=RISK_FREE_RATE_ANNUAL)
            rows.append({
                "Ativo": a.name + (" 🔸" if pdat.is_synthetic else ""),
                "Unidade": a.unit,
                "Último": row["last_price"],
                "1D": row["chg_1d"], "1S": row["chg_1w"], "1M": row["chg_1m"], "YTD": row["chg_ytd"],
                "Vol.Anual": row["vol_annual"], "Sharpe": row["sharpe"], "Sortino": row["sortino"],
                "Max DD": row["max_drawdown"], "Calmar": row["calmar"],
                "Momentum": row["momentum"], "Tendência": row["trend"],
            })
        except Exception:
            rows.append({
                "Ativo": a.name + " 🔸 (erro)", "Unidade": a.unit,
                "Último": close.iloc[-1] if not close.empty else None,
                "1D": None, "1S": None, "1M": None, "YTD": None,
                "Vol.Anual": None, "Sharpe": None, "Sortino": None,
                "Max DD": None, "Calmar": None, "Momentum": None, "Tendência": "N/D",
            })
    return pd.DataFrame(rows).set_index("Ativo")

pct_cols = ["1D", "1S", "1M", "YTD", "Vol.Anual", "Max DD", "Momentum"]
fmt = {c: "{:.2%}" for c in pct_cols}
fmt.update({"Último": "{:.2f}", "Sharpe": "{:.2f}", "Sortino": "{:.2f}", "Calmar": "{:.2f}"})

tabs = st.tabs(list(ALL_SECTORS.keys()) + ["Todos"])
for tab, (sector_name, assets) in zip(tabs[:-1], ALL_SECTORS.items()):
    with tab:
        df = build_table(assets)
        st.dataframe(df.style.format(fmt, na_rep="-"), width='stretch',
                     height=min(38 * (len(df) + 1) + 20, 400))
with tabs[-1]:
    df_all = build_table(ALL_ASSETS)
    st.dataframe(df_all.style.format(fmt, na_rep="-"), width='stretch', height=560)
    download_dataframe(df_all, filename_stem="dashboard_global_todos_ativos")

with st.expander("📐 Como as métricas são calculadas? (Fórmulas)"):
    st.markdown(r"**Retornos:** Variação percentual simples. **Volatilidade Anualizada:** $\sigma \times \sqrt{252}$. **Sharpe:** $(\bar{R} - R_f)/\sigma$. **Sortino:** $(\bar{R} - R_f)/\sigma_{\text{down}}$. **Max DD:** maior queda acumulada. **Calmar:** retorno acumulado / |MDD|. **Momentum:** média dos retornos de 1,3,6,12 meses. **Tendência:** compara SMA(20) com SMA(100).")

st.divider()

# --------------------------------------------------------------------------
# TREEMAP DE PERFORMANCE (com proteção)
# --------------------------------------------------------------------------
st.subheader("🗺️ Mapa de Performance (1 mês)")
st.caption("Cada bloco representa um ativo; o tamanho é proporcional ao valor absoluto da variação no mês.")

labels, parents, values = [], [], []
for sector_name, assets in ALL_SECTORS.items():
    labels.append(sector_name)
    parents.append("")
    values.append(1)
    for a in assets:
        pdat = price_data.get(a.ticker)
        if pdat is None or pdat.df.empty:
            continue
        close = pdat.df["Close"].dropna()
        if len(close) < 2:
            continue
        chg = metrics.pct_change_over(close, 21) or 0.0
        labels.append(a.name)
        parents.append(sector_name)
        values.append(abs(chg) + 0.01)

if len(labels) > len(ALL_SECTORS):
    st.plotly_chart(
        charts.treemap_chart(labels, parents, values, title="Tamanho = |variação 1M| (ilustrativo)"),
        width='stretch'
    )
else:
    st.info("Não há dados suficientes para gerar o mapa de performance.", icon="ℹ️")

st.divider()

# --------------------------------------------------------------------------
# RISCO-RETORNO E PAINEL DE SINAIS (usando módulo signals)
# --------------------------------------------------------------------------
st.subheader("🎯 Risco vs. Retorno — Visão Consolidada")
st.caption("Cada bolha: X = volatilidade anualizada, Y = Sharpe Ratio, tamanho = |momentum|.")

with st.spinner("Calculando sinais..."):
    scatter_rows, signal_rows = [], []
    for sector_name, assets in ALL_SECTORS.items():
        for a in assets:
            pdat = price_data.get(a.ticker)
            if pdat is None or pdat.df.empty:
                continue
            close = pdat.df["Close"].dropna()
            if len(close) < 20:
                continue
            try:
                row = metrics.summary_row(close, risk_free_annual=RISK_FREE_RATE_ANNUAL)
                if row["last_price"] is None or pd.isna(row["vol_annual"]):
                    continue
                scatter_rows.append({
                    "name": a.name, "sector": sector_name,
                    "vol": row["vol_annual"], "sharpe": row["sharpe"], "momentum": row["momentum"],
                })
                # Sinais via módulo signals
                sig = signals.build_signal_row(close, momentum=row["momentum"])
                signal_rows.append({
                    "Ativo": a.name + (" 🔸" if pdat.is_synthetic else ""),
                    "Setor": sector_name,
                    "Regime de Vol.": sig["vol_regime"],
                    "Momentum": sig["momentum_label"],
                    "Z-Score Retorno (1d)": sig["return_zscore"],
                    "Movimento Atípico": "⚠️ Sim" if sig["extreme_move"] else "Não",
                })
            except Exception:
                continue

scatter_df = pd.DataFrame(scatter_rows)
if not scatter_df.empty:
    st.plotly_chart(charts.risk_return_scatter(scatter_df), width='stretch')
else:
    st.info("Dados insuficientes para o gráfico de risco-retorno.", icon="ℹ️")

st.divider()

st.subheader("🚨 Painel de Sinais Consolidado")
st.caption("Leitura rápida de regime de volatilidade e movimentos atípicos.")

if signal_rows:
    signal_df = pd.DataFrame(signal_rows).set_index("Ativo")
    n_alerts = (signal_df["Movimento Atípico"] == "⚠️ Sim").sum()
    n_high_vol = signal_df["Regime de Vol."].str.contains("Elevada").sum()

    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Ativos Monitorados", len(signal_df))
    sc2.metric("Em Volatilidade Elevada", int(n_high_vol))
    sc3.metric("Movimentos Atípicos Hoje", int(n_alerts))

    filter_option = st.radio("Filtrar", ["Todos", "Só Vol. Elevada", "Só Movimentos Atípicos"],
                             horizontal=True, key="signal_filter")
    display_df = signal_df.copy()
    if filter_option == "Só Vol. Elevada":
        display_df = display_df[display_df["Regime de Vol."].str.contains("Elevada")]
    elif filter_option == "Só Movimentos Atípicos":
        display_df = display_df[display_df["Movimento Atípico"] == "⚠️ Sim"]

    st.dataframe(
        display_df.style.format({"Z-Score Retorno (1d)": "{:.2f}"}, na_rep="-"),
        width='stretch',
        height=min(38 * (len(display_df) + 1) + 20, 500),
    )
else:
    st.info("Nenhum sinal disponível (dados insuficientes).", icon="ℹ️")

with st.expander("📐 Metodologia do Painel de Sinais"):
    st.markdown(r"""
    **Regime de Volatilidade:** percentil da vol. realizada (21d) vs. distribuição histórica (252d). ≥80% = elevada; ≤20% = baixa.
    **Z-Score do Retorno:** normalização do retorno diário pelos últimos 63 dias. |z|>2 = movimento atípico.
    """)

st.divider()
st.caption(
    f"Fontes: Yahoo Finance · fallback sintético. Taxa livre de risco = {RISK_FREE_RATE_ANNUAL:.2%} a.a."
)
with st.expander("📚 Referências Acadêmicas"):
    st.markdown("Sharpe (1966), Sortino & Price (1994), Campbell et al. (1997).")