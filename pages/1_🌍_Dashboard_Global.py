import streamlit as st
import pandas as pd

from config.settings import ENERGY_ASSETS, METALS_ASSETS, AGRI_ASSETS, APP_NAME, RISK_FREE_RATE_ANNUAL
from data.data_manager import load_price_history_bulk
from analytics import metrics
from charts import plotly_charts as charts

st.set_page_config(page_title=f"Dashboard Global — {APP_NAME}", page_icon="🌍", layout="wide")

st.title("🌍 Dashboard Global de Commodities")
st.caption("Visão consolidada — Energia, Metais e Agricultura, com métricas de retorno, risco e tendência.")

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

# -------- KPIs de topo --------
gold = price_data.get("GC=F")
brent = price_data.get("BZ=F")
soy = price_data.get("ZS=F")
copper = price_data.get("HG=F")

kpi_cols = st.columns(4)
for col, pdat, label in zip(
    kpi_cols,
    [brent, gold, copper, soy],
    ["Brent (USD/bbl)", "Ouro (USD/oz)", "Cobre (USD/lb)", "Soja (USd/bu)"],
):
    if pdat is None:
        continue
    close = pdat.df["Close"]
    chg = metrics.pct_change_over(close, 1)
    with col:
        st.metric(label, f"{close.iloc[-1]:.2f}", f"{chg:+.2%}" if chg is not None else None)

st.divider()

# -------- Tabela mestre por setor --------
tabs = st.tabs(list(ALL_SECTORS.keys()) + ["Todos"])

def build_table(assets):
    rows = []
    for a in assets:
        pdat = price_data[a.ticker]
        close = pdat.df["Close"]
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
    return pd.DataFrame(rows).set_index("Ativo")


pct_cols = ["1D", "1S", "1M", "YTD", "Vol.Anual", "Max DD", "Momentum"]
fmt = {c: "{:.2%}" for c in pct_cols} | {"Último": "{:.2f}", "Sharpe": "{:.2f}", "Sortino": "{:.2f}", "Calmar": "{:.2f}"}

for tab, (sector_name, assets) in zip(tabs[:-1], ALL_SECTORS.items()):
    with tab:
        df = build_table(assets)
        st.dataframe(df.style.format(fmt), use_container_width=True, height=min(38 * (len(df) + 1) + 20, 400))

with tabs[-1]:
    df_all = build_table(ALL_ASSETS)
    st.dataframe(df_all.style.format(fmt), use_container_width=True, height=560)

st.divider()

# -------- Treemap de performance --------
st.subheader("Mapa de Performance (1 mês)")
labels, parents, values, colors = [], [], [], []
for sector_name, assets in ALL_SECTORS.items():
    labels.append(sector_name)
    parents.append("")
    values.append(1)  # peso neutro no nível de setor
    for a in assets:
        close = price_data[a.ticker].df["Close"]
        chg = metrics.pct_change_over(close, 21) or 0.0
        labels.append(a.name)
        parents.append(sector_name)
        values.append(abs(chg) + 0.01)

st.plotly_chart(charts.treemap_chart(labels, parents, values, title="Tamanho = |variação 1M| (ilustrativo)"),
                 use_container_width=True)

st.caption(
    "Fontes: Yahoo Finance (preços de futuros/proxies) com fallback sintético automático. "
    "Métricas: retornos, volatilidade anualizada (63d), Sharpe/Sortino/Calmar (252d, "
    f"taxa livre de risco = {RISK_FREE_RATE_ANNUAL:.2%} a.a.), máximo drawdown (252d) e momentum "
    "composto (1M/3M/6M/12M)."
)