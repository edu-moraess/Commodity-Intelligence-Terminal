import streamlit as st
import pandas as pd

from config.settings import ALL_ASSETS, MACRO_SERIES, APP_NAME
from data.data_manager import load_price_history_bulk, load_macro_series, build_price_panel
from analytics import correlation
from charts import plotly_charts as charts

st.set_page_config(page_title=f"Macro — {APP_NAME}", page_icon="🔗", layout="wide")

st.title("🔗 Macroeconomia & Correlações")
st.caption(
    "Correlaciona o universo de commodities com DXY, Treasury 10Y, Fed Funds, CPI, PPI e "
    "produção industrial dos EUA. PMI (ISM) e Baltic Dry Index não têm série pública gratuita "
    "no FRED — requer integração paga (Trading Economics / IHS Markit)."
)

sel_assets_names = st.multiselect(
    "Ativos para correlacionar",
    [a.name for a in ALL_ASSETS],
    default=[a.name for a in ALL_ASSETS[:6]],
)
sel_assets = [a for a in ALL_ASSETS if a.name in sel_assets_names]

if not sel_assets:
    st.info("Selecione ao menos 2 ativos para gerar as análises.")
    st.stop()

with st.spinner("Carregando preços e séries macro..."):
    price_data = load_price_history_bulk(sel_assets)
    macro_data = {}
    macro_synthetic_flags = {}
    for key, meta in MACRO_SERIES.items():
        series, is_synth = load_macro_series(key, meta["code"])
        macro_data[meta["name"]] = series
        macro_synthetic_flags[meta["name"]] = is_synth

if any(macro_synthetic_flags.values()):
    synth_names = [k for k, v in macro_synthetic_flags.items() if v]
    st.warning(
        "⚠️ Séries macro simuladas (sem `FRED_API_KEY` configurada ou API indisponível): "
        + ", ".join(synth_names),
        icon="⚠️",
    )

# -------- Painel combinado --------
panel = build_price_panel(price_data)
panel.columns = [a.name for a in sel_assets if a.ticker in panel.columns]

macro_panel = pd.DataFrame(macro_data).ffill()

# 🛡️ CORREÇÃO: Remove o fuso horário dos índices para evitar o TypeError
# "Cannot join tz-naive with tz-aware DatetimeIndex"
if panel.index.tz is not None:
    panel.index = panel.index.tz_localize(None)
if macro_panel.index.tz is not None:
    macro_panel.index = macro_panel.index.tz_localize(None)

combined = panel.join(macro_panel, how="inner").dropna(how="all")

st.subheader("Heatmap de Correlação — Commodities × Macro")
window = st.slider("Janela (pregões)", 60, 500, 252, step=20)
corr = correlation.correlation_matrix(combined, window=window)
st.plotly_chart(charts.correlation_heatmap(corr, title=f"Correlação ({window} pregões)"), use_container_width=True)

st.divider()

st.subheader("Correlação Rolante")
col1, col2 = st.columns(2)
with col1:
    asset_a = st.selectbox("Série A", combined.columns, index=0)
with col2:
    asset_b = st.selectbox("Série B", combined.columns, index=min(1, len(combined.columns) - 1))
roll_window = st.slider("Janela rolante (pregões)", 20, 250, 63, step=5)

roll_corr = correlation.rolling_correlation(combined[asset_a], combined[asset_b], window=roll_window)
st.plotly_chart(
    charts.line_chart({f"Corr({asset_a}, {asset_b})": roll_corr}, title="Correlação Rolante", y_title="ρ"),
    use_container_width=True,
)

st.divider()

st.subheader("PCA — Componentes Principais dos Retornos")
pca_result = correlation.pca_components(panel, n_components=3)
if pca_result["explained_variance_ratio"]:
    exp_var = pca_result["explained_variance_ratio"]
    st.plotly_chart(
        charts.bar_chart([f"PC{i+1}" for i in range(len(exp_var))], exp_var,
                          title="Variância Explicada por Componente", positive_negative=False),
        use_container_width=True,
    )
    st.dataframe(pca_result["loadings"].style.format("{:.3f}"), use_container_width=True)
else:
    st.info("Selecione ao menos 2 ativos com histórico suficiente para calcular PCA.")