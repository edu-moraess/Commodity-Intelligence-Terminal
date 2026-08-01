import streamlit as st
import pandas as pd
import numpy as np

from config.settings import ALL_ASSETS, MACRO_SERIES, APP_NAME
from data.data_manager import load_price_history_bulk, load_macro_series, build_price_panel
from analytics import correlation
from charts import plotly_charts as charts

st.title("🔗 Macroeconomia & Correlações")
st.caption(
    "Snapshot macro (FRED), correlação commodities × macro, beta rolante, "
    "lead-lag e PCA. Séries: DXY, yields, Fed Funds, VIX, CPI/PPI, INDPRO, M2, China CPI."
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
        "⚠️ Séries macro simuladas (sem `FRED_API_KEY` ou API indisponível): "
        + ", ".join(synth_names),
        icon="⚠️",
    )

st.subheader("Snapshot Macro — Últimos Valores")
snap_cols = st.columns(4)
items = list(macro_data.items())
for i, (name, series) in enumerate(items):
    col = snap_cols[i % 4]
    s = series.dropna()
    if s.empty:
        col.metric(name, "—")
        continue
    last = float(s.iloc[-1])
    if len(s) >= 22:
        prev = float(s.iloc[-22])
        if last < 50 and ("Yield" in name or "Funds" in name or "Unemployment" in name
                          or "Breakeven" in name or "VIX" in name):
            col.metric(name, f"{last:.2f}", f"{last - prev:+.2f}")
        else:
            delta = (last - prev) / prev if prev != 0 else 0.0
            col.metric(name, f"{last:,.2f}", f"{delta:+.1%}")
    else:
        col.metric(name, f"{last:,.2f}")

st.divider()

panel = build_price_panel(price_data)
if panel.empty:
    st.error("❌ Não foi possível carregar dados de preço.")
    st.stop()

ticker_to_name = {a.ticker: a.name for a in sel_assets}
panel = panel.rename(columns={c: ticker_to_name.get(c, c) for c in panel.columns})

macro_panel = pd.DataFrame(macro_data).ffill()
combined = panel.join(macro_panel, how="inner").dropna(how="all")

if combined.empty or combined.shape[1] < 2:
    st.error("❌ Dados insuficientes para correlações.")
    st.stop()

if combined.shape[0] < 30:
    st.warning(f"⚠️ Apenas {combined.shape[0]} observações — correlações podem ser instáveis.")

st.subheader("Heatmap de Correlação — Commodities × Macro")
window = st.slider("Janela (pregões)", 60, 500, 252, step=20, key="corr_window")
corr = correlation.correlation_matrix(combined, window=window)
if corr.empty:
    st.warning("⚠️ Matriz vazia.")
else:
    st.plotly_chart(
        charts.correlation_heatmap(corr, title=f"Correlação ({window} pregões)"),
        width="stretch",
    )

st.divider()

st.subheader("Correlação Rolante")
col1, col2 = st.columns(2)
with col1:
    asset_a = st.selectbox("Série A", combined.columns, index=0, key="roll_a")
with col2:
    asset_b = st.selectbox("Série B", combined.columns, index=min(1, len(combined.columns) - 1), key="roll_b")
roll_window = st.slider("Janela rolante (pregões)", 20, 250, 63, step=5, key="roll_w")
roll_corr = correlation.rolling_correlation(combined[asset_a], combined[asset_b], window=roll_window)
if roll_corr.empty:
    st.warning("⚠️ Dados insuficientes.")
else:
    st.plotly_chart(
        charts.line_chart({f"Corr({asset_a}, {asset_b})": roll_corr}, title="Correlação Rolante", y_title="ρ"),
        width="stretch",
    )

st.divider()

st.subheader("Beta Rolante — Commodity vs Fator Macro")
st.caption("β = Cov(r_asset, r_factor) / Var(r_factor).")

asset_names_only = [c for c in panel.columns if c in combined.columns]
macro_names_only = [c for c in macro_panel.columns if c in combined.columns]

if asset_names_only and macro_names_only:
    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        beta_asset = st.selectbox("Ativo", asset_names_only, key="beta_asset")
    with bc2:
        beta_factor = st.selectbox("Fator macro", macro_names_only, key="beta_factor")
    with bc3:
        beta_window = st.slider("Janela beta", 20, 250, 63, step=5, key="beta_w")

    if hasattr(correlation, "rolling_beta"):
        beta_series = correlation.rolling_beta(
            combined[beta_asset], combined[beta_factor], window=beta_window
        )
        if beta_series.empty:
            st.warning("Dados insuficientes para beta.")
        else:
            st.plotly_chart(
                charts.line_chart(
                    {f"β({beta_asset} | {beta_factor})": beta_series},
                    title="Beta Rolante", y_title="β",
                ),
                width="stretch",
            )
            st.caption(f"Beta atual: **{float(beta_series.iloc[-1]):.3f}**")
    else:
        st.info("Atualize analytics/correlation.py.")

st.divider()

st.subheader("Lead–Lag Correlation")
st.caption("Lag > 0 → fator B lidera ativo A. Lag < 0 → ativo A lidera fator B.")

if asset_names_only and macro_names_only and hasattr(correlation, "lead_lag_correlation"):
    lc1, lc2, lc3 = st.columns(3)
    with lc1:
        ll_asset = st.selectbox("Ativo (A)", asset_names_only, key="ll_a")
    with lc2:
        ll_factor = st.selectbox("Fator (B)", macro_names_only, key="ll_b")
    with lc3:
        max_lag = st.slider("Max lag (dias)", 5, 40, 20, key="ll_lag")

    ll_df = correlation.lead_lag_correlation(
        combined[ll_asset], combined[ll_factor], max_lag=max_lag
    )
    if ll_df.empty:
        st.warning("Dados insuficientes.")
    else:
        st.plotly_chart(
            charts.bar_chart(
                ll_df["lag"].astype(str).tolist(),
                ll_df["correlation"].tolist(),
                title=f"Lead–Lag: {ll_asset} × {ll_factor}",
                positive_negative=True,
            ),
            width="stretch",
        )
        best = ll_df.loc[ll_df["correlation"].abs().idxmax()]
        st.caption(f"Maior |ρ|: lag = **{int(best['lag'])}**, ρ = **{best['correlation']:.3f}**")

st.divider()

st.subheader("PCA — Componentes Principais dos Retornos")
pca_result = correlation.pca_components(panel, n_components=3)
if pca_result["explained_variance_ratio"]:
    exp_var = pca_result["explained_variance_ratio"]
    st.plotly_chart(
        charts.bar_chart(
            [f"PC{i+1}" for i in range(len(exp_var))], exp_var,
            title="Variância Explicada por Componente", positive_negative=False,
        ),
        width="stretch",
    )
    st.dataframe(pca_result["loadings"].style.format("{:.3f}"), width="stretch")
else:
    st.info("Selecione ≥ 2 ativos com histórico suficiente.")

st.divider()
st.caption(
    "Fontes: FRED. Configure FRED_API_KEY no secrets para dados ao vivo. "
    "PMI ISM e Baltic Dry requerem fonte paga."
)