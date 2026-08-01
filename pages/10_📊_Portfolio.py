import streamlit as st
import pandas as pd
import numpy as np

from config.settings import ALL_ASSETS, RISK_FREE_RATE_ANNUAL, APP_NAME
from data.data_manager import load_price_history_bulk, build_price_panel
from analytics import portfolio as port
from charts import plotly_charts as charts

st.title("📊 Portfolio Optimization")
st.caption(
    "Markowitz (Min Variance / Max Sharpe), Risk Parity, Max Diversification, "
    "Min CVaR e Equal Weight. Fronteira eficiente e contribuições de risco."
)

# ---- Controles ----
sel_names = st.multiselect(
    "Universo de ativos",
    [a.name for a in ALL_ASSETS],
    default=[a.name for a in ALL_ASSETS[:6]],
)
sel_assets = [a for a in ALL_ASSETS if a.name in sel_names]

if len(sel_assets) < 2:
    st.info("Selecione ao menos 2 ativos.")
    st.stop()

col_m, col_w, col_rf = st.columns(3)
with col_m:
    method_labels = {
        "Max Sharpe": "max_sharpe",
        "Min Variance": "min_variance",
        "Risk Parity (ERC)": "risk_parity",
        "Max Diversification": "max_diversification",
        "Min CVaR (95%)": "min_cvar",
        "Equal Weight": "equal_weight",
    }
    method_label = st.selectbox("Método", list(method_labels.keys()), index=0)
    method = method_labels[method_label]
with col_w:
    window = st.slider("Janela de estimação (pregões)", 60, 500, 252, step=20)
with col_rf:
    rf = st.number_input(
        "Risk-free anual",
        min_value=0.0,
        max_value=0.20,
        value=float(RISK_FREE_RATE_ANNUAL),
        step=0.005,
        format="%.3f",
    )

long_only = st.checkbox("Long-only (sem short)", value=True)

with st.spinner("Carregando preços e otimizando..."):
    price_data = load_price_history_bulk(sel_assets)
    panel = build_price_panel(price_data)

if panel.empty or panel.shape[1] < 2:
    st.error("Dados de preço insuficientes.")
    st.stop()

ticker_to_name = {a.ticker: a.name for a in sel_assets}
panel = panel.rename(columns={c: ticker_to_name.get(c, c) for c in panel.columns})
panel = panel[[c for c in panel.columns if c in sel_names]]

try:
    result = port.optimize_portfolio(
        panel, method=method, window=window, risk_free=rf, long_only=long_only
    )
except Exception as exc:
    st.error(f"Falha na otimização: {exc}")
    st.stop()

# ---- Métricas ----
s = result["stats"]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Retorno Esperado (anual.)", f"{s['expected_return']:.2%}")
m2.metric("Volatilidade (anual.)", f"{s['volatility']:.2%}")
m3.metric("Sharpe", f"{s['sharpe']:.2f}")
m4.metric("Max DD (in-sample)", f"{result['max_drawdown']:.2%}")

st.caption(
    f"Método: **{method_label}** · Janela: {window} pregões · N ativos: {result['n_assets']}"
)

st.divider()

# ---- Pesos ----
st.subheader("Alocação Ótima")
weights = result["weights"].sort_values(ascending=False)
w_df = pd.DataFrame({
    "Ativo": weights.index,
    "Peso": weights.values,
    "Retorno Anual. Estimado": result["mu"].reindex(weights.index).values,
    "Vol Anual. Estimada": result["vol_asset"].reindex(weights.index).values,
    "Contrib. Risco": result["risk_contributions"].reindex(weights.index).values,
})
st.dataframe(
    w_df.style.format({
        "Peso": "{:.1%}",
        "Retorno Anual. Estimado": "{:.2%}",
        "Vol Anual. Estimada": "{:.2%}",
        "Contrib. Risco": "{:.1%}",
    }),
    width="stretch",
    hide_index=True,
)

c_bar1, c_bar2 = st.columns(2)
with c_bar1:
    st.plotly_chart(
        charts.bar_chart(
            weights.index.tolist(),
            weights.values.tolist(),
            title="Pesos do Portfólio",
            positive_negative=False,
        ),
        width="stretch",
    )
with c_bar2:
    rc = result["risk_contributions"].sort_values(ascending=False)
    st.plotly_chart(
        charts.bar_chart(
            rc.index.tolist(),
            rc.values.tolist(),
            title="Contribuição ao Risco",
            positive_negative=False,
        ),
        width="stretch",
    )

st.divider()

# ---- Equity curve ----
st.subheader("Curva de Equity (in-sample, pesos fixos)")
eq = result["equity_curve"]
if not eq.empty:
    st.plotly_chart(
        charts.line_chart(
            {"Portfolio": eq},
            title="Equity Curve (rebased 1.0)",
            y_title="NAV",
        ),
        width="stretch",
    )

st.divider()

# ---- Fronteira eficiente ----
st.subheader("Fronteira Eficiente")
if st.button("Calcular fronteira eficiente", type="primary"):
    with st.spinner("Otimizando grade de retornos-alvo..."):
        rets = port._returns_matrix(panel, window=window)
        mu, cov = port._cov_mean(rets)
        frontier = port.efficient_frontier(mu, cov, n_points=25, long_only=long_only)
    if frontier.empty:
        st.warning("Não foi possível traçar a fronteira.")
    else:
        method_points = []
        for lab, mid in method_labels.items():
            try:
                r = port.optimize_portfolio(
                    panel, method=mid, window=window, risk_free=rf, long_only=long_only
                )
                method_points.append({
                    "method": lab,
                    "vol": r["stats"]["volatility"],
                    "ret": r["stats"]["expected_return"],
                })
            except Exception:
                continue

        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=frontier["volatility"],
            y=frontier["return"],
            mode="lines+markers",
            name="Fronteira",
            line=dict(color="#3fb1ce"),
        ))
        for pt in method_points:
            fig.add_trace(go.Scatter(
                x=[pt["vol"]],
                y=[pt["ret"]],
                mode="markers+text",
                name=pt["method"],
                text=[pt["method"]],
                textposition="top center",
                marker=dict(size=12),
            ))
        fig.update_layout(
            title="Fronteira Eficiente (anualizada)",
            xaxis_title="Volatilidade",
            yaxis_title="Retorno Esperado",
            template="plotly_dark",
            height=480,
        )
        st.plotly_chart(fig, width="stretch")

st.divider()

# ---- Comparação de métodos ----
st.subheader("Comparação de Métodos")
if st.button("Comparar todos os métodos"):
    with st.spinner("Otimizando todos os métodos..."):
        cmp = port.compare_methods(panel, window=window, risk_free=rf)
    if cmp is not None and not cmp.empty:
        fmt = {}
        for c in cmp.columns:
            if c in ("Método", "Erro", "N ativos > 1%"):
                continue
            fmt[c] = "{:.2%}" if ("DD" in c or "Retorno" in c or "Vol" in c) else "{:.2f}"
        st.dataframe(cmp.style.format(fmt, na_rep="—"), width="stretch", hide_index=True)
    else:
        st.warning("Comparação sem resultados.")

st.divider()
st.caption(
    "Otimização mean-variance clássica (Markowitz), Risk Parity ERC, "
    "Max Diversification Ratio e Min CVaR (Rockafellar-Uryasev via cenários históricos). "
    "Pesos long-only por default. Resultados in-sample — não constituem recomendação de investimento."
)