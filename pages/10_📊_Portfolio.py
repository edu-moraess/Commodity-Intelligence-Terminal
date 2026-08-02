import streamlit as st
import pandas as pd
import numpy as np

from config.settings import ALL_ASSETS, RISK_FREE_RATE_ANNUAL, APP_NAME, THEME
from data.data_manager import load_price_history_bulk, build_price_panel
from analytics import portfolio as port
from analytics import portfolio_advanced as port_adv
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

# ---- Seletor de janela com YTD ----
col_w_opt, col_m, col_rf = st.columns(3)
with col_w_opt:
    window_option = st.selectbox(
        "Base de dados para otimização",
        ["Últimos 252 pregões", "Últimos 63 pregões", "YTD (desde jan/2026)", "Personalizado"],
        index=0,
    )
    if window_option == "YTD (desde jan/2026)":
        ytd_start = pd.Timestamp(pd.Timestamp.now().year, 1, 1)
        window = (pd.Timestamp.now() - ytd_start).days
        window = max(60, min(500, int(window * 0.7)))
        st.caption(f"Janela YTD: ~{window} pregões")
    elif window_option == "Personalizado":
        window = st.slider("Janela de estimação (pregões)", 60, 500, 252, step=20)
    elif window_option == "Últimos 63 pregões":
        window = 63
    else:  # 252
        window = 252

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

# ---- Configurações Avançadas ----
with st.expander("⚙️ Configurações Avançadas"):
    remove_outliers = st.checkbox("Remover outliers (|z|>3)", value=False)
    if remove_outliers:
        st.caption("Remove retornos com z-score > 3 para evitar distorções.")

# ---- Carregamento ----
with st.spinner("Carregando preços e otimizando..."):
    price_data = load_price_history_bulk(sel_assets)
    panel_raw = build_price_panel(price_data)

if panel_raw.empty or panel_raw.shape[1] < 2:
    st.error("Dados de preço insuficientes.")
    st.stop()

ticker_to_name = {a.ticker: a.name for a in sel_assets}
panel_raw = panel_raw.rename(columns={c: ticker_to_name.get(c, c) for c in panel_raw.columns})
panel_raw = panel_raw[[c for c in panel_raw.columns if c in sel_names]]

# ---- Aplica janela selecionada ----
panel = port_adv.get_window_data(panel_raw, window_option, custom_window=window)

# ---- Diagnóstico das séries ----
with st.expander("🔍 Diagnóstico das Séries (verifique anomalias)"):
    diag_df = port_adv.asset_diagnostics(panel_raw, window=window)
    st.dataframe(
        diag_df.style.format({
            "Retorno Janela (anual.)": "{:.2%}",
            "Retorno YTD": "{:.2%}",
            "Vol. Janela": "{:.2%}",
            "Max DD": "{:.2%}",
            "Skew": "{:.2f}",
            "Kurtosis": "{:.2f}",
        }, na_rep="-"),
        width="stretch"
    )
    anomalies = diag_df[diag_df["Anomalia"] != ""]
    if not anomalies.empty:
        st.warning(f"⚠️ Ativos com divergência significativa: {', '.join(anomalies.index.tolist())}")

# ---- Validação de outliers ----
if remove_outliers:
    rets = panel.pct_change().dropna()
    panel_clean = port_adv.validate_returns(rets)
    # Reconstrói painel a partir dos retornos limpos
    panel = (1 + panel_clean).cumprod() * panel.iloc[0]
    st.success("Outliers removidos com sucesso.")

# ---- Otimização ----
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
            line=dict(color=THEME["accent"]),
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

# ---- Comparação avançada de métodos ----
st.subheader("Comparação Avançada de Métodos")
if st.button("Comparar todos os métodos (com diagnóstico)", key="compare_adv"):
    with st.spinner("Otimizando todos os métodos..."):
        cmp_df = port_adv.compare_methods_advanced(
            panel, window=window, risk_free=rf, long_only=long_only
        )
    if cmp_df is not None and not cmp_df.empty:
        fmt = {
            "Retorno": "{:.2%}",
            "Vol": "{:.2%}",
            "Sharpe": "{:.2f}",
            "Max DD": "{:.2%}",
            "HHI (concentração)": "{:.3f}",
            "Turnover (vs Equal)": "{:.2%}",
        }
        st.dataframe(cmp_df.style.format(fmt, na_rep="—"), width="stretch")
    else:
        st.warning("Comparação sem resultados.")

st.divider()

# ---- Walk-Forward Backtest (out-of-sample) ----
with st.expander("📊 Backtest Out-of-Sample (Walk-Forward)"):
    st.caption("Rebalanceamento mensal (21 pregões) – avalia o desempenho real da estratégia fora da amostra.")
    if st.button("Rodar Walk-Forward Backtest", key="wf_btn"):
        from analytics.portfolio_advanced import walk_forward_backtest
        with st.spinner("Executando walk-forward (isso pode levar alguns segundos)..."):
            wf = walk_forward_backtest(
                panel, method=method, window=window,
                risk_free=rf, long_only=long_only, rebalance_freq=21
            )
        if wf["equity_curve"].empty:
            st.warning("Não foi possível executar o backtest.")
        else:
            st.subheader("Curva de Equity (Out-of-Sample)")
            st.plotly_chart(
                charts.line_chart(
                    {"Out-of-Sample (Walk-Forward)": wf["equity_curve"]},
                    title="Desempenho Real (fora da amostra)",
                    y_title="NAV"
                ),
                width="stretch"
            )
            s = wf["stats"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Retorno (anual.)", f"{s['expected_return']:.2%}")
            c2.metric("Vol. (anual.)", f"{s['volatility']:.2%}")
            c3.metric("Sharpe (out-of-sample)", f"{s['sharpe']:.2f}")
            c4.metric("Max DD", f"{s['max_drawdown']:.2%}")
            st.info(f"Rebalanceamentos realizados: {wf['n_rebalances']}")

st.divider()
st.caption(
    "Otimização mean-variance clássica (Markowitz), Risk Parity ERC, "
    "Max Diversification Ratio e Min CVaR (Rockafellar-Uryasev via cenários históricos). "
    "Pesos long-only por default. Resultados in-sample — não constituem recomendação de investimento."
)