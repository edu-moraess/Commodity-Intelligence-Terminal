import streamlit as st
import numpy as np
import pandas as pd

from config.settings import ALL_ASSETS, FORECAST_HORIZONS, APP_NAME
from data.data_manager import load_price_history
from forecasting import models as fc
from charts import plotly_charts as charts

st.title("📈 Forecast — Cenários Probabilísticos")
st.caption(
    "Ensemble de tendência + Monte Carlo avançado (Stationary Block Bootstrap, GBM, "
    "Jump Diffusion, Student-t, GARCH-MC). Cenários Base/Otimista/Pessimista, fan chart "
    "e probabilidades de rompimento."
)

asset_names = {a.name: a for a in ALL_ASSETS}
selected_name = st.selectbox("Ativo", list(asset_names.keys()))
asset = asset_names[selected_name]

# ---- Controles ----
col_h, col_n, col_method = st.columns(3)
with col_h:
    horizon_label = st.selectbox("Horizonte", list(FORECAST_HORIZONS.keys()), index=1)
with col_n:
    n_sims = st.select_slider("Simulações Monte Carlo", options=[500, 1000, 2000, 5000], value=2000)
with col_method:
    mc_methods = {
        "Stationary Block Bootstrap": "block_bootstrap",
        "GBM": "gbm",
        "Jump Diffusion (Merton)": "jump_diffusion",
        "Student-t": "student_t",
        "GARCH-MC": "garch_mc",
    }
    mc_label = st.selectbox("Método Monte Carlo", list(mc_methods.keys()), index=0)
    mc_method = mc_methods[mc_label]

# Controles avançados de reprodutibilidade
col_seed, col_block = st.columns(2)
with col_seed:
    seed = st.number_input("Seed (reprodutibilidade)", min_value=0, max_value=999999, value=42, step=1)
with col_block:
    block_size_input = st.number_input(
        "Block size (0 = automático via ACF)",
        min_value=0, max_value=30, value=0, step=1,
        help="Usado apenas no Stationary Block Bootstrap. 0 = estimado automaticamente."
    )
block_size = None if block_size_input == 0 else int(block_size_input)

horizon_days = FORECAST_HORIZONS[horizon_label]

# Modelos de tendência disponíveis
trend_options = ["Linear", "Ridge", "Lasso", "ElasticNet", "RandomForest"]
registry = getattr(fc, "MODEL_REGISTRY", None)
if registry is None and hasattr(fc, "_build_model_registry"):
    try:
        registry = fc._build_model_registry()
        trend_options = list(registry.keys())
    except Exception:
        pass
elif registry:
    trend_options = list(registry.keys())

col_trend, col_auto = st.columns(2)
with col_trend:
    trend_model = st.selectbox("Modelo de tendência (baseline)", trend_options, index=0)
with col_auto:
    auto_trend = st.checkbox("Selecionar melhor tendência via walk-forward", value=False)

with st.spinner("Carregando dados e simulando cenários..."):
    pdat = load_price_history(asset)
    close = pdat.df["Close"]

    try:
        scenario = fc.scenario_summary(
            close,
            horizon_days,
            n_sims=n_sims,
            method=mc_method,
            seed=int(seed),
            block_size=block_size,
        )
    except TypeError:
        # fallback de compatibilidade
        scenario = fc.scenario_summary(close, horizon_days, n_sims=n_sims, method=mc_method)

    if auto_trend and hasattr(fc, "select_best_trend_model"):
        best_name, ranking = fc.select_best_trend_model(close, train_window=120, n_folds=10)
        trend_model = best_name
        st.caption(f"Melhor tendência (walk-forward RMSE): **{best_name}**")
    trend_pred = fc.trend_forecast(close, horizon_days, model_name=trend_model)

if pdat.is_synthetic:
    st.warning("⚠️ Base de preço **simulada** — projeções abaixo herdam essa limitação.", icon="⚠️")

# ---- Métricas principais ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("Preço Atual", f"{scenario['preco_atual']:.2f}")
c2.metric(
    "Cenário Pessimista (P10)",
    f"{scenario['cenario_pessimista']:.2f}",
    f"{scenario['cenario_pessimista']/scenario['preco_atual']-1:+.1%}",
)
c3.metric(
    "Cenário Base (Mediana)",
    f"{scenario['cenario_base']:.2f}",
    f"{scenario['cenario_base']/scenario['preco_atual']-1:+.1%}",
)
c4.metric(
    "Cenário Otimista (P90)",
    f"{scenario['cenario_otimista']:.2f}",
    f"{scenario['cenario_otimista']/scenario['preco_atual']-1:+.1%}",
)

extra = st.columns(4)
if "expected_price" in scenario:
    extra[0].metric("Expected Price", f"{scenario['expected_price']:.2f}")
if "expected_return" in scenario:
    extra[1].metric("Expected Return", f"{scenario['expected_return']:.2%}")
if "prob_alta" in scenario:
    extra[2].metric("P(Alta)", f"{scenario['prob_alta']:.1%}")
if "prob_baixa" in scenario:
    extra[3].metric("P(Baixa)", f"{scenario['prob_baixa']:.1%}")

ic = scenario.get("intervalo_confianca_90", (np.nan, np.nan))
block_used = scenario.get("block_size_used")
st.caption(
    f"Método MC: **{scenario.get('method', mc_method)}** · "
    f"Seed: **{scenario.get('seed', seed)}** · "
    f"Block size: **{block_used if block_used is not None else '—'}** · "
    f"IC 90%: [{ic[0]:.2f}, {ic[1]:.2f}] · "
    f"Tendência: **{trend_model}**"
)

st.plotly_chart(
    charts.fan_chart(
        scenario["fan_chart"],
        scenario["preco_atual"],
        close.index[-1],
        title=f"{asset.name} — Fan Chart ({horizon_label})",
    ),
    width="stretch",
)

# ---- Probabilidades de rompimento ----
if any(k in scenario for k in ("prob_rompe_suporte", "prob_rompe_resistencia", "prob_acima_sma20")):
    st.subheader("Probabilidades de Rompimento")
    br1, br2, br3, br4 = st.columns(4)
    if "prob_rompe_suporte" in scenario:
        br1.metric("P(rompe suporte)", f"{scenario['prob_rompe_suporte']:.1%}")
        if "support" in scenario:
            br1.caption(f"Suporte ≈ {scenario['support']:.2f}")
    if "prob_rompe_resistencia" in scenario:
        br2.metric("P(rompe resistência)", f"{scenario['prob_rompe_resistencia']:.1%}")
        if "resistance" in scenario:
            br2.caption(f"Resistência ≈ {scenario['resistance']:.2f}")
    if "prob_acima_sma20" in scenario:
        br3.metric("P(acima SMA20)", f"{scenario['prob_acima_sma20']:.1%}")
    if "prob_acima_bb_upper" in scenario:
        br4.metric("P(acima BB upper)", f"{scenario['prob_acima_bb_upper']:.1%}")
    if "prob_abaixo_bb_lower" in scenario:
        st.caption(f"P(abaixo BB lower): {scenario['prob_abaixo_bb_lower']:.1%}")

# ---- Diagnósticos de distribuição ----
if "skewness" in scenario:
    st.subheader("Diagnóstico da Distribuição Simulada")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Skewness", f"{scenario['skewness']:.3f}")
    d2.metric("Kurtosis (excess)", f"{scenario['kurtosis']:.3f}")
    d3.metric("Jarque-Bera", f"{scenario.get('jarque_bera_stat', float('nan')):.2f}")
    jb_p = scenario.get("jarque_bera_pvalue", np.nan)
    d4.metric("JB p-valor", f"{jb_p:.4f}" if pd.notna(jb_p) else "—")
    if pd.notna(jb_p) and jb_p < 0.05:
        st.caption("JB rejeita normalidade (p < 0.05) — caudas/assimetria relevantes.")
    else:
        st.caption("JB não rejeita normalidade no nível 5%.")

st.divider()

st.subheader(f"Baseline de Tendência — {trend_model}")
st.plotly_chart(
    charts.line_chart(
        {"Histórico (180d)": close.tail(180), f"Projeção {trend_model}": trend_pred},
        title=f"Extrapolação de Tendência — {trend_model}",
    ),
    width="stretch",
)

st.divider()

st.subheader("Distribuição de Preços Finais (Monte Carlo)")
st.plotly_chart(
    charts.histogram_chart(
        scenario["final_prices_dist"],
        title=f"Distribuição do preço em {horizon_days} dias ({mc_label})",
        x_title="Preço simulado",
    ),
    width="stretch",
)

st.divider()
st.subheader("Comparação de Métodos Monte Carlo")
st.caption("Roda Stationary Block Bootstrap, GBM, Jump Diffusion, Student-t e GARCH-MC lado a lado.")

if st.button("Comparar métodos Monte Carlo", type="primary"):
    if hasattr(fc, "compare_monte_carlo_methods"):
        with st.spinner("Simulando todos os métodos..."):
            cmp = fc.compare_monte_carlo_methods(
                close, horizon_days=horizon_days, n_sims=min(n_sims, 1500), seed=int(seed)
            )
        if cmp is not None and not cmp.empty:
            fmt = {}
            for c in cmp.columns:
                if c == "Método":
                    continue
                if "P(" in c or "Return" in c or "Skew" in c or "Kurt" in c or "p-value" in c or "JB" in c:
                    fmt[c] = "{:.3f}"
                else:
                    fmt[c] = "{:.2f}"
            st.dataframe(cmp.style.format(fmt, na_rep="—"), width="stretch", hide_index=True)
        else:
            st.warning("Comparação não retornou resultados.")
    else:
        st.info("Função `compare_monte_carlo_methods` não disponível — atualize forecasting/models.py.")

st.info(
    f"**Metodologia ativa:** {mc_label}. "
    "Stationary Block Bootstrap (Politis & Romano) preserva autocorrelação e clusters de vol "
    "com block length automático via ACF. "
    "GBM assume retornos i.i.d. normais. "
    "Jump Diffusion (Merton) adiciona saltos poissonianos. "
    "Student-t captura caudas pesadas. "
    "GARCH-MC usa recursão real de volatilidade condicional. "
    "Nenhum método incorpora eventos geopolíticos ou de oferta/demanda fora do padrão histórico recente.",
    icon="ℹ️",
)
