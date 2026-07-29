import streamlit as st

from config.settings import ALL_ASSETS, FORECAST_HORIZONS, APP_NAME
from data.data_manager import load_price_history
from forecasting import models as fc
from charts import plotly_charts as charts

# NOTA v4.5.0: st.set_page_config() removido daqui — agora é chamado
# uma única vez em app.py, antes de st.navigation(...).run().

st.title("📈 Forecast — Cenários Probabilísticos")
st.caption(
    "Projeção via Monte Carlo (bootstrap de blocos dos retornos históricos) combinada com "
    "regressão de tendência log-linear. Cenários Base (mediana), Otimista (p90) e Pessimista "
    "(p10), com fan chart de intervalo de confiança."
)

asset_names = {a.name: a for a in ALL_ASSETS}
selected_name = st.selectbox("Ativo", list(asset_names.keys()))
asset = asset_names[selected_name]

col_h, col_n, col_m = st.columns(3)
with col_h:
    horizon_label = st.selectbox("Horizonte", list(FORECAST_HORIZONS.keys()), index=1)
with col_n:
    n_sims = st.select_slider("Simulações Monte Carlo", options=[500, 1000, 2000, 5000], value=2000)
with col_m:
    trend_model = st.selectbox("Modelo de tendência (baseline)", ["Linear", "Ridge", "Lasso"])

horizon_days = FORECAST_HORIZONS[horizon_label]

with st.spinner("Carregando dados e simulando cenários..."):
    pdat = load_price_history(asset)
    close = pdat.df["Close"]
    scenario = fc.scenario_summary(close, horizon_days, n_sims=n_sims)
    trend_pred = fc.trend_forecast(close, horizon_days, model_name=trend_model)

if pdat.is_synthetic:
    st.warning("⚠️ Base de preço **simulada** — projeções abaixo herdam essa limitação.", icon="⚠️")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Preço Atual", f"{scenario['preco_atual']:.2f}")
c2.metric("Cenário Pessimista (P10)", f"{scenario['cenario_pessimista']:.2f}",
          f"{scenario['cenario_pessimista']/scenario['preco_atual']-1:+.1%}")
c3.metric("Cenário Base (Mediana)", f"{scenario['cenario_base']:.2f}",
          f"{scenario['cenario_base']/scenario['preco_atual']-1:+.1%}")
c4.metric("Cenário Otimista (P90)", f"{scenario['cenario_otimista']:.2f}",
          f"{scenario['cenario_otimista']/scenario['preco_atual']-1:+.1%}")

st.caption(
    f"Probabilidade de alta ao final do horizonte: **{scenario['prob_alta']:.1%}** · "
    f"Intervalo de confiança 90%: [{scenario['intervalo_confianca_90'][0]:.2f}, "
    f"{scenario['intervalo_confianca_90'][1]:.2f}]"
)

st.plotly_chart(
    charts.fan_chart(scenario["fan_chart"], scenario["preco_atual"], close.index[-1],
                      title=f"{asset.name} — Fan Chart ({horizon_label})"),
    use_container_width=True,
)

st.divider()

st.subheader("Baseline de Tendência (Regressão)")
st.plotly_chart(
    charts.line_chart(
        {"Histórico (180d)": close.tail(180), f"Projeção {trend_model}": trend_pred},
        title=f"Extrapolação de Tendência — {trend_model}",
    ),
    use_container_width=True,
)

st.divider()

st.subheader("Distribuição de Preços Finais (Monte Carlo)")
st.plotly_chart(
    charts.histogram_chart(scenario["final_prices_dist"],
                            title=f"Distribuição do preço em {horizon_days} dias",
                            x_title="Preço simulado"),
    use_container_width=True,
)

st.info(
    "Metodologia: simulação de blocos (block bootstrap, tamanho 5) sobre os retornos diários "
    "dos últimos 504 pregões, preservando parcialmente a autocorrelação e os clusters de "
    "volatilidade observados historicamente — mais realista que um GBM i.i.d. puro. Não incorpora "
    "eventos geopolíticos ou de oferta/demanda fora do padrão histórico recente.",
    icon="ℹ️",
)