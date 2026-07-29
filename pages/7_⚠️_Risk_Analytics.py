import streamlit as st
import pandas as pd

from config.settings import ALL_ASSETS, APP_NAME
from data.data_manager import load_price_history
from analytics import risk, metrics
from charts import plotly_charts as charts

st.set_page_config(page_title=f"Risk Analytics — {APP_NAME}", page_icon="⚠️", layout="wide")

st.title("⚠️ Risk Analytics")
st.caption("VaR histórico e paramétrico, CVaR/Expected Shortfall, stress test e métricas de risco ajustado.")

asset_names = {a.name: a for a in ALL_ASSETS}
selected_name = st.selectbox("Ativo", list(asset_names.keys()))
asset = asset_names[selected_name]

col_conf, col_window = st.columns(2)
with col_conf:
    confidence = st.select_slider("Nível de confiança", options=[0.90, 0.95, 0.975, 0.99], value=0.95)
with col_window:
    window = st.slider("Janela histórica (pregões)", 60, 500, 252, step=20)

with st.spinner("Carregando dados..."):
    pdat = load_price_history(asset)

if pdat.is_synthetic:
    st.warning("⚠️ Exibindo **dados simulados** — fonte ao vivo indisponível neste ambiente.", icon="⚠️")

close = pdat.df["Close"]

# -------- VaR / CVaR --------
r = risk.risk_summary(close, confidence=confidence, window=window)
c1, c2, c3 = st.columns(3)
c1.metric(f"VaR Histórico ({confidence:.1%})", f"{r['var_historico']:.2%}")
c2.metric(f"VaR Paramétrico ({confidence:.1%})", f"{r['var_parametrico']:.2%}")
c3.metric("CVaR / Expected Shortfall", f"{r['cvar']:.2%}")

st.caption(
    f"Interpretação: com {confidence:.0%} de confiança, a perda diária não deve exceder "
    f"**{r['var_historico']:.2%}** do valor posicionado (janela de {window} pregões). O CVaR "
    f"informa a perda média condicional nos cenários que ultrapassam esse limite."
)

st.divider()

# -------- Métricas de risco ajustado --------
st.subheader("Risco Ajustado ao Retorno")
row = metrics.summary_row(close, window=window) if False else metrics.summary_row(close)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Volatilidade Anualizada", f"{metrics.annualized_volatility(close, window=window):.2%}")
m2.metric("Sharpe Ratio (252d)", f"{metrics.sharpe_ratio(close, window=252):.2f}")
m3.metric("Sortino Ratio (252d)", f"{metrics.sortino_ratio(close, window=252):.2f}")
m4.metric("Máximo Drawdown (252d)", f"{metrics.max_drawdown(close.tail(252)):.2%}")

st.divider()

# -------- Stress Test --------
st.subheader("Stress Test — Choques Instantâneos")
custom_shocks = st.text_input("Choques (%), separados por vírgula", "-30,-20,-10,-5,5,10,20")
try:
    shocks = [float(x.strip()) / 100 for x in custom_shocks.split(",") if x.strip()]
except ValueError:
    shocks = None
    st.error("Formato inválido — use números separados por vírgula, ex: -30,-10,10,20")

if shocks:
    stress_df = risk.stress_test(close, shocks_pct=shocks)
    st.dataframe(stress_df, use_container_width=True, hide_index=True)
    st.plotly_chart(
        charts.bar_chart(stress_df["choque"].tolist(), stress_df["variacao_absoluta"].tolist(),
                          title="Impacto Absoluto por Cenário de Choque"),
        use_container_width=True,
    )

st.divider()

# -------- Distribuição de retornos --------
st.subheader("Distribuição de Retornos Diários")
rets = metrics.daily_returns(close).tail(window)
st.plotly_chart(charts.histogram_chart(rets.values, title="Histograma de Retornos", x_title="Retorno diário"),
                 use_container_width=True)