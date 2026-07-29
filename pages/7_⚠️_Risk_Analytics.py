import streamlit as st
import pandas as pd
import numpy as np

from config.settings import ALL_ASSETS, APP_NAME
from data.data_manager import load_price_history
from analytics import risk, metrics
from charts import plotly_charts as charts

st.set_page_config(page_title=f"Risk Analytics — {APP_NAME}", page_icon="⚠️", layout="wide")

st.title("⚠️ Risk Analytics")
st.caption(
    "Módulo de análise de risco com VaR, CVaR, métricas ajustadas, stress test e distribuição de retornos. "
    "Todas as métricas são acompanhadas de suas fórmulas e interpretações."
)

# --------------------------------------------------------------------------
# SIDEBAR COM METODOLOGIA GERAL
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📘 Metodologias")
    with st.expander("📖 VaR Histórico"):
        st.markdown("""
        **Definição:** O VaR (Value at Risk) histórico é o percentil da distribuição empírica dos retornos passados.
        
        **Fórmula:**
        $$
        \\text{VaR}_{\\text{hist}} = -\\text{Percentil}(R, 1 - \\alpha)
        $$
        onde $\\alpha$ é o nível de confiança (ex: 0.95) e $R$ são os retornos diários.
        
        **Interpretação:** Com $\\alpha$ de confiança, a perda diária não deve superar o VaR.
        """)
    
    with st.expander("📖 VaR Paramétrico"):
        st.markdown("""
        **Definição:** Assume que os retornos seguem uma distribuição normal.
        
        **Fórmula:**
        $$
        \\text{VaR}_{\\text{param}} = -\\left( \\mu + \\sigma \\cdot z_{\\alpha} \\right)
        $$
        onde $\\mu$ é a média dos retornos, $\\sigma$ o desvio padrão, e $z_{\\alpha}$ o quantil da normal padrão.
        
        **Limitação:** Subestima riscos de cauda (eventos extremos).
        """)
    
    with st.expander("📖 CVaR (Expected Shortfall)"):
        st.markdown("""
        **Definição:** Média das perdas que excedem o VaR.
        
        **Fórmula (contínua):**
        $$
        \\text{CVaR} = -\\mathbb{E}[R \\mid R < -\\text{VaR}]
        $$
        
        **Interpretação:** Em cenários de estresse, a perda média esperada é o CVaR.
        """)
    
    with st.expander("📖 Sharpe Ratio"):
        st.markdown("""
        **Definição:** Mede o retorno ajustado ao risco, considerando a volatilidade total.
        
        **Fórmula:**
        $$
        \\text{Sharpe} = \\frac{\\bar{R} - R_f}{\\sigma(R)}
        $$
        onde $\\bar{R}$ é o retorno médio, $R_f$ a taxa livre de risco, e $\\sigma$ o desvio padrão.
        
        **Referência:** Sharpe (1966) - "Mutual Fund Performance".
        """)
    
    with st.expander("📖 Sortino Ratio"):
        st.markdown("""
        **Definição:** Similar ao Sharpe, mas penaliza apenas a volatilidade negativa (downside risk).
        
        **Fórmula:**
        $$
        \\text{Sortino} = \\frac{\\bar{R} - R_f}{\\sigma_{\\text{down}}}
        $$
        onde $\\sigma_{\\text{down}}$ é o desvio padrão dos retornos negativos.
        
        **Vantagem:** Mais adequado para investidores avessos a perdas.
        """)
    
    with st.expander("📖 Max Drawdown"):
        st.markdown("""
        **Definição:** Maior queda acumulada do preço de um pico a um vale, antes de um novo pico.
        
        **Fórmula:**
        $$
        \\text{MDD} = \\max_{t} \\left( \\frac{\\max_{s \\leq t} P_s - P_t}{\\max_{s \\leq t} P_s} \\right)
        $$
        
        **Interpretação:** Representa a pior perda histórica em termos de drawdown.
        """)

st.divider()

# --------------------------------------------------------------------------
# SELEÇÃO DE ATIVO E PARÂMETROS
# --------------------------------------------------------------------------
asset_names = {a.name: a for a in ALL_ASSETS}
selected_name = st.selectbox("📌 Selecione o Ativo", list(asset_names.keys()))
asset = asset_names[selected_name]

col_conf, col_window = st.columns(2)
with col_conf:
    confidence = st.select_slider("🎯 Nível de confiança", options=[0.90, 0.95, 0.975, 0.99], value=0.95)
with col_window:
    window = st.slider("📆 Janela histórica (pregões)", 60, 500, 252, step=20)

with st.spinner("Carregando dados..."):
    pdat = load_price_history(asset)

if pdat.is_synthetic:
    st.warning("⚠️ Exibindo **dados simulados** — fonte ao vivo indisponível neste ambiente.", icon="⚠️")

close = pdat.df["Close"]

# --------------------------------------------------------------------------
# V A R  E  C V A R
# --------------------------------------------------------------------------
r = risk.risk_summary(close, confidence=confidence, window=window)

st.subheader("📉 Value at Risk (VaR) e Expected Shortfall (CVaR)")

c1, c2, c3 = st.columns(3)
c1.metric(
    label=f"VaR Histórico ({confidence:.0%})",
    value=f"{r['var_historico']:.2%}",
    delta="Percentil empírico",
)
c2.metric(
    label=f"VaR Paramétrico ({confidence:.0%})",
    value=f"{r['var_parametrico']:.2%}",
    delta="Normal assumida",
)
c3.metric(
    label="CVaR / Expected Shortfall",
    value=f"{r['cvar']:.2%}",
    delta="Média das perdas extremas",
)

# Interpretação automática do VaR
st.info(
    f"**Interpretação:** Com {confidence:.0%} de confiança, a perda diária máxima esperada para "
    f"**{asset.name}** é de **{r['var_historico']:.2%}** do valor posicionado (janela de {window} pregões). "
    f"Em cenários de estresse que superam esse limiar, a perda média é de **{r['cvar']:.2%}** (CVaR)."
)

# Explicação metodológica com fórmula
with st.expander("📐 Metodologia de Cálculo do VaR e CVaR"):
    st.markdown(f"""
    **VaR Histórico:** Ordenam-se os retornos diários dos últimos {window} pregões. O VaR é o negativo do
    percentil correspondente a `1 - {confidence}`.
    
    **VaR Paramétrico:** Assume-se normalidade dos retornos. Calcula-se a média ($\\mu$) e o desvio padrão
    ($\\sigma$) amostrais. O VaR é:
    $$
    \\text{VaR}_{\\text{param}} = -\\left( \\mu + \\sigma \\cdot z_{{{1-{confidence}}}} \\right)
    $$
    onde $z$ é o quantil da normal padrão.
    
    **CVaR (Expected Shortfall):** Média dos retornos que são inferiores a `-VaR_historico`.
    """)

st.divider()

# --------------------------------------------------------------------------
# MÉTRICAS DE RISCO AJUSTADO
# --------------------------------------------------------------------------
st.subheader("📊 Risco Ajustado ao Retorno")

# Métricas principais
vol = metrics.annualized_volatility(close, window=window)
sharpe = metrics.sharpe_ratio(close, window=252)
sortino = metrics.sortino_ratio(close, window=252)
max_dd = metrics.max_drawdown(close.tail(252))

m1, m2, m3, m4 = st.columns(4)
m1.metric("Volatilidade Anualizada", f"{vol:.2%}")
m2.metric("Sharpe Ratio (252d)", f"{sharpe:.2f}")
m3.metric("Sortino Ratio (252d)", f"{sortino:.2f}")
m4.metric("Máximo Drawdown (252d)", f"{max_dd:.2%}")

# Interpretação inteligente
st.markdown("**Interpretação:**")
col_interpret = st.columns(2)
with col_interpret[0]:
    if sharpe > 1:
        st.success(f"✅ Sharpe > 1: retorno ajustado ao risco **bom** (excede a volatilidade).")
    elif sharpe > 0:
        st.warning(f"⚠️ Sharpe entre 0 e 1: retorno ajustado ao risco **moderado**.")
    else:
        st.error(f"❌ Sharpe negativo: retorno **inferior** à taxa livre de risco.")
    
    if sortino > sharpe:
        st.info(f"ℹ️ Sortino ({sortino:.2f}) > Sharpe ({sharpe:.2f}): indica que a volatilidade negativa é menor que a total.")
    else:
        st.info(f"ℹ️ Sortino ({sortino:.2f}) ≤ Sharpe ({sharpe:.2f}): o risco de baixa é relevante.")

with col_interpret[1]:
    if max_dd < -0.20:
        st.warning(f"⚠️ Drawdown máximo de {max_dd:.2%}: perda histórica significativa.")
    else:
        st.success(f"✅ Drawdown máximo de {max_dd:.2%}: dentro de patamares aceitáveis.")

# Fórmulas
with st.expander("📐 Metodologia das Métricas de Risco Ajustado"):
    st.markdown(f"""
    **Volatilidade Anualizada:**
    $$
    \\sigma_{\\text{anual}} = \\sigma_{\\text{diário}} \\times \\sqrt{252}
    $$
    
    **Sharpe Ratio:**
    $$
    \\text{Sharpe} = \\frac{{\\bar{{R}} - R_f}}{{\\sigma}}
    $$
    onde $\\bar{{R}}$ é a média dos retornos diários (anualizada), $R_f$ é a taxa livre de risco (4.5% a.a.), e $\\sigma$ é o desvio padrão anualizado.
    
    **Sortino Ratio:**
    $$
    \\text{Sortino} = \\frac{{\\bar{{R}} - R_f}}{{\\sigma_{\\text{down}}}}
    $$
    onde $\\sigma_{\\text{down}}$ é o desvio padrão dos retornos negativos.
    
    **Máximo Drawdown:**
    $$
    \\text{MDD} = \\max_{t} \\left( \\frac{{\\max_{{s \\leq t}} P_s - P_t}}{{\\max_{{s \\leq t}} P_s}} \\right)
    $$
    """)

st.divider()

# --------------------------------------------------------------------------
# STRESS TEST
# --------------------------------------------------------------------------
st.subheader("🔨 Stress Test — Choques Instantâneos")
st.caption("Simula o impacto de choques percentuais no preço atual sobre o retorno e a perda monetária.")

custom_shocks = st.text_input(
    "Choques (%), separados por vírgula",
    "-30,-20,-10,-5,5,10,20",
    help="Exemplo: -10,-5,5,10 representa quedas de 10% e 5%, e altas de 5% e 10%."
)

try:
    shocks = [float(x.strip()) / 100 for x in custom_shocks.split(",") if x.strip()]
except ValueError:
    shocks = None
    st.error("Formato inválido — use números separados por vírgula, ex: -30,-10,10,20")

if shocks:
    stress_df = risk.stress_test(close, shocks_pct=shocks)
    st.dataframe(stress_df, use_container_width=True, hide_index=True)
    st.plotly_chart(
        charts.bar_chart(
            stress_df["choque"].tolist(),
            stress_df["variacao_absoluta"].tolist(),
            title="Impacto Absoluto no Preço por Cenário de Choque",
            positive_negative=True,
        ),
        use_container_width=True,
    )
    
    with st.expander("📐 Metodologia do Stress Test"):
        st.markdown("""
        **Cenários:** Para cada choque percentual $s$ (ex: -0.10), calcula-se:
        
        - **Novo preço:** $P_{\\text{novo}} = P_{\\text{atual}} \\times (1 + s)$
        - **Variação absoluta:** $\\Delta P = P_{\\text{novo}} - P_{\\text{atual}}$
        - **Variação relativa:** $s$ (já definido)
        
        O gráfico mostra o impacto monetário (em unidades da moeda do ativo) para cada cenário.
        """)

st.divider()

# --------------------------------------------------------------------------
# DISTRIBUIÇÃO DE RETORNOS
# --------------------------------------------------------------------------
st.subheader("📊 Distribuição de Retornos Diários")
st.caption(f"Histograma dos retornos diários (últimos {window} pregões).")

rets = metrics.daily_returns(close).tail(window)
st.plotly_chart(
    charts.histogram_chart(
        rets.values,
        title=f"Distribuição de Retornos Diários - {asset.name}",
        x_title="Retorno diário",
        bins=50,
    ),
    use_container_width=True,
)

with st.expander("📐 Metodologia do Histograma"):
    st.markdown("""
    **Construção:** Os retornos diários são calculados como:
    $$
    R_t = \\frac{P_t}{P_{t-1}} - 1
    $$
    O histograma agrupa os retornos em intervalos (bins) e exibe a frequência de ocorrência.
    
    **Interpretação:**
    - Se a distribuição for simétrica e com caudas leves, os retornos aproximam-se de uma normal.
    - Caudas pesadas indicam maior probabilidade de eventos extremos (risco de cauda).
    - Assimetria negativa sugere que grandes perdas são mais frequentes do que grandes ganhos.
    """)

st.divider()

# --------------------------------------------------------------------------
# NOTAS FINAIS E REFERÊNCIAS
# --------------------------------------------------------------------------
with st.expander("📚 Referências Bibliográficas"):
    st.markdown("""
    - **Jorion, P. (2007).** *Value at Risk: The New Benchmark for Managing Financial Risk.* McGraw-Hill.
    - **Sharpe, W. F. (1966).** Mutual Fund Performance. *Journal of Business*, 39(1), 119-138.
    - **Sortino, F. A., & Price, L. N. (1994).** Performance Measurement in a Downside Risk Framework. *Journal of Investing*, 3(3), 59-64.
    - **Campbell, J. Y., Lo, A. W., & MacKinlay, A. C. (1997).** *The Econometrics of Financial Markets.* Princeton University Press.
    - **Basel Committee on Banking Supervision (2019).** *Minimum Capital Requirements for Market Risk.* (FRTB - Fundamental Review of the Trading Book).
    """)