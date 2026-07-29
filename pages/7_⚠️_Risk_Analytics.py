import streamlit as st
import pandas as pd
import numpy as np

from config.settings import ALL_ASSETS, APP_NAME
from data.data_manager import load_price_history
from analytics import risk, metrics
from charts import plotly_charts as charts

st.set_page_config(page_title=f"Risk Analytics — {APP_NAME}", page_icon="⚠️", layout="wide")

# --------------------------------------------------------------------------
# CABEÇALHO (Metodologia Geral da Página)
# --------------------------------------------------------------------------
st.title("⚠️ Risk Analytics")
st.caption(
    "Módulo de análise de risco com VaR, CVaR, métricas ajustadas, stress test e distribuição de retornos. "
    "Abaixo de cada indicador, você encontrará a **fórmula matemática** e a **interpretação prática**."
)

with st.expander("📘 Sobre esta página (Metodologia Geral)", expanded=False):
    st.markdown(r"""
    **Objetivo:** Quantificar o risco de mercado de um ativo específico, utilizando abordagens complementares.
    
    **1. Value at Risk (VaR):** Mede a perda máxima esperada em um horizonte de tempo, sob um determinado nível de confiança.
    - *Histórico:* usa a distribuição empírica dos retornos passados.
    - *Paramétrico:* assume normalidade (distribuição Gaussiana).
    
    **2. Expected Shortfall (CVaR):** Complementa o VaR medindo a perda *média* nos cenários mais extremos (cauda da distribuição).
    
    **3. Métricas Ajustadas (Sharpe, Sortino, Drawdown):** Avaliam a eficiência do retorno em relação ao risco corrido.
    
    **4. Stress Test:** Simula o impacto de choques instantâneos no preço atual.
    
    **5. Distribuição dos Retornos:** Visualiza a frequência dos retornos diários para identificar assimetrias e caudas pesadas.
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
# SEÇÃO 1: VAR E CVAR
# --------------------------------------------------------------------------
st.header("📉 Value at Risk (VaR) e Expected Shortfall (CVaR)")

r = risk.risk_summary(close, confidence=confidence, window=window)

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

st.info(
    f"**Interpretação prática:** Com {confidence:.0%} de confiança, a perda diária máxima esperada para "
    f"**{asset.name}** é de **{r['var_historico']:.2%}** do valor posicionado (janela de {window} pregões). "
    f"Em cenários de estresse que superam esse limiar, a perda média é de **{r['cvar']:.2%}** (CVaR)."
)

with st.expander("📐 Como calculamos o VaR e o CVaR? (Fórmulas)"):
    st.markdown(
        f"""
        **VaR Histórico (Não-Paramétrico):**
        Ordenam-se os retornos diários dos últimos {window} pregões. O VaR é o negativo do percentil empírico `(1 - {confidence})`.
        
        **VaR Paramétrico (Normal):**
        Assume-se que os retornos seguem uma distribuição normal. Calcula-se a média ($\\mu$) e o desvio padrão ($\\sigma$) amostrais.
        """
    )
    st.latex(rf"\text{{VaR}}_{{\text{{param}}}} = -\left( \mu + \sigma \cdot z_{{1-{confidence}}} \right)")
    st.markdown(
        f"""
        onde $z$ é o quantil da distribuição normal padrão para o nível de confiança {confidence:.0%}.
        
        **CVaR (Expected Shortfall):**
        É a média aritmética de todos os retornos diários que são **menores** que `-VaR_historico`.
        """
    )

st.divider()

# --------------------------------------------------------------------------
# SEÇÃO 2: MÉTRICAS DE RISCO AJUSTADO
# --------------------------------------------------------------------------
st.header("📊 Risco Ajustado ao Retorno")

vol = metrics.annualized_volatility(close, window=window)
sharpe = metrics.sharpe_ratio(close, window=252)
sortino = metrics.sortino_ratio(close, window=252)
max_dd = metrics.max_drawdown(close.tail(252))

m1, m2, m3, m4 = st.columns(4)
m1.metric("Volatilidade Anualizada", f"{vol:.2%}")
m2.metric("Sharpe Ratio (252d)", f"{sharpe:.2f}")
m3.metric("Sortino Ratio (252d)", f"{sortino:.2f}")
m4.metric("Máximo Drawdown (252d)", f"{max_dd:.2%}")

st.markdown("**Interpretação dos Resultados:**")
col_interpret = st.columns(2)
with col_interpret[0]:
    if sharpe > 1:
        st.success(f"✅ Sharpe > 1: retorno ajustado ao risco **bom** (excede a volatilidade).")
    elif sharpe > 0:
        st.warning(f"⚠️ Sharpe entre 0 e 1: retorno ajustado ao risco **moderado**.")
    else:
        st.error(f"❌ Sharpe negativo: retorno **inferior** à taxa livre de risco.")
    
    if sortino > sharpe:
        st.info(f"ℹ️ Sortino ({sortino:.2f}) > Sharpe ({sharpe:.2f}): o risco de baixa (downside) é menor que o risco total.")
    else:
        st.info(f"ℹ️ Sortino ({sortino:.2f}) ≤ Sharpe ({sharpe:.2f}): o risco de baixa é relevante.")

with col_interpret[1]:
    if max_dd < -0.20:
        st.warning(f"⚠️ Drawdown máximo de {max_dd:.2%}: perda histórica significativa (pico ao vale).")
    else:
        st.success(f"✅ Drawdown máximo de {max_dd:.2%}: dentro de patamares aceitáveis para a maioria dos ativos.")

with st.expander("📐 Como calculamos as métricas ajustadas? (Fórmulas)"):
    st.markdown(
        r"""
        **Volatilidade Anualizada:**
        $$
        \sigma_{\text{anual}} = \sigma_{\text{diário}} \times \sqrt{252}
        $$
        
        **Sharpe Ratio (Sharpe, 1966):**
        $$
        \text{Sharpe} = \frac{\bar{R} - R_f}{\sigma}
        $$
        onde $\bar{R}$ é o retorno médio diário (anualizado), $R_f$ é a taxa livre de risco (4.5% a.a.), e $\sigma$ é o desvio padrão anualizado.
        
        **Sortino Ratio (Sortino & Price, 1994):**
        $$
        \text{Sortino} = \frac{\bar{R} - R_f}{\sigma_{\text{down}}}
        $$
        onde $\sigma_{\text{down}}$ é o desvio padrão **apenas** dos retornos negativos (downside deviation).
        
        **Máximo Drawdown (MDD):**
        $$
        \text{MDD} = \max_{t} \left( \frac{\max_{s \leq t} P_s - P_t}{\max_{s \leq t} P_s} \right)
        $$
        mede a maior queda acumulada do preço de um pico a um vale.
        """
    )

st.divider()

# --------------------------------------------------------------------------
# SEÇÃO 3: STRESS TEST
# --------------------------------------------------------------------------
st.header("🔨 Stress Test — Choques Instantâneos")
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
    
    with st.expander("📐 Como funciona o Stress Test?"):
        st.markdown(r"""
        Para cada cenário de choque $s$ (ex: -0.10 para queda de 10%), calculamos:
        
        - **Novo preço:** $P_{\text{novo}} = P_{\text{atual}} \times (1 + s)$
        - **Variação absoluta:** $\Delta P = P_{\text{novo}} - P_{\text{atual}}$
        - **Variação relativa:** $s$ (já definido)
        
        O gráfico mostra o impacto monetário (em unidades da moeda do ativo) para cada cenário, permitindo visualizar rapidamente a sensibilidade do ativo a choques adversos.
        """)

st.divider()

# --------------------------------------------------------------------------
# SEÇÃO 4: DISTRIBUIÇÃO DE RETORNOS (CORRIGIDA)
# --------------------------------------------------------------------------
st.header("📊 Distribuição de Retornos Diários")
st.caption(f"Histograma dos retornos diários (últimos {window} pregões).")

rets = metrics.daily_returns(close).tail(window)

# CORREÇÃO: removido o argumento 'bins=50' que causava o erro
st.plotly_chart(
    charts.histogram_chart(
        rets.values,
        title=f"Distribuição de Retornos Diários - {asset.name}",
        x_title="Retorno diário"
    ),
    use_container_width=True,
)

with st.expander("📐 Como interpretar o histograma?"):
    st.markdown(r"""
    **Construção do Histograma:**
    Os retornos diários são calculados como:
    $$
    R_t = \frac{P_t}{P_{t-1}} - 1
    $$
    O histograma agrupa esses retornos em intervalos (bins) e exibe a frequência de ocorrência.
    
    **O que observar:**
    - **Simetria:** Se a distribuição for simétrica, os retornos se aproximam de uma normal.
    - **Caudas pesadas:** Se houver barras altas nas extremidades, há maior probabilidade de eventos extremos (risco de cauda).
    - **Assimetria (Skewness):** Se a cauda esquerda for mais longa, grandes perdas são mais frequentes do que grandes ganhos (risco assimétrico).
    """)

st.divider()

# --------------------------------------------------------------------------
# SEÇÃO 5: REFERÊNCIAS BIBLIOGRÁFICAS
# --------------------------------------------------------------------------
with st.expander("📚 Referências Acadêmicas e Regulatórias"):
    st.markdown(r"""
    - **Jorion, P. (2007).** *Value at Risk: The New Benchmark for Managing Financial Risk.* McGraw-Hill.
    - **Sharpe, W. F. (1966).** Mutual Fund Performance. *Journal of Business*, 39(1), 119-138.
    - **Sortino, F. A., & Price, L. N. (1994).** Performance Measurement in a Downside Risk Framework. *Journal of Investing*, 3(3), 59-64.
    - **Campbell, J. Y., Lo, A. W., & MacKinlay, A. C. (1997).** *The Econometrics of Financial Markets.* Princeton University Press.
    - **Basel Committee on Banking Supervision (2019).** *Minimum Capital Requirements for Market Risk* (FRTB - Fundamental Review of the Trading Book).
    """)