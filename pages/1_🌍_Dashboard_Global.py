import streamlit as st
import pandas as pd

from config.settings import ENERGY_ASSETS, METALS_ASSETS, AGRI_ASSETS, APP_NAME, RISK_FREE_RATE_ANNUAL
from data.data_manager import load_price_history_bulk
from analytics import metrics, signals
from charts import plotly_charts as charts

# NOTA v4.5.0: st.set_page_config() removido daqui — agora é chamado
# uma única vez em app.py, antes de st.navigation(...).run().

# --------------------------------------------------------------------------
# CABEÇALHO COM METODOLOGIA GERAL
# --------------------------------------------------------------------------
st.title("🌍 Dashboard Global de Commodities")
st.caption(
    "Visão consolidada — Energia, Metais e Agricultura, com métricas de retorno, risco e tendência. "
    "Abaixo, você encontra a **metodologia** de cada indicador e sua interpretação prática."
)

with st.expander("📘 Sobre este Dashboard (Metodologia Geral)", expanded=False):
    st.markdown(r"""
    **Objetivo:** Fornecer uma visão panorâmica dos principais mercados de commodities, com métricas padronizadas para comparação entre setores.
    
    **Métricas apresentadas:**
    - **Retornos (1D, 1S, 1M, YTD):** Variação percentual do preço no período.
    - **Volatilidade Anualizada:** Desvio padrão dos retornos diários anualizado (multiplicado por √252).
    - **Sharpe Ratio:** Retorno ajustado ao risco total (volatilidade). Quanto maior, melhor.
    - **Sortino Ratio:** Similar ao Sharpe, mas penaliza apenas o risco de baixa (downside).
    - **Máximo Drawdown:** Maior queda acumulada do preço de um pico a um vale (janela de 252 pregões).
    - **Calmar Ratio:** Relação entre o retorno acumulado e o máximo drawdown – mede a eficiência em relação à pior perda.
    - **Momentum Composto:** Média simples dos retornos de 1, 3, 6 e 12 meses – sinaliza a tendência de curto/médio prazo.
    - **Tendência:** Classificação por cruzamento de médias móveis (alta, lateral, baixa) — ver detalhe abaixo.
    """)

st.divider()

# --------------------------------------------------------------------------
# CARREGAMENTO DE DADOS
# --------------------------------------------------------------------------
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

# --------------------------------------------------------------------------
# KPIs DE TOPO (com tratamento de NaN)
# --------------------------------------------------------------------------
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
    if not close.empty and pd.notna(close.iloc[-1]):
        last_price = close.iloc[-1]
        chg = metrics.pct_change_over(close, 1)
        display_price = f"{last_price:.2f}"
        display_delta = f"{chg:+.2%}" if pd.notna(chg) else None
    else:
        display_price = "N/D"
        display_delta = None
    with col:
        st.metric(label, display_price, display_delta)

st.divider()

# --------------------------------------------------------------------------
# TABELA MESTRE POR SETOR (com formatação e tratamento de NaN)
# --------------------------------------------------------------------------
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

# Formatação da tabela
pct_cols = ["1D", "1S", "1M", "YTD", "Vol.Anual", "Max DD", "Momentum"]
fmt = {c: "{:.2%}" for c in pct_cols}
fmt.update({"Último": "{:.2f}", "Sharpe": "{:.2f}", "Sortino": "{:.2f}", "Calmar": "{:.2f}"})

tabs = st.tabs(list(ALL_SECTORS.keys()) + ["Todos"])

for tab, (sector_name, assets) in zip(tabs[:-1], ALL_SECTORS.items()):
    with tab:
        df = build_table(assets)
        st.dataframe(
            df.style.format(fmt, na_rep="-"),
            use_container_width=True,
            height=min(38 * (len(df) + 1) + 20, 400)
        )

with tabs[-1]:
    df_all = build_table(ALL_ASSETS)
    st.dataframe(
        df_all.style.format(fmt, na_rep="-"),
        use_container_width=True,
        height=560
    )

# --------------------------------------------------------------------------
# EXPANDER DE METODOLOGIA DAS MÉTRICAS (após a tabela)
# --------------------------------------------------------------------------
with st.expander("📐 Como as métricas são calculadas? (Fórmulas)"):
    st.markdown(r"""
    **Retornos (1D, 1S, 1M, YTD):** Variação percentual simples entre o preço atual e o preço de \(n\) dias atrás.
    
    **Volatilidade Anualizada:**
    $$
    \sigma_{\text{anual}} = \sigma_{\text{diário}} \times \sqrt{252}
    $$
    
    **Sharpe Ratio (Sharpe, 1966):**
    $$
    \text{Sharpe} = \frac{\bar{R} - R_f}{\sigma}
    $$
    onde \(\bar{R}\) é o retorno médio diário (anualizado), \(R_f\) é a taxa livre de risco (4.5% a.a.), e \(\sigma\) é o desvio padrão anualizado.
    
    **Sortino Ratio (Sortino & Price, 1994):**
    $$
    \text{Sortino} = \frac{\bar{R} - R_f}{\sigma_{\text{down}}}
    $$
    onde \(\sigma_{\text{down}}\) é o desvio padrão **apenas** dos retornos negativos (downside deviation).
    
    **Máximo Drawdown (MDD):**
    $$
    \text{MDD} = \max_{t} \left( \frac{\max_{s \leq t} P_s - P_t}{\max_{s \leq t} P_s} \right)
    $$
    mede a maior queda acumulada do preço de um pico a um vale (janela de 252 pregões).
    
    **Calmar Ratio:**
    $$
    \text{Calmar} = \frac{\text{Retorno Acumulado (252d)}}{|\text{MDD}|}
    $$
    Quanto maior, melhor o desempenho ajustado à pior perda histórica.
    
    **Momentum Composto:**
    Média simples (não ponderada) dos retornos de 1, 3, 6 e 12 meses. Sinaliza a direção da tendência.
    
    **Tendência:** Compara a média móvel simples curta (20 pregões) com a longa (100 pregões):
    - **alta**: SMA(20) > SMA(100) × 1,01
    - **baixa**: SMA(20) < SMA(100) × 0,99
    - **lateral**: caso contrário (as duas médias estão próximas)
    """)

st.divider()

# --------------------------------------------------------------------------
# TREEMAP DE PERFORMANCE (com metodologia integrada)
# --------------------------------------------------------------------------
st.subheader("🗺️ Mapa de Performance (1 mês)")
st.caption("Cada bloco representa um ativo; o tamanho é proporcional ao valor absoluto da variação no mês.")

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

st.plotly_chart(
    charts.treemap_chart(labels, parents, values, title="Tamanho = |variação 1M| (ilustrativo)"),
    use_container_width=True
)

with st.expander("📐 Como interpretar o Treemap?"):
    st.markdown(r"""
    **Construção:**
    - Cada retângulo representa um ativo (ou setor, no nível superior).
    - A **área** do retângulo é proporcional ao valor absoluto da variação percentual do ativo no último mês (\(|R_{1M}|\)).
    - Ativos com maior variação (positiva ou negativa) aparecem com blocos maiores.
    
    **Utilidade:**
    - Permite identificar rapidamente quais ativos tiveram maior volatilidade ou movimento no período.
    - Cores podem ser adicionadas para diferenciar setores (verde para alta, vermelho para baixa, etc.).
    """)

st.divider()

# --------------------------------------------------------------------------
# RISCO-RETORNO CONSOLIDADO (scatter) + PAINEL DE SINAIS
# --------------------------------------------------------------------------
st.subheader("🎯 Risco vs. Retorno — Visão Consolidada")
st.caption(
    "Cada bolha é um ativo: eixo X = volatilidade anualizada, eixo Y = Sharpe Ratio, "
    "tamanho da bolha = |momentum composto|. Ativos no quadrante superior-esquerdo "
    "(baixa vol, Sharpe alto) têm o melhor perfil risco-retorno recente."
)

with st.spinner("Calculando sinais de risco para todos os ativos..."):
    scatter_rows = []
    signal_rows = []
    for sector_name, assets in ALL_SECTORS.items():
        for a in assets:
            pdat = price_data[a.ticker]
            close = pdat.df["Close"]
            row = metrics.summary_row(close, risk_free_annual=RISK_FREE_RATE_ANNUAL)
            if row["last_price"] is None or pd.isna(row["vol_annual"]):
                continue
            scatter_rows.append({
                "name": a.name, "sector": sector_name,
                "vol": row["vol_annual"], "sharpe": row["sharpe"], "momentum": row["momentum"],
            })
            sig = signals.build_signal_row(close, momentum=row["momentum"])
            signal_rows.append({
                "Ativo": a.name + (" 🔸" if pdat.is_synthetic else ""),
                "Setor": sector_name,
                "Regime de Vol.": sig["vol_regime"],
                "Momentum": sig["momentum_label"],
                "Z-Score Retorno (1d)": sig["return_zscore"],
                "Movimento Atípico": "⚠️ Sim" if sig["extreme_move"] else "Não",
            })

scatter_df = pd.DataFrame(scatter_rows)
if not scatter_df.empty:
    st.plotly_chart(charts.risk_return_scatter(scatter_df), use_container_width=True)
else:
    st.info("Dados insuficientes para o gráfico de risco-retorno.")

st.divider()

st.subheader("🚨 Painel de Sinais Consolidado")
st.caption(
    "Leitura rápida de regime de volatilidade (percentil da vol. realizada 21d vs. a própria "
    "história do ativo) e movimentos atípicos (z-score do retorno diário). Para análise "
    "aprofundada de um ativo específico — incluindo HMM completo e backtesting formal de VaR — "
    "use a página **Risk Analytics**."
)

signal_df = pd.DataFrame(signal_rows).set_index("Ativo")
n_alerts = (signal_df["Movimento Atípico"] == "⚠️ Sim").sum()
n_high_vol = signal_df["Regime de Vol."].str.contains("Elevada").sum()

sc1, sc2, sc3 = st.columns(3)
sc1.metric("Ativos Monitorados", len(signal_df))
sc2.metric("Em Volatilidade Elevada", int(n_high_vol))
sc3.metric("Movimentos Atípicos Hoje", int(n_alerts))

filter_option = st.radio(
    "Filtrar", ["Todos", "Só Vol. Elevada", "Só Movimentos Atípicos"],
    horizontal=True, key="signal_filter",
)
display_df = signal_df.copy()
if filter_option == "Só Vol. Elevada":
    display_df = display_df[display_df["Regime de Vol."].str.contains("Elevada")]
elif filter_option == "Só Movimentos Atípicos":
    display_df = display_df[display_df["Movimento Atípico"] == "⚠️ Sim"]

st.dataframe(
    display_df.style.format({"Z-Score Retorno (1d)": "{:.2f}"}, na_rep="-"),
    use_container_width=True,
    height=min(38 * (len(display_df) + 1) + 20, 500),
)

with st.expander("📐 Metodologia do Painel de Sinais"):
    st.markdown(r"""
    **Regime de Volatilidade:** calcula a volatilidade realizada em janela curta (21 pregões) e
    compara com a distribuição de volatilidade rolante do próprio ativo nos últimos 252 pregões,
    reportando o percentil. ≥80º percentil = "Vol. Elevada"; ≤20º percentil = "Vol. Baixa".
    É um proxy rápido — para uma classificação de regime estatisticamente rigorosa (Hidden Markov
    Model com teste de significância), use a aba **Regimes de Volatilidade** em Risk Analytics.

    **Z-Score do Retorno:** normaliza o retorno do último pregão pela média e desvio-padrão dos
    últimos 63 retornos. \(|z| > 2\) é sinalizado como movimento atípico (~2 desvios-padrão,
    ocorrência esperada em ~5% dos dias sob normalidade).
    """)


st.caption(
    "Fontes: Yahoo Finance (preços de futuros/proxies) com fallback sintético automático. "
    "Métricas: retornos, volatilidade anualizada (63d), Sharpe/Sortino/Calmar (252d, "
    f"taxa livre de risco = {RISK_FREE_RATE_ANNUAL:.2%} a.a.), máximo drawdown (252d) e momentum "
    "composto (1M/3M/6M/12M)."
)

# Expander com referências
with st.expander("📚 Referências Acadêmicas"):
    st.markdown(r"""
    - **Sharpe, W. F. (1966).** Mutual Fund Performance. *Journal of Business*, 39(1), 119-138.
    - **Sortino, F. A., & Price, L. N. (1994).** Performance Measurement in a Downside Risk Framework. *Journal of Investing*, 3(3), 59-64.
    - **Campbell, J. Y., Lo, A. W., & MacKinlay, A. C. (1997).** *The Econometrics of Financial Markets.* Princeton University Press.
    """)