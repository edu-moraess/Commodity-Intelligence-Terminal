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
if panel.empty:
    st.error("❌ Não foi possível carregar dados de preço para os ativos selecionados. "
             "Tente recarregar a página ou selecionar outros ativos.")
    st.stop()

panel.columns = [a.name for a in sel_assets if a.ticker in panel.columns]

macro_panel = pd.DataFrame(macro_data).ffill()
combined = panel.join(macro_panel, how="inner").dropna(how="all")

if combined.empty or combined.shape[1] < 2:
    st.error("❌ Dados insuficientes para calcular correlações. "
             "O painel combinado está vazio ou tem menos de 2 séries válidas.")
    st.stop()

if combined.shape[0] < 30:
    st.warning(f"⚠️ Apenas {combined.shape[0]} observações válidas — correlações podem ser instáveis.", icon="⚠️")

st.subheader("Heatmap de Correlação — Commodities × Macro")
window = st.slider("Janela (pregões)", 60, 500, 252, step=20)
corr = correlation.correlation_matrix(combined, window=window)

if corr.empty:
    st.warning("⚠️ Matriz de correlação vazia — dados insuficientes ou séries muito curtas.")
else:
    st.plotly_chart(
        charts.correlation_heatmap(corr, title=f"Correlação ({window} pregões)"),
        use_container_width=True,
    )

with st.expander("📖 Metodologia — Correlação de Pearson"):
    st.markdown(f"""
    O heatmap exibe o **coeficiente de correlação de Pearson (ρ)** entre os **retornos diários** de cada par de séries, calculado sobre uma janela móvel de **{window} pregões**.
    
    **Fórmula:** ρ(X,Y) = Cov(X,Y) / (σₓ × σᵧ)
    
    | ρ | Interpretação |
    |---|---------------|
    | +1.0 | Movimentos perfeitamente sincronizados |
    | +0.5 a +0.9 | Correlação positiva forte (comum entre commodities e DXY inverso) |
    | 0 a ±0.3 | Correlação fraca ou nula |
    | –0.5 a –0.9 | Correlação negativa forte (ex: ouro vs DXY) |
    
    **⚠️ Atenção:** Correlação ≠ causalidade. Uma correlação alta pode ser espúria (ambas respondendo a um terceiro fator, como liquidez global). Além disso, correlações são instáveis no tempo — por isso incluímos a análise rolante abaixo.
    """)

st.divider()

st.subheader("Correlação Rolante")
col1, col2 = st.columns(2)
with col1:
    asset_a = st.selectbox("Série A", combined.columns, index=0)
with col2:
    asset_b = st.selectbox("Série B", combined.columns, index=min(1, len(combined.columns) - 1))
roll_window = st.slider("Janela rolante (pregões)", 20, 250, 63, step=5)

roll_corr = correlation.rolling_correlation(combined[asset_a], combined[asset_b], window=roll_window)
if roll_corr.empty:
    st.warning("⚠️ Dados insuficientes para correlação rolante.")
else:
    st.plotly_chart(
        charts.line_chart({f"Corr({asset_a}, {asset_b})": roll_corr}, title="Correlação Rolante", y_title="ρ"),
        use_container_width=True,
    )

with st.expander("📖 Metodologia — Correlação Rolante"):
    st.markdown(f"""
    A correlação rolante calcula o coeficiente de Pearson em uma **janela móvel de {roll_window} pregões**, deslizando dia após dia ao longo da série histórica.
    
    **Por que usar?**  
    A correlação entre ativos **não é constante**. Em crises, correlações tendem a convergir para +1 (tudo cai junto). Em períodos normais, podem ser mais baixas. A série rolante revela essa evolução temporal.
    
    **Como interpretar:**
    - Linha subindo → as séries estão ficando mais sincronizadas
    - Linha cruzando zero → a relação inverteu de sinal
    - Picos abruptos → eventos de stress de mercado
    """)

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

with st.expander("📖 Metodologia — Análise de Componentes Principais (PCA)"):
    st.markdown("""
    O PCA é uma técnica de redução de dimensionalidade que transforma as séries de retornos em componentes ortogonais (independentes), ordenados por quanto de variância explicam.
    
    **Como funciona:**
    1. Calcula a matriz de covariância dos retornos diários
    2. Aplica decomposição em valores singulares (SVD)
    3. Os autovetores são os "loadings" (pesos de cada ativo no componente)
    4. Os autovalores determinam a variância explicada
    
    **Interpretação prática:**
    - **PC1** geralmente captura o "fator mercado" — o movimento comum de todas as commodities
    - **PC2** pode capturar a divergência entre energia vs. metais/agricultura
    - Se PC1 explica > 70% da variância, o universo é altamente correlacionado (pouca diversificação)
    - Se a variância está espalhada entre vários PCs, há mais independência entre os ativos
    
    **Loadings:** valores positivos altos indicam que o ativo contribui fortemente na direção daquele componente; valores negativos indicam contribuição na direção oposta.
    """)
