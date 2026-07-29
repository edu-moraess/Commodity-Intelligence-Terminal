import streamlit as st
import numpy as np
import pandas as pd

from config.settings import ALL_ASSETS, APP_NAME
from data.data_manager import load_price_history
from analytics import volatility as vol_mod
from forecasting import models as fc
from charts import plotly_charts as charts

st.set_page_config(page_title=f"Quant Research — {APP_NAME}", page_icon="🧮", layout="wide")

st.title("🧮 Quant Research")
st.caption(
    "Modelagem de volatilidade condicional (GARCH), comparação de modelos de tendência via "
    "walk-forward validation e Monte Carlo já detalhado na página Forecast."
)

asset_names = {a.name: a for a in ALL_ASSETS}
selected_name = st.selectbox("Ativo", list(asset_names.keys()))
asset = asset_names[selected_name]

with st.spinner("Carregando dados..."):
    pdat = load_price_history(asset)
close = pdat.df["Close"]

if pdat.is_synthetic:
    st.warning("⚠️ Exibindo **dados simulados** — fonte ao vivo indisponível neste ambiente.", icon="⚠️")

# ============================================================
# GARCH(1,1)
# ============================================================
st.subheader("Volatilidade Condicional — GARCH(1,1)")
lookback = st.slider("Janela de estimação (pregões)", 250, 1000, 500, step=50)

with st.spinner("Ajustando GARCH(1,1) via máxima verossimilhança..."):
    garch = vol_mod.fit_garch11(close, lookback=lookback)
ewma = vol_mod.ewma_volatility(close, window=lookback)

g1, g2, g3, g4 = st.columns(4)
g1.metric("ω (omega)", f"{garch['omega']:.4f}")
g2.metric("α (alpha)", f"{garch['alpha']:.4f}")
g3.metric("β (beta)", f"{garch['beta']:.4f}")
g4.metric("Persistência (α+β)", f"{garch['persistence']:.4f}")

st.caption(
    f"Log-likelihood: {garch['log_likelihood']:.1f} · Convergência: "
    f"{'✅ sim' if garch['converged'] else '⚠️ não — resultado indicativo'} · "
    f"Previsão de volatilidade anualizada para o próximo pregão: "
    f"**{garch['forecast_1d_vol_annualized']:.2%}**"
)

st.plotly_chart(
    charts.line_chart(
        {"GARCH(1,1)": garch["conditional_vol_annualized"], "EWMA (RiskMetrics λ=0.94)": ewma},
        title="Volatilidade Condicional Anualizada", y_title="%",
    ),
    use_container_width=True,
)

with st.expander("📖 Metodologia — GARCH(1,1)"):
    st.markdown(f"""
    ### Modelo GARCH(1,1)
    
    O GARCH (Generalized Autoregressive Conditional Heteroskedasticity) modela a volatilidade como um processo que depende do próprio passado:
    
    **σ²ₜ = ω + α·r²ₜ₋₁ + β·σ²ₜ₋₁**
    
    | Parâmetro | Significado | Interpretação |
    |-----------|-------------|---------------|
    | **ω (omega)** | Variância de longo prazo | Quanto maior, maior a volatilidade média do ativo |
    | **α (alpha)** | Reação a choques recentes | Quanto maior, mais a volatilidade reage a movimentos bruscos de preço |
    | **β (beta)** | Persistência da volatilidade | Quanto maior, mais tempo a volatilidade leva para voltar à média |
    | **Persistência (α+β)** | Memória do processo | Se próximo de 1, choques são muito persistentes; se < 0.5, volatilidade volta rápido à média |
    
    **Previsão 1 dia:** σ²ₜ₊₁ = ω + α·r²ₜ + β·σ²ₜ
    
    ### EWMA (RiskMetrics)
    Método mais simples onde a volatilidade é uma média móvel exponencial dos retornos ao quadrado, com fator de decaimento λ = 0.94. É um caso limite do GARCH onde ω = 0 e α+β = 1.
    
    **Janela de estimação desta rodada:** {lookback} pregões.
    """)

st.divider()

# ============================================================
# Comparação de modelos de tendência — walk-forward validation
# ============================================================
st.subheader("Comparação de Modelos — Walk-Forward Validation")
st.caption(
    "Cada modelo é reajustado em janelas móveis e avaliado fora da amostra (1 passo à frente), "
    "reportando MAE e RMSE agregados — evita o otimismo de um único ajuste in-sample."
)

n_folds = st.slider("Número de folds (walk-forward)", 5, 40, 15)
train_window = st.slider("Janela de treino por fold (pregões)", 60, 252, 120, step=10)

if st.button("Rodar Walk-Forward Validation", type="primary"):
    log_close = np.log(close.values)
    results = {name: {"mae": [], "rmse": []} for name in fc.MODEL_REGISTRY}
    total_len = len(log_close)
    fold_starts = np.linspace(train_window, total_len - 2, n_folds, dtype=int)

    progress = st.progress(0.0, text="Rodando folds...")
    for i, start in enumerate(fold_starts):
        y_train = log_close[start - train_window:start]
        X_train = np.arange(train_window).reshape(-1, 1)
        y_true_next = log_close[start]
        X_next = np.array([[train_window]])

        for name, base_model in fc.MODEL_REGISTRY.items():
            from sklearn.base import clone
            model = clone(base_model)
            model.fit(X_train, y_train)
            pred = model.predict(X_next)[0]
            err = np.exp(pred) - np.exp(y_true_next)
            results[name]["mae"].append(abs(err))
            results[name]["rmse"].append(err**2)
        progress.progress((i + 1) / len(fold_starts), text=f"Fold {i+1}/{len(fold_starts)}")
    progress.empty()

    summary = []
    for name, r in results.items():
        summary.append({
            "Modelo": name,
            "MAE (1-step, unid. preço)": np.mean(r["mae"]),
            "RMSE (1-step, unid. preço)": np.sqrt(np.mean(r["rmse"])),
        })
    df_summary = pd.DataFrame(summary).set_index("Modelo").sort_values("RMSE (1-step, unid. preço)")
    st.dataframe(df_summary.style.format("{:.4f}"), use_container_width=True)
    best = df_summary.index[0]
    st.success(f"Melhor modelo por RMSE fora da amostra: **{best}**")
else:
    st.info("Configure os parâmetros e clique em **Rodar Walk-Forward Validation**.")

with st.expander("📖 Metodologia — Walk-Forward Validation"):
    st.markdown(f"""
    ### Por que não usar apenas in-sample?
    Ajustar um modelo em toda a série histórica e testar no mesmo dados gera **otimismo de seleção** — o modelo parece melhor do que realmente é. O walk-forward simula como o modelo seria usado na prática: treina numa janela, prevê o próximo ponto, desliza a janela e repete.
    
    ### Como funciona
    1. **Janela de treino:** {train_window} pregões (~{train_window//21:.0f} meses)
    2. **Previsão:** 1 passo à frente (próximo pregão)
    3. **Erro:** `|exp(pred_log) – exp(real_log)|` em unidades de preço
    4. **Deslizamento:** a janela avança e o modelo é **reajustado do zero**
    5. **Folds:** {n_folds} iterações, cobrindo diferentes regimes de mercado
    
    ### Métricas
    - **MAE (Mean Absolute Error):** erro médio absoluto — robusto a outliers
    - **RMSE (Root Mean Squared Error):** penaliza erros grandes mais severamente
    
    **Interpretação:** o modelo com menor RMSE fora da amostra tem melhor capacidade preditiva genuína. Não necessariamente o mais complexo — às vezes Ridge supera Lasso por ser mais estável.
    """)

st.divider()
st.caption(
    "Roadmap de expansão deste módulo: VAR/VECM/Cointegração, Kalman Filter, Hidden Markov "
    "Models, XGBoost/LightGBM/CatBoost, LSTM/GRU/Temporal Fusion Transformer, SHAP values — "
    "ver README para priorização e requisitos de dados."
)
