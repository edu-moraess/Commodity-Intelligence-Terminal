import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats

from config.settings import ALL_ASSETS, APP_NAME, RISK_FREE_RATE_ANNUAL
from data.data_manager import load_price_history
from analytics import volatility as vol_mod
from analytics import risk, metrics
from forecasting import models as fc
from charts import plotly_charts as charts

st.set_page_config(page_title=f"Quant Research — {APP_NAME}", page_icon="🧮", layout="wide")

st.title("🧮 Quant Research")
st.caption(
    "Modelagem de volatilidade condicional (GARCH), comparação de modelos de tendência via "
    "walk-forward validation, teste de Diebold-Mariano e painel de diagnóstico de saúde do modelo."
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
# PAINEL DE DIAGNÓSTICO — v5.0
# ============================================================
st.subheader("📊 Painel de Diagnóstico do Modelo")

# AJUSTE PROMPT — v5.0: painel de diagnóstico com 3 indicadores
lookback_diag = 252
rets_diag = metrics.daily_returns(close).tail(lookback_diag)

# (i) Persistência da volatilidade nos últimos 30 dias
garch_diag = vol_mod.fit_garch11(close, lookback=500)
persistencia = garch_diag["persistence"]

# (ii) Violações de VaR na última semana (5 pregões)
var_95 = risk.historical_var(close, confidence=0.95, window=252)
violacoes_semana = sum(rets_diag.tail(5) < -var_95) if pd.notna(var_95) else 0

# (iii) Bias de volatilidade: diferença entre GARCH forecast e vol realizada EWMA
vol_realizada = rets_diag.std() * np.sqrt(252)
vol_garch_forecast = garch_diag["forecast_1d_vol_annualized"]
bias_vol = ((vol_garch_forecast - vol_realizada) / vol_realizada * 100) if vol_realizada > 0 else None

d1, d2, d3 = st.columns(3)
with d1:
    st.metric("Persistência Vol (α+β)", f"{persistencia:.4f}",
              delta="alta persistência" if persistencia > 0.95 else "normal")
with d2:
    st.metric("Violações VaR (5d)", f"{violacoes_semana}/5",
              delta="alerta" if violacoes_semana >= 2 else "ok")
with d3:
    if bias_vol is not None:
        st.metric("Bias Vol GARCH", f"{bias_vol:+.1f}%",
                  delta="superestima" if bias_vol > 10 else "subestima" if bias_vol < -10 else "ok")
    else:
        st.metric("Bias Vol GARCH", "N/D")

with st.expander("📖 Metodologia — Painel de Diagnóstico"):
    st.markdown("""
    **Persistência (α+β):** mede quanto tempo a volatilidade leva para voltar à média após um choque.  
    - > 0.95: choques são muito persistentes (memória longa)
    - 0.85–0.95: regime normal
    - < 0.85: volatilidade reverte rapidamente

    **Violações VaR (5d):** conta quantos dos últimos 5 pregões tiveram perda maior que o VaR 95% histórico.  
    - 0–1: modelo calibrado
    - 2–3: alerta — possível subestimação de risco
    - 4–5: risco sistêmico ou modelo quebrado

    **Bias de Volatilidade:** diferença percentual entre a previsão GARCH 1-dia e a volatilidade realizada (EWMA de 5 dias).  
    - > +10%: GARCH está superestimando o risco
    - < -10%: GARCH está subestimando o risco
    """)

st.divider()

# ============================================================
# GARCH(1,1) + EWMA
# ============================================================
st.subheader("Volatilidade Condicional — GARCH(1,1) vs EWMA")
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

# AJUSTE PROMPT — v5.0: RMSE rolling GARCH vs EWMA
with st.spinner("Calculando RMSE rolling de previsão de volatilidade..."):
    rmse_result = vol_mod.rolling_volatility_rmse(close, lookback=min(lookback, 252), forecast_horizon=1)

if rmse_result["n"] > 0:
    st.caption(
        f"**RMSE de previsão 1-dia à frente** (últimos {rmse_result['n']} pregões): "
        f"GARCH ≈ {rmse_result['garch_rmse']:.2%} · EWMA ≈ {rmse_result['ewma_rmse']:.2%}"
    )
    if rmse_result["ewma_rmse"] and rmse_result["garch_rmse"]:
        vencedor = "EWMA" if rmse_result["ewma_rmse"] < rmse_result["garch_rmse"] else "GARCH"
        st.success(f"🏆 Melhor previsão de volatilidade nesta janela: **{vencedor}**")

with st.expander("📖 Metodologia — GARCH vs EWMA"):
    st.markdown("""
    **GARCH(1,1):** modela a volatilidade como processo autoregressivo com memória longa.  
    **EWMA:** média móvel exponencial dos retornos ao quadrado — mais reativa a choques recentes.

    **RMSE Rolling:** para cada dia, calculamos a previsão de volatilidade 1-dia à frente e comparamos com a volatilidade realizada (proxy: |retorno diário| × √252). O modelo com menor RMSE tem melhor poder preditivo genuíno.
    """)

st.divider()

# ============================================================
# Comparação de modelos de tendência — walk-forward validation
# ============================================================
st.subheader("Comparação de Modelos — Walk-Forward Validation + Diebold-Mariano")
st.caption(
    "Cada modelo é reajustado em janelas móveis e avaliado fora da amostra (1 passo à frente), "
    "reportando MAE, RMSE e teste de Diebold-Mariano para significância estatística."
)

n_folds = st.slider("Número de folds (walk-forward)", 5, 40, 30)
train_window = st.slider("Janela de treino por fold (pregões)", 60, 252, 120, step=10)

if st.button("Rodar Walk-Forward Validation + Diebold-Mariano", type="primary"):
    log_close = np.log(close.values)
    results = {name: {"mae": [], "rmse": [], "errors": []} for name in fc.MODEL_REGISTRY}
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
            results[name]["errors"].append(err)
        progress.progress((i + 1) / len(fold_starts), text=f"Fold {i+1}/{len(fold_starts)}")
    progress.empty()

    # Tabela resumo MAE/RMSE
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

    # AJUSTE PROMPT — v5.0: Teste de Diebold-Mariano
    st.subheader("Teste de Diebold-Mariano (Significância Estatística)")
    model_names = list(results.keys())
    dm_pairs = []
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            m1, m2 = model_names[i], model_names[j]
            e1 = np.array(results[m1]["errors"])
            e2 = np.array(results[m2]["errors"])
            # Loss differential: squared error
            d = e1**2 - e2**2
            if len(d) > 1 and np.std(d) > 0:
                t_stat = np.mean(d) / (np.std(d, ddof=1) / np.sqrt(len(d)))
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(d) - 1))
                dm_pairs.append({
                    "Par": f"{m1} vs {m2}",
                    "t-stat": t_stat,
                    "p-valor": p_value,
                    "Significativo (α=5%)": "Sim" if p_value < 0.05 else "Não",
                    "Melhor": m1 if np.mean(d) < 0 else m2,
                })
            else:
                dm_pairs.append({
                    "Par": f"{m1} vs {m2}",
                    "t-stat": np.nan,
                    "p-valor": np.nan,
                    "Significativo (α=5%)": "N/A",
                    "Melhor": "-",
                })

    if dm_pairs:
        df_dm = pd.DataFrame(dm_pairs)
        st.dataframe(df_dm.style.format({"t-stat": "{:.3f}", "p-valor": "{:.3f}"}), use_container_width=True, hide_index=True)
        st.caption("**Interpretação:** p-valor < 0.05 indica que a diferença de acurácia entre os dois modelos é estatisticamente significativa (não é devida ao acaso).")

else:
    st.info("Configure os parâmetros e clique em **Rodar Walk-Forward Validation + Diebold-Mariano**.")

with st.expander("📖 Metodologia — Walk-Forward e Diebold-Mariano"):
    st.markdown("""
    **Walk-Forward Validation:** simula o uso prático do modelo — treina numa janela, prevê o próximo ponto, desliza e repete. Evita o otimismo de ajustar e testar no mesmo período.

    **Teste de Diebold-Mariano:** compara dois modelos de previsão testando se a diferença entre suas funções de perda (erro quadrático) é estatisticamente diferente de zero.
    - H0: os modelos têm a mesma acurácia preditiva
    - Se p < 0.05: rejeitamos H0 — o modelo "Melhor" é significativamente superior
    """)

st.divider()
st.caption(
    "Roadmap de expansão deste módulo: VAR/VECM/Cointegração, Kalman Filter, Hidden Markov "
    "Models, XGBoost/LightGBM/CatBoost, LSTM/GRU/Temporal Fusion Transformer, SHAP values — "
    "ver README para priorização e requisitos de dados."
)
