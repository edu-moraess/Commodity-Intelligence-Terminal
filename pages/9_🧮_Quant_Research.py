import streamlit as st
import numpy as np
import pandas as pd

from config.settings import ALL_ASSETS, APP_NAME
from data.data_manager import load_price_history
from analytics import volatility as vol_mod
from analytics import cached
from forecasting import models as fc
from charts import plotly_charts as charts
from utils.export import download_dataframe

st.title("🧮 Quant Research")
st.caption(
    "Família GARCH (GARCH / EGARCH / GJR / APARCH), seleção automática por AIC/BIC, "
    "forecast multi-horizonte e comparação de modelos de tendência via walk-forward."
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
# CONTROLES GARCH
# ============================================================
st.subheader("Volatilidade Condicional — Família GARCH")

col_lb, col_model, col_crit = st.columns(3)
with col_lb:
    lookback = st.slider("Janela de estimação (pregões)", 250, 1000, 500, step=50)
with col_model:
    model_options = ["Auto (melhor AIC)", "GARCH", "EGARCH", "GJR-GARCH", "APARCH"]
    model_choice = st.selectbox("Modelo", model_options, index=0)
with col_crit:
    criterion = st.selectbox("Critério de seleção", ["aic", "bic", "log_likelihood"], index=0)

with st.spinner("Ajustando modelo de volatilidade..."):
    try:
        if model_choice.startswith("Auto"):
            if hasattr(vol_mod, "select_best_volatility_model"):
                fit = cached.cached_select_best_volatility_model(close, lookback=lookback, criterion=criterion)
            else:
                fit = cached.cached_fit_garch11(close, lookback=lookback)
        else:
            if hasattr(vol_mod, "fit_volatility_model"):
                fit = cached.cached_fit_volatility_model(close, model=model_choice, lookback=lookback)
            else:
                fit = cached.cached_fit_garch11(close, lookback=lookback)
        fit_ok = True
        fit_err = None
    except Exception as exc:
        fit_ok = False
        fit_err = str(exc)
        fit = None

if not fit_ok:
    st.error(f"Falha no ajuste: {fit_err}")
else:
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Modelo", fit.get("model", "GARCH(1,1)"))
    g2.metric("ω (omega)", f"{fit.get('omega', float('nan')):.4f}" if pd.notna(fit.get("omega")) else "—")
    g3.metric("α (alpha)", f"{fit.get('alpha', float('nan')):.4f}" if pd.notna(fit.get("alpha")) else "—")
    g4.metric("β (beta)", f"{fit.get('beta', float('nan')):.4f}" if pd.notna(fit.get("beta")) else "—")

    g5, g6, g7, g8 = st.columns(4)
    gamma = fit.get("gamma", np.nan)
    g5.metric("γ (assimetria)", f"{gamma:.4f}" if pd.notna(gamma) else "—")
    g6.metric("Persistência", f"{fit.get('persistence', float('nan')):.4f}" if pd.notna(fit.get("persistence")) else "—")
    g7.metric("Log-Likelihood", f"{fit.get('log_likelihood', float('nan')):.1f}" if pd.notna(fit.get("log_likelihood")) else "—")
    aic_val = fit.get("aic", np.nan)
    g8.metric("AIC", f"{aic_val:.1f}" if pd.notna(aic_val) else "—")

    conv = fit.get("converged", True)
    f1d = fit.get("forecast_1d_vol_annualized", np.nan)
    st.caption(
        f"Convergência: {'✅ sim' if conv else '⚠️ não'} · "
        f"Previsão vol. anualizada 1d: **{f1d:.2%}**" if pd.notna(f1d) else
        f"Convergência: {'✅ sim' if conv else '⚠️ não'}"
    )
    if fit.get("warning"):
        st.warning(fit["warning"])

    ewma = vol_mod.ewma_volatility(close, window=lookback)
    series_dict = {}
    if "conditional_vol_annualized" in fit and fit["conditional_vol_annualized"] is not None:
        series_dict[fit.get("model", "GARCH")] = fit["conditional_vol_annualized"]
    series_dict["EWMA (λ=0.94)"] = ewma

    st.plotly_chart(
        charts.line_chart(series_dict, title="Volatilidade Condicional Anualizada", y_title="%"),
        width="stretch",
    )

    st.subheader("Forecast de Volatilidade Multi-Horizonte")
    horizons = [1, 5, 10, 30]
    fh_rows = []
    for h in horizons:
        key = f"forecast_{h}d_vol_annualized"
        val = fit.get(key, np.nan)
        fh_rows.append({"Horizonte (dias)": h, "Vol. Anualizada Prevista": val})
    fh_df = pd.DataFrame(fh_rows)
    st.dataframe(
        fh_df.style.format({"Vol. Anualizada Prevista": "{:.2%}"}),
        width="stretch",
        hide_index=True,
    )
    if fh_df["Vol. Anualizada Prevista"].notna().any():
        st.plotly_chart(
            charts.bar_chart(
                [str(h) for h in horizons],
                fh_df["Vol. Anualizada Prevista"].fillna(0).tolist(),
                title="Vol Prevista por Horizonte",
                positive_negative=False,
            ),
            width="stretch",
        )

# ============================================================
# COMPARAÇÃO DE MODELOS GARCH
# ============================================================
st.divider()
st.subheader("Comparação de Modelos — Família GARCH")
st.caption("Ranking por AIC / BIC / Log-Likelihood. Menor AIC/BIC = melhor. Maior LL = melhor.")

if st.button("Comparar GARCH / EGARCH / GJR / APARCH", type="primary"):
    if hasattr(vol_mod, "compare_volatility_models"):
        with st.spinner("Ajustando todos os modelos..."):
            ranking = cached.cached_compare_volatility_models(close, lookback=lookback)
        if ranking is not None and not ranking.empty:
            st.dataframe(
                ranking.style.format({
                    "Log-Likelihood": "{:.2f}",
                    "AIC": "{:.2f}",
                    "BIC": "{:.2f}",
                    "Persistência": "{:.4f}",
                    "Forecast 1d (vol anual.)": "{:.2%}",
                }, na_rep="—"),
                width="stretch",
            )
            best_name = ranking.iloc[0]["Modelo"]
            st.success(f"Melhor modelo por AIC: **{best_name}**")
        else:
            st.warning("Não foi possível comparar os modelos (arch pode não estar instalado).")
    else:
        st.info("Função `compare_volatility_models` não disponível — atualize analytics/volatility.py.")

# ============================================================
# WALK-FORWARD DE MODELOS DE TENDÊNCIA
# ============================================================
st.divider()
st.subheader("Comparação de Modelos de Tendência — Walk-Forward Validation")
st.caption(
    "Cada modelo é reajustado em janelas móveis e avaliado fora da amostra (1 passo à frente). "
    "Métricas: MAE, RMSE, MAPE, R²."
)

n_folds = st.slider("Número de folds (walk-forward)", 5, 40, 15)
train_window = st.slider("Janela de treino por fold (pregões)", 60, 252, 120, step=10)

if st.button("Rodar Walk-Forward Validation", type="primary", key="wf_btn"):
    with st.spinner("Rodando walk-forward..."):
        if hasattr(fc, "walk_forward_validation"):
            df_summary = fc.walk_forward_validation(
                close, train_window=train_window, n_folds=n_folds
            )
        else:
            log_close = np.log(close.dropna().values)
            registry = getattr(fc, "MODEL_REGISTRY", {})
            results = {name: {"mae": [], "rmse": []} for name in registry}
            total_len = len(log_close)
            fold_starts = np.linspace(train_window, total_len - 2, n_folds, dtype=int)
            progress = st.progress(0.0, text="Rodando folds...")
            for i, start in enumerate(fold_starts):
                y_train = log_close[start - train_window:start]
                X_train = np.arange(train_window).reshape(-1, 1)
                y_true_next = log_close[start]
                X_next = np.array([[train_window]])
                for name, base_model in registry.items():
                    from sklearn.base import clone
                    try:
                        model = clone(base_model)
                        model.fit(X_train, y_train)
                        pred = model.predict(X_next)[0]
                        err = np.exp(pred) - np.exp(y_true_next)
                        results[name]["mae"].append(abs(err))
                        results[name]["rmse"].append(err ** 2)
                    except Exception:
                        continue
                progress.progress((i + 1) / len(fold_starts), text=f"Fold {i+1}/{len(fold_starts)}")
            progress.empty()
            summary = []
            for name, r in results.items():
                if not r["mae"]:
                    continue
                summary.append({
                    "Modelo": name,
                    "MAE (1-step)": np.mean(r["mae"]),
                    "RMSE (1-step)": np.sqrt(np.mean(r["rmse"])),
                })
            df_summary = pd.DataFrame(summary)
            if not df_summary.empty:
                df_summary = df_summary.sort_values("RMSE (1-step)").reset_index(drop=True)

    if df_summary is not None and not df_summary.empty:
        fmt = {}
        for c in df_summary.columns:
            if c != "Modelo" and c != "n_folds":
                fmt[c] = "{:.4f}"
        st.dataframe(df_summary.style.format(fmt, na_rep="—"), width="stretch", hide_index=True)
        download_dataframe(df_summary, filename_stem="quant_research_walkforward")
        best = df_summary.iloc[0]["Modelo"] if "Modelo" in df_summary.columns else df_summary.index[0]
        st.success(f"Melhor modelo por RMSE fora da amostra: **{best}**")
    else:
        st.warning("Nenhum modelo produziu resultados. Verifique histórico e dependências.")
else:
    st.info("Configure os parâmetros e clique em **Rodar Walk-Forward Validation**.")

st.divider()
st.caption(
    "Modelos de tendência disponíveis (quando instalados): Linear, Ridge, Lasso, ElasticNet, "
    "RandomForest, XGBoost, LightGBM, CatBoost. "
    "Família GARCH requer pacote `arch`. Roadmap: VAR/VECM, Kalman, LSTM/TFT, SHAP."
) 