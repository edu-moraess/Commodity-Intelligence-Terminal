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
    "Família GARCH, walk-forward de tendência e **density backtest** out-of-sample "
    "(CRPS / PIT / coverage) para ranking de métodos Monte Carlo."
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
            df_summary = pd.DataFrame()

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

# ============================================================
# FASE 4 — DENSITY BACKTEST (CRPS / PIT)
# ============================================================
st.divider()
st.subheader("Density Backtest — Ranking de Métodos MC (CRPS / PIT)")
st.caption(
    "Walk-forward densitário **sem look-ahead**: em cada fold, o MC usa apenas histórico até t "
    "e é avaliado no preço realizado em t+h. Menor Mean CRPS = melhor. "
    "PIT ~ U(0,1) indica calibração."
)

col_h, col_f, col_s = st.columns(3)
with col_h:
    dens_horizon = st.selectbox("Horizonte densitário (dias)", [1, 5, 10, 21], index=1)
with col_f:
    dens_folds = st.slider("Folds densitários", 5, 20, 10)
with col_s:
    dens_sims = st.select_slider("Sims por fold", options=[100, 200, 300, 500], value=300)

if st.button("Rodar Density Backtest (CRPS/PIT)", type="primary", key="dens_btn"):
    if not hasattr(fc, "walk_forward_density_backtest"):
        st.error("`walk_forward_density_backtest` não disponível — atualize forecasting/density_backtest.py.")
    else:
        with st.spinner("Walk-forward densitário em andamento (pode levar 1–2 min)..."):
            try:
                result = fc.walk_forward_density_backtest(
                    close,
                    horizon_days=int(dens_horizon),
                    n_folds=int(dens_folds),
                    min_train=max(120, train_window),
                    n_sims=int(dens_sims),
                    seed=42,
                )
            except Exception as exc:
                st.error(f"Falha no density backtest: {exc}")
                result = None

        if result is not None and not result["ranking"].empty:
            ranking = result["ranking"]
            st.success(
                f"Melhor método por Mean CRPS: **{result.get('best_method')}** "
                f"({result['n_folds']} folds · h={result['horizon_days']}d)"
            )

            fmt = {
                "Mean CRPS": "{:.4f}",
                "Median CRPS": "{:.4f}",
                "Coverage 80%": "{:.1%}",
                "Coverage 90%": "{:.1%}",
                "Avg Width 90%": "{:.2f}",
                "PIT mean": "{:.3f}",
                "PIT KS p-value": "{:.4f}",
            }
            st.dataframe(
                ranking.style.format(fmt, na_rep="—"),
                width="stretch",
                hide_index=True,
            )
            download_dataframe(ranking, filename_stem="density_backtest_ranking")

            # PIT diagnostics
            pit_map = result.get("pit_by_method", {})
            if pit_map:
                st.subheader("Diagnóstico PIT por método")
                pit_rows = []
                for m, d in pit_map.items():
                    pit_rows.append({
                        "Método": m,
                        "PIT mean": d.get("mean"),
                        "PIT std": d.get("std"),
                        "KS stat": d.get("ks_stat"),
                        "KS p-value": d.get("ks_pvalue"),
                        "Calibrado (p>0.05)": d.get("uniform_ok"),
                    })
                pit_df = pd.DataFrame(pit_rows)
                st.dataframe(
                    pit_df.style.format({
                        "PIT mean": "{:.3f}",
                        "PIT std": "{:.3f}",
                        "KS stat": "{:.3f}",
                        "KS p-value": "{:.4f}",
                    }, na_rep="—"),
                    width="stretch",
                    hide_index=True,
                )

            details = result.get("fold_details")
            if details is not None and not details.empty and "crps" in details.columns:
                st.subheader("CRPS por fold (detalhe)")
                pivot = details.pivot_table(index="fold", columns="method", values="crps", aggfunc="mean")
                st.dataframe(pivot.style.format("{:.4f}", na_rep="—"), width="stretch")
        elif result is not None:
            st.warning("Density backtest não produziu ranking (histórico insuficiente?).")
else:
    st.info("Clique em **Rodar Density Backtest** para rankear métodos MC por CRPS out-of-sample.")

st.divider()
st.caption(
    "Modelos de tendência: Linear, Ridge, Lasso, ElasticNet, RandomForest (+ boosting se instalado). "
    "Família GARCH requer `arch`. Density backtest: Gneiting & Raftery (2007), Diebold et al. (1998). "
    "Roadmap: Diebold-Mariano test, ensemble ponderado por CRPS, regime-conditional densities."
)
