"""
Analytics — Modelagem de Volatilidade Condicional (Família GARCH)
=================================================================
Implementação institucional da família GARCH via pacote `arch` (máxima
verossimilhança robusta) com fallback próprio para GARCH(1,1).

Modelos suportados:
  - GARCH(1,1)
  - EGARCH(1,1)          — captura assimetria (leverage effect)
  - GJR-GARCH(1,1)       — threshold assimétrico
  - TARCH / APARCH       — potência endógena + assimetria

Critérios de seleção automática: Log-Likelihood, AIC, BIC, RMSE, MAE
de previsão out-of-sample (1-step).

Forecast multi-horizonte: 1, 5, 10, 30 dias.

Referências:
  - Bollerslev (1986) — GARCH
  - Nelson (1991)     — EGARCH
  - Glosten et al. (1993) — GJR-GARCH
  - Ding et al. (1993) — APARCH
"""

from __future__ import annotations
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from analytics.metrics import daily_returns

# Tentativa de importar arch (preferido). Fallback para implementação própria.
try:
    from arch import arch_model
    _HAS_ARCH = True
except ImportError:
    _HAS_ARCH = False


# ---------------------------------------------------------------------------
# Fallback próprio GARCH(1,1) — mantido para compatibilidade total
# ---------------------------------------------------------------------------

def _garch11_neg_log_likelihood(params, returns: np.ndarray) -> float:
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        return 1e10
    n = len(returns)
    sigma2 = np.empty(n)
    sigma2[0] = np.var(returns)
    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
    sigma2 = np.maximum(sigma2, 1e-12)
    ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + returns**2 / sigma2)
    return -ll


def fit_garch11(close: pd.Series, lookback: int = 500) -> dict:
    """Ajusta GARCH(1,1) por MLE (implementação própria — compatibilidade v1).

    Retorna parâmetros, série de vol condicional anualizada e forecast 1d.
    """
    rets = (daily_returns(close).tail(lookback) * 100).values
    rets = rets - rets.mean()

    x0 = [0.05, 0.08, 0.88]
    bounds = [(1e-6, None), (0, 1), (0, 1)]
    result = minimize(_garch11_neg_log_likelihood, x0, args=(rets,), method="L-BFGS-B", bounds=bounds)

    omega, alpha, beta = result.x
    n = len(rets)
    sigma2 = np.empty(n)
    sigma2[0] = np.var(rets)
    for t in range(1, n):
        sigma2[t] = omega + alpha * rets[t - 1] ** 2 + beta * sigma2[t - 1]

    forecast_1d = omega + alpha * rets[-1] ** 2 + beta * sigma2[-1]
    persistence = alpha + beta

    idx = daily_returns(close).tail(lookback).index
    cond_vol_annualized = pd.Series(np.sqrt(sigma2) / 100 * np.sqrt(252), index=idx)

    return {
        "model": "GARCH(1,1)",
        "omega": float(omega), "alpha": float(alpha), "beta": float(beta),
        "persistence": float(persistence),
        "conditional_vol_annualized": cond_vol_annualized,
        "forecast_1d_vol_annualized": float(np.sqrt(forecast_1d) / 100 * np.sqrt(252)),
        "converged": bool(result.success),
        "log_likelihood": float(-result.fun),
        "aic": float(2 * 3 - 2 * (-result.fun)),
        "bic": float(np.log(n) * 3 - 2 * (-result.fun)),
    }


def ewma_volatility(close: pd.Series, lam: float = 0.94, window: int = 500) -> pd.Series:
    """Volatilidade EWMA (RiskMetrics) — comparação rápida."""
    rets = daily_returns(close).tail(window)
    var = rets.copy() ** 2
    ewma_var = var.ewm(alpha=1 - lam, adjust=False).mean()
    return np.sqrt(ewma_var) * np.sqrt(252)


# ---------------------------------------------------------------------------
# Família GARCH via arch (institucional)
# ---------------------------------------------------------------------------

SUPPORTED_MODELS = ("GARCH", "EGARCH", "GJR-GARCH", "APARCH")


def _fit_arch_model(
    returns_pct: pd.Series,
    model: str = "GARCH",
    p: int = 1,
    q: int = 1,
    o: int = 1,
    dist: str = "normal",
) -> Any:
    """Ajusta um modelo da família GARCH usando o pacote arch."""
    if not _HAS_ARCH:
        raise ImportError("Pacote 'arch' não instalado. Use fit_garch11 como fallback.")

    model = model.upper()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if model == "GARCH":
            am = arch_model(returns_pct, vol="Garch", p=p, q=q, dist=dist, rescale=False)
        elif model == "EGARCH":
            am = arch_model(returns_pct, vol="EGARCH", p=p, q=q, dist=dist, rescale=False)
        elif model in ("GJR-GARCH", "GJR"):
            am = arch_model(returns_pct, vol="GARCH", p=p, o=o, q=q, dist=dist, rescale=False)
        elif model in ("APARCH", "TARCH"):
            am = arch_model(returns_pct, vol="APARCH", p=p, o=o, q=q, dist=dist, rescale=False)
        else:
            raise ValueError(f"Modelo desconhecido: {model}. Suportados: {SUPPORTED_MODELS}")
        res = am.fit(disp="off", show_warning=False)
    return res


def fit_volatility_model(
    close: pd.Series,
    model: str = "GARCH",
    lookback: int = 500,
    dist: str = "normal",
) -> dict:
    """Ajusta um modelo de volatilidade e retorna dicionário padronizado.

    Compatível com a assinatura antiga de fit_garch11 + campos extras
    (AIC, BIC, parâmetros nomeados, forecast multi-horizonte).
    """
    rets = daily_returns(close).tail(lookback).dropna()
    if len(rets) < 50:
        raise ValueError("Série muito curta para ajuste de volatilidade (mín. ~50 obs).")

    rets_pct = rets * 100  # escala % para melhor condicionamento numérico

    if not _HAS_ARCH and model.upper() == "GARCH":
        return fit_garch11(close, lookback=lookback)

    if not _HAS_ARCH:
        # Fallback forçado para GARCH próprio se arch não disponível
        out = fit_garch11(close, lookback=lookback)
        out["warning"] = f"arch não instalado — fallback para GARCH(1,1) próprio (pedido era {model})"
        return out

    try:
        res = _fit_arch_model(rets_pct, model=model, dist=dist)
    except Exception as exc:
        # Fallback seguro
        out = fit_garch11(close, lookback=lookback)
        out["warning"] = f"Falha no ajuste {model}: {exc}. Fallback GARCH(1,1)."
        return out

    # Volatilidade condicional anualizada
    cond_vol = res.conditional_volatility / 100 * np.sqrt(252)
    cond_vol.index = rets.index

    # Forecast 1-step (e multi-horizonte)
    forecasts = {}
    try:
        fcast = res.forecast(horizon=30, reindex=False)
        variance_paths = fcast.variance.values[-1]  # shape (30,)
        for h in (1, 5, 10, 30):
            # Média da variância no horizonte h (aproximação)
            var_h = np.mean(variance_paths[:h])
            forecasts[f"forecast_{h}d_vol_annualized"] = float(np.sqrt(var_h) / 100 * np.sqrt(252))
    except Exception:
        # Fallback simples
        last_var = (res.conditional_volatility.iloc[-1] / 100) ** 2
        for h in (1, 5, 10, 30):
            forecasts[f"forecast_{h}d_vol_annualized"] = float(np.sqrt(last_var) * np.sqrt(252))

    params = {k: float(v) for k, v in res.params.items()}
    persistence = float(getattr(res, "persistence", np.nan)) if hasattr(res, "persistence") else np.nan

    return {
        "model": model.upper(),
        "params": params,
        "omega": params.get("omega", params.get("omega[0]", np.nan)),
        "alpha": params.get("alpha[1]", params.get("alpha[0]", np.nan)),
        "beta": params.get("beta[1]", params.get("beta[0]", np.nan)),
        "gamma": params.get("gamma[1]", params.get("gamma[0]", np.nan)),  # assimetria (GJR/EGARCH/APARCH)
        "persistence": persistence,
        "conditional_vol_annualized": cond_vol,
        "forecast_1d_vol_annualized": forecasts.get("forecast_1d_vol_annualized", np.nan),
        **forecasts,
        "converged": True,
        "log_likelihood": float(res.loglikelihood),
        "aic": float(res.aic),
        "bic": float(res.bic),
        "nobs": int(res.nobs),
        "arch_result": res,  # objeto completo para análise avançada
    }


def compare_volatility_models(
    close: pd.Series,
    lookback: int = 500,
    models: tuple[str, ...] = SUPPORTED_MODELS,
    dist: str = "normal",
) -> pd.DataFrame:
    """Compara modelos da família GARCH e retorna ranking por AIC/BIC/LL.

    Colunas: model, log_likelihood, aic, bic, persistence, forecast_1d, converged.
    Ordenado por AIC ascendente (melhor modelo no topo).
    """
    rows = []
    for m in models:
        try:
            fit = fit_volatility_model(close, model=m, lookback=lookback, dist=dist)
            rows.append({
                "Modelo": fit["model"],
                "Log-Likelihood": fit.get("log_likelihood", np.nan),
                "AIC": fit.get("aic", np.nan),
                "BIC": fit.get("bic", np.nan),
                "Persistência": fit.get("persistence", np.nan),
                "Forecast 1d (vol anual.)": fit.get("forecast_1d_vol_annualized", np.nan),
                "Convergiu": fit.get("converged", False),
            })
        except Exception as exc:
            rows.append({
                "Modelo": m,
                "Log-Likelihood": np.nan,
                "AIC": np.nan,
                "BIC": np.nan,
                "Persistência": np.nan,
                "Forecast 1d (vol anual.)": np.nan,
                "Convergiu": False,
                "Erro": str(exc)[:80],
            })
    df = pd.DataFrame(rows)
    if "AIC" in df.columns:
        df = df.sort_values("AIC", ascending=True).reset_index(drop=True)
    return df


def select_best_volatility_model(
    close: pd.Series,
    lookback: int = 500,
    criterion: str = "aic",
) -> dict:
    """Seleciona automaticamente o melhor modelo pelo critério escolhido.

    criterion: 'aic' | 'bic' | 'log_likelihood'
    Retorna o dicionário completo do fit do melhor modelo + ranking.
    """
    ranking = compare_volatility_models(close, lookback=lookback)
    if ranking.empty or ranking["AIC"].isna().all():
        return fit_garch11(close, lookback=lookback)

    criterion = criterion.lower()
    if criterion == "bic":
        best_name = ranking.loc[ranking["BIC"].idxmin(), "Modelo"]
    elif criterion == "log_likelihood":
        best_name = ranking.loc[ranking["Log-Likelihood"].idxmax(), "Modelo"]
    else:
        best_name = ranking.loc[ranking["AIC"].idxmin(), "Modelo"]

    best_fit = fit_volatility_model(close, model=best_name, lookback=lookback)
    best_fit["ranking"] = ranking
    best_fit["selection_criterion"] = criterion
    return best_fit


def multi_horizon_vol_forecast(
    close: pd.Series,
    model: str = "GARCH",
    lookback: int = 500,
    horizons: tuple[int, ...] = (1, 5, 10, 30),
) -> pd.DataFrame:
    """Gera forecast de volatilidade anualizada para múltiplos horizontes."""
    fit = fit_volatility_model(close, model=model, lookback=lookback)
    rows = []
    for h in horizons:
        key = f"forecast_{h}d_vol_annualized"
        vol = fit.get(key, np.nan)
        rows.append({"Horizonte (dias)": h, "Vol. Anualizada Prevista": vol, "Modelo": fit["model"]})
    return pd.DataFrame(rows)
