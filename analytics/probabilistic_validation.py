"""
Analytics — Validação Probabilística de Forecasts
=================================================
Métricas formais para avaliar a qualidade das distribuições preditivas
geradas por Monte Carlo / ensemble.

Métricas implementadas:
  - PIT (Probability Integral Transform) + histograma / KS test
  - CRPS (Continuous Ranked Probability Score) — amostra empírica
  - Interval Coverage (P10–P90, P5–P95, etc.)
  - Calibration curve (reliability diagram simplificado)
  - Brier Score para eventos binários (P(Alta), rompimento)

Referências:
  - Gneiting & Raftery (2007) — Strictly Proper Scoring Rules
  - Diebold, Gunther & Tay (1998) — Evaluating Density Forecasts
  - Christoffersen (1998) — Evaluating Interval Forecasts
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as _stats
from typing import Sequence


def pit_values(realized: np.ndarray, forecast_samples: np.ndarray) -> np.ndarray:
    """Probability Integral Transform.

    Para cada observação i, PIT_i = F_hat_i(y_i) onde F_hat é a CDF empírica
    construída a partir das amostras preditivas (n_sims,).

    Parameters
    ----------
    realized : array (n_obs,)
    forecast_samples : array (n_obs, n_sims) ou (n_sims,) se n_obs=1

    Returns
    -------
    pit : array (n_obs,) ∈ [0, 1]
    """
    realized = np.asarray(realized, dtype=float).ravel()
    samples = np.asarray(forecast_samples, dtype=float)

    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
        if len(realized) == 1:
            pass
        else:
            # broadcast single predictive sample set to all observations (não ideal, mas defensivo)
            samples = np.repeat(samples, len(realized), axis=0)

    if samples.shape[0] != len(realized):
        raise ValueError(
            f"Dimensão incompatível: realized={len(realized)}, samples={samples.shape[0]}"
        )

    pit = np.array([
        np.mean(samples[i] <= realized[i]) for i in range(len(realized))
    ])
    # evita 0/1 exatos para testes (Rosenblatt)
    eps = 1e-6
    return np.clip(pit, eps, 1 - eps)


def pit_diagnostics(pit: np.ndarray) -> dict:
    """Diagnósticos sobre a série PIT.

    Sob H0 de calibração perfeita, PIT ~ U(0,1).
    """
    pit = np.asarray(pit, dtype=float)
    n = len(pit)
    if n < 5:
        return {"n": n, "ks_stat": np.nan, "ks_pvalue": np.nan, "mean": np.nan, "std": np.nan}

    ks_stat, ks_p = _stats.kstest(pit, "uniform")
    return {
        "n": n,
        "mean": float(np.mean(pit)),
        "std": float(np.std(pit, ddof=1)),
        "ks_stat": float(ks_stat),
        "ks_pvalue": float(ks_p),
        "uniform_ok": bool(ks_p > 0.05),
        "histogram_bins": np.histogram(pit, bins=10, range=(0, 1))[0].tolist(),
    }


def crps_empirical(realized: float | np.ndarray, samples: np.ndarray) -> float | np.ndarray:
    """Continuous Ranked Probability Score (amostra empírica).

    CRPS(F, y) = E|X - y| - 0.5 E|X - X'|
    onde X, X' ~ F independentes.

    Parameters
    ----------
    realized : escalar ou array (n_obs,)
    samples : (n_sims,) ou (n_obs, n_sims)

    Returns
    -------
    crps : escalar ou array (n_obs,)
    """
    realized = np.asarray(realized, dtype=float)
    samples = np.asarray(samples, dtype=float)

    if samples.ndim == 1:
        y = float(realized) if realized.ndim == 0 else float(realized.ravel()[0])
        term1 = np.mean(np.abs(samples - y))
        # E|X-X'| via fórmula O(n log n) ou amostragem
        n = len(samples)
        if n > 2000:
            # aproximação por subamostra
            rng = np.random.default_rng(0)
            idx = rng.choice(n, size=min(1000, n), replace=False)
            s = samples[idx]
            term2 = 0.5 * np.mean(np.abs(s[:, None] - s[None, :]))
        else:
            term2 = 0.5 * np.mean(np.abs(samples[:, None] - samples[None, :]))
        return float(term1 - term2)

    # múltiplas observações
    n_obs, n_sims = samples.shape
    y = realized.ravel()
    if len(y) != n_obs:
        raise ValueError("realized e samples incompatíveis")

    crps_vals = np.empty(n_obs)
    for i in range(n_obs):
        s = samples[i]
        term1 = np.mean(np.abs(s - y[i]))
        if n_sims > 1500:
            rng = np.random.default_rng(i)
            idx = rng.choice(n_sims, size=800, replace=False)
            ss = s[idx]
            term2 = 0.5 * np.mean(np.abs(ss[:, None] - ss[None, :]))
        else:
            term2 = 0.5 * np.mean(np.abs(s[:, None] - s[None, :]))
        crps_vals[i] = term1 - term2
    return crps_vals


def interval_coverage(
    realized: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    nominal_level: float = 0.90,
) -> dict:
    """Taxa de cobertura empírica de intervalos preditivos.

    Parameters
    ----------
    realized, lower, upper : arrays (n_obs,)
    nominal_level : nível nominal (ex: 0.90 para P5–P95)
    """
    realized = np.asarray(realized, dtype=float).ravel()
    lower = np.asarray(lower, dtype=float).ravel()
    upper = np.asarray(upper, dtype=float).ravel()
    n = len(realized)
    if n == 0:
        return {"n": 0, "coverage": np.nan, "nominal": nominal_level}

    inside = (realized >= lower) & (realized <= upper)
    cov = float(np.mean(inside))
    # teste binomial simples H0: coverage = nominal
    from scipy.stats import binomtest
    try:
        bt = binomtest(int(inside.sum()), n, nominal_level, alternative="two-sided")
        p_value = float(bt.pvalue)
    except Exception:
        p_value = np.nan

    return {
        "n": n,
        "coverage": cov,
        "nominal": nominal_level,
        "n_inside": int(inside.sum()),
        "avg_width": float(np.mean(upper - lower)),
        "p_value_binomial": p_value,
        "well_calibrated": bool(p_value > 0.05) if np.isfinite(p_value) else None,
    }


def brier_score(events: np.ndarray, probs: np.ndarray) -> float:
    """Brier Score para eventos binários.

    BS = mean( (p - o)^2 )  — quanto menor, melhor (0 = perfeito).
    """
    events = np.asarray(events, dtype=float).ravel()
    probs = np.asarray(probs, dtype=float).ravel()
    if len(events) != len(probs):
        raise ValueError("events e probs devem ter o mesmo comprimento")
    return float(np.mean((probs - events) ** 2))


def calibration_curve(
    events: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Reliability diagram simplificado (calibração de probabilidade).

    Agrupa previsões em bins e compara frequência observada vs probabilidade média.
    """
    events = np.asarray(events, dtype=float).ravel()
    probs = np.asarray(probs, dtype=float).ravel()
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1] if i < n_bins - 1 else probs <= bins[i + 1])
        if mask.sum() == 0:
            continue
        rows.append({
            "bin_left": bins[i],
            "bin_right": bins[i + 1],
            "n": int(mask.sum()),
            "mean_prob": float(np.mean(probs[mask])),
            "obs_freq": float(np.mean(events[mask])),
        })
    return pd.DataFrame(rows)


def evaluate_density_forecast(
    realized: np.ndarray,
    forecast_samples: np.ndarray,
    quantiles: Sequence[float] = (0.05, 0.10, 0.50, 0.90, 0.95),
) -> dict:
    """Pipeline completo de validação densitária.

    Parameters
    ----------
    realized : (n_obs,)
    forecast_samples : (n_obs, n_sims)

    Returns
    -------
    dict com PIT, CRPS médio, coverage de intervalos, etc.
    """
    realized = np.asarray(realized, dtype=float).ravel()
    samples = np.asarray(forecast_samples, dtype=float)
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)

    pit = pit_values(realized, samples)
    pit_diag = pit_diagnostics(pit)

    crps = crps_empirical(realized, samples)
    mean_crps = float(np.mean(crps)) if np.ndim(crps) > 0 else float(crps)

    # cobertura P10–P90 e P5–P95
    q10 = np.percentile(samples, 10, axis=1)
    q90 = np.percentile(samples, 90, axis=1)
    q05 = np.percentile(samples, 5, axis=1)
    q95 = np.percentile(samples, 95, axis=1)

    cov_80 = interval_coverage(realized, q10, q90, nominal_level=0.80)
    cov_90 = interval_coverage(realized, q05, q95, nominal_level=0.90)

    return {
        "n_obs": len(realized),
        "pit": pit_diag,
        "mean_crps": mean_crps,
        "coverage_80": cov_80,
        "coverage_90": cov_90,
        "quantile_levels": list(quantiles),
    }
