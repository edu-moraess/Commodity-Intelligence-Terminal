"""
Forecasting — Calibração de Jump Diffusion (Merton)
====================================================
Estima lambda, mu_j, sigma_j a partir de retornos históricos.

Método:
  1. Identifica candidatos a jump via threshold (|r| > k * sigma_robust)
  2. Estima intensidade lambda = n_jumps / n_obs
  3. Estima mu_j e sigma_j a partir dos retornos classificados como jump
  4. Ajusta o componente difusivo residual

Referência: Cont & Tankov (2004), Merton (1976).
"""

from __future__ import annotations

import numpy as np
from scipy import stats as _stats


def calibrate_jump_diffusion(
    rets: np.ndarray,
    k_threshold: float = 3.0,
    min_jumps: int = 3,
) -> dict:
    """Calibra parâmetros Merton Jump-Diffusion a partir de retornos diários.

    Parameters
    ----------
    rets : array de retornos log
    k_threshold : múltiplo do desvio robusto para classificar jump
    min_jumps : mínimo de observações para estimar mu_j/sigma_j

    Returns
    -------
    dict com lambda, jump_mu, jump_sigma, mu_diffusive, sigma_diffusive, n_jumps
    """
    rets = np.asarray(rets, dtype=float)
    rets = rets[np.isfinite(rets)]
    n = len(rets)
    if n < 30:
        return {
            "jump_lambda": 0.05,
            "jump_mu": -0.02,
            "jump_sigma": 0.04,
            "mu": float(np.mean(rets)) if n > 0 else 0.0,
            "sigma": float(np.std(rets, ddof=1)) if n > 1 else 0.02,
            "n_jumps": 0,
            "method": "fallback_defaults",
        }

    # desvio robusto (MAD-based)
    med = np.median(rets)
    mad = np.median(np.abs(rets - med)) * 1.4826
    sigma_rob = mad if mad > 1e-8 else float(np.std(rets, ddof=1))

    threshold = k_threshold * sigma_rob
    jump_mask = np.abs(rets - med) > threshold
    n_jumps = int(jump_mask.sum())

    # intensidade (por dia)
    lam = n_jumps / n

    if n_jumps >= min_jumps:
        jump_rets = rets[jump_mask]
        jump_mu = float(np.mean(jump_rets))
        jump_sigma = float(np.std(jump_rets, ddof=1)) if n_jumps > 1 else abs(jump_mu) * 0.5
        jump_sigma = max(jump_sigma, 1e-4)
    else:
        # poucos jumps → priors conservadores
        lam = max(lam, 0.02)
        jump_mu = -0.015
        jump_sigma = 0.04

    # componente difusivo (retornos sem jumps)
    diff_rets = rets[~jump_mask]
    if len(diff_rets) > 5:
        mu_d = float(np.mean(diff_rets))
        sigma_d = float(np.std(diff_rets, ddof=1))
    else:
        mu_d = float(np.mean(rets))
        sigma_d = float(np.std(rets, ddof=1))

    # bounds de sanidade para commodities diárias
    lam = float(np.clip(lam, 0.005, 0.40))
    jump_sigma = float(np.clip(jump_sigma, 0.01, 0.25))
    sigma_d = float(max(sigma_d, 1e-4))

    return {
        "jump_lambda": lam,
        "jump_mu": jump_mu,
        "jump_sigma": jump_sigma,
        "mu": mu_d,
        "sigma": sigma_d,
        "n_jumps": n_jumps,
        "n_obs": n,
        "threshold": float(threshold),
        "method": "threshold_mad",
    }


def path_dependent_barrier_probs(
    paths: np.ndarray,
    support: float,
    resistance: float,
    sma20: float | None = None,
    bb_upper: float | None = None,
    bb_lower: float | None = None,
) -> dict:
    """Probabilidades de rompimento path-dependent (toca barreira em qualquer t).

    paths : (n_sims, horizon)
    """
    n_sims = paths.shape[0]
    # min/max ao longo do caminho
    path_min = paths.min(axis=1)
    path_max = paths.max(axis=1)

    out = {
        "prob_rompe_suporte_path": float(np.mean(path_min < support)),
        "prob_rompe_resistencia_path": float(np.mean(path_max > resistance)),
        "prob_rompe_suporte_terminal": float(np.mean(paths[:, -1] < support)),
        "prob_rompe_resistencia_terminal": float(np.mean(paths[:, -1] > resistance)),
    }

    if sma20 is not None:
        out["prob_acima_sma20_path"] = float(np.mean(path_max > sma20))
        out["prob_abaixo_sma20_path"] = float(np.mean(path_min < sma20))
    if bb_upper is not None:
        out["prob_acima_bb_upper_path"] = float(np.mean(path_max > bb_upper))
    if bb_lower is not None:
        out["prob_abaixo_bb_lower_path"] = float(np.mean(path_min < bb_lower))

    return out
