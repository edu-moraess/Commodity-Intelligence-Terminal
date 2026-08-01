"""
Analytics — Detecção de Regimes via Hidden Markov Model (multi-estado)
======================================================================
HMM Gaussiano de 2–4 estados implementado via Baum-Welch + Viterbi.
Seleção automática do número de estados por AIC / BIC / Log-Likelihood.

Compatível com a API v1 (fit_hmm_2state + regime_summary).

Referência: Hamilton, J. D. (1989). Econometrica.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any

from analytics.metrics import daily_returns


def _gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = max(sigma, 1e-8)
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def fit_hmm(
    returns: np.ndarray,
    n_states: int = 2,
    n_iter: int = 120,
    tol: float = 1e-6,
    seed: int = 42,
) -> dict:
    """Ajusta HMM Gaussiano de n_states estados via Baum-Welch (EM)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(returns, dtype=float)
    n = len(x)
    if n < 40:
        raise ValueError("Série muito curta para HMM (mínimo \~40 observações).")
    k = n_states

    abs_x = np.abs(x - x.mean())
    quantiles = np.quantile(abs_x, np.linspace(0, 1, k + 1))
    mu = np