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
    mu = np.zeros(k)
    sigma = np.zeros(k)
    for i in range(k):
        mask = (abs_x >= quantiles[i]) & (abs_x < quantiles[i + 1] + 1e-12)
        if mask.sum() < 5:
            mask = np.argsort(abs_x)[i * (n // k):(i + 1) * (n // k)]
            mu[i] = x[mask].mean()
            sigma[i] = x[mask].std(ddof=1) + 1e-6
        else:
            mu[i] = x[mask].mean()
            sigma[i] = x[mask].std(ddof=1) + 1e-6

    order = np.argsort(sigma)
    mu, sigma = mu[order], sigma[order]

    A = np.full((k, k), 0.05 / max(k - 1, 1))
    np.fill_diagonal(A, 0.95)
    A = A / A.sum(axis=1, keepdims=True)
    pi0 = np.ones(k) / k

    prev_loglik = -np.inf
    gamma = np.zeros((n, k))

    for it in range(n_iter):
        B = np.column_stack([_gaussian_pdf(x, mu[j], sigma[j]) for j in range(k)])
        B = np.maximum(B, 1e-300)

        alpha = np.zeros((n, k))
        c = np.zeros(n)
        alpha[0] = pi0 * B[0]
        c[0] = alpha[0].sum() or 1e-300
        alpha[0] /= c[0]
        for t in range(1, n):
            alpha[t] = (alpha[t - 1] @ A) * B[t]
            c[t] = alpha[t].sum() or 1e-300
            alpha[t] /= c[t]

        beta = np.zeros((n, k))
        beta[-1] = 1.0
        for t in range(n - 2, -1, -1):
            beta[t] = (A @ (B[t + 1] * beta[t + 1])) / c[t + 1]

        gamma = alpha * beta
        gamma_sum = gamma.sum(axis=1, keepdims=True)
        gamma = np.where(gamma_sum > 0, gamma / gamma_sum, 1.0 / k)

        xi_sum = np.zeros((k, k))
        for t in range(n - 1):
            denom = c[t + 1]
            xi_t = np.outer(alpha[t], B[t + 1] * beta[t + 1]) * A / denom
            xi_sum += xi_t

        pi0 = gamma[0].copy()
        row_sums = xi_sum.sum(axis=1, keepdims=True)
        A_new = np.where(row_sums > 0, xi_sum / row_sums, A)
        gamma_col = gamma.sum(axis=0)
        mu_new = (gamma * x[:, None]).sum(axis=0) / np.maximum(gamma_col, 1e-12)
        sigma_new = np.sqrt((gamma * (x[:, None] - mu_new) ** 2).sum(axis=0) / np.maximum(gamma_col, 1e-12))
        sigma_new = np.maximum(sigma_new, 1e-6)

        order = np.argsort(sigma_new)
        mu_new, sigma_new = mu_new[order], sigma_new[order]
        A_new = A_new[order][:, order]
        gamma = gamma[:, order]
        pi0 = pi0[order]

        loglik = float(np.sum(np.log(c)))
        A, mu, sigma = A_new, mu_new, sigma_new
        if abs(loglik - prev_loglik) < tol:
            prev_loglik = loglik
            break
        prev_loglik = loglik

    B = np.column_stack([_gaussian_pdf(x, mu[j], sigma[j]) for j in range(k)])
    B = np.maximum(B, 1e-300)
    log_A = np.log(np.maximum(A, 1e-300))
    log_B = np.log(B)
    log_pi0 = np.log(np.maximum(pi0, 1e-300))

    delta = np.zeros((n, k))
    psi = np.zeros((n, k), dtype=int)
    delta[0] = log_pi0 + log_B[0]
    for t in range(1, n):
        for j in range(k):
            scores = delta[t - 1] + log_A[:, j]
            psi[t, j] = int(np.argmax(scores))
            delta[t, j] = scores[psi[t, j]] + log_B[t, j]

    states = np.zeros(n, dtype=int)
    states[-1] = int(np.argmax(delta[-1]))
    for t in range(n - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]

    n_params = k + k + k * (k - 1)
    aic = 2 * n_params - 2 * prev_loglik
    bic = np.log(n) * n_params - 2 * prev_loglik

    expected_duration = 1.0 / np.maximum(1.0 - np.diag(A), 1e-8)
    last_gamma = gamma[-1]
    next_regime_prob = last_gamma @ A

    return {
        "n_states": k,
        "transition_matrix": A,
        "means": mu,
        "stds": sigma,
        "state_probs": gamma,
        "viterbi_states": states,
        "log_likelihood": prev_loglik,
        "aic": float(aic),
        "bic": float(bic),
        "expected_duration": expected_duration,
        "next_regime_prob": next_regime_prob,
        "n_iter_used": it + 1,
    }


def fit_hmm_2state(returns: np.ndarray, n_iter: int = 100, tol: float = 1e-6, seed: int = 42) -> dict:
    """Wrapper de compatibilidade com a API v1."""
    return fit_hmm(returns, n_states=2, n_iter=n_iter, tol=tol, seed=seed)


def select_best_hmm(
    returns: np.ndarray,
    max_states: int = 4,
    criterion: str = "bic",
) -> dict:
    """Seleciona automaticamente o número de estados (2..max_states) por AIC/BIC/LL."""
    results = []
    for k in range(2, max_states + 1):
        try:
            fit = fit_hmm(returns, n_states=k)
            results.append(fit)
        except Exception:
            continue
    if not results:
        return fit_hmm(returns, n_states=2)

    if criterion == "aic":
        best = min(results, key=lambda r: r["aic"])
    elif criterion == "log_likelihood":
        best = max(results, key=lambda r: r["log_likelihood"])
    else:
        best = min(results, key=lambda r: r["bic"])

    best["candidates"] = [
        {"n_states": r["n_states"], "aic": r["aic"], "bic": r["bic"], "ll": r["log_likelihood"]}
        for r in results
    ]
    return best


def regime_summary(close: pd.Series, n_states: int = 2, n_iter: int = 100, auto_select: bool = False) -> dict:
    """Wrapper de conveniência: retorna tudo indexado por data, pronto para plot."""
    rets = daily_returns(close).dropna()
    if auto_select:
        result = select_best_hmm(rets.values, max_states=4)
        n_states = result["n_states"]
    else:
        result = fit_hmm(rets.values, n_states=n_states, n_iter=n_iter)

    idx = rets.index
    k = result["n_states"]
    cols = [f"prob_estado_{i}" for i in range(k)]
    if k == 2:
        cols = ["prob_baixa_vol", "prob_alta_vol"]
    elif k == 3:
        cols = ["prob_baixa_vol", "prob_media_vol", "prob_alta_vol"]

    state_probs_df = pd.DataFrame(result["state_probs"], index=idx, columns=cols[:k])
    viterbi = pd.Series(result["viterbi_states"], index=idx, name="regime")

    ann_factor = np.sqrt(252)
    labels = [f"Estado {i}" for i in range(k)]
    if k == 2:
        labels = ["Baixa Volatilidade", "Alta Volatilidade"]
    elif k == 3:
        labels = ["Baixa Volatilidade", "Média Volatilidade", "Alta Volatilidade"]

    regime_stats = pd.DataFrame({
        "Retorno médio diário": result["means"],
        "Volatilidade anualizada": result["stds"] * ann_factor,
        "Persistência": np.diag(result["transition_matrix"]),
        "Duração esperada (pregões)": result["expected_duration"],
    }, index=labels)

    current_regime = int(viterbi.iloc[-1])
    current_label = labels[current_regime]
    current_prob = float(state_probs_df.iloc[-1, current_regime])

    return {
        "n_states": k,
        "state_probs": state_probs_df,
        "viterbi_states": viterbi,
        "regime_stats": regime_stats,
        "transition_matrix": result["transition_matrix"],
        "current_regime_label": current_label,
        "current_regime_prob": current_prob,
        "next_regime_prob": result["next_regime_prob"],
        "expected_duration": result["expected_duration"],
        "log_likelihood": result["log_likelihood"],
        "aic": result.get("aic"),
        "bic": result.get("bic"),
        "labels": labels,
    }