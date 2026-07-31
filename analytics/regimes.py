"""
Analytics — Detecção de Regimes via Hidden Markov Model (2 estados)
=======================================================================
HMM Gaussiano de 2 estados (Low-Vol / High-Vol) implementado do zero via
algoritmo Baum-Welch (EM) + Viterbi, sem depender de `hmmlearn` — mesma
filosofia do GARCH(1,1) próprio em `analytics/volatility.py`: reduzir
dependências externas mantendo rigor estatístico.

Uso típico em risk management: regimes de alta volatilidade concentram
a maior parte das perdas de cauda — identificá-los ajuda a interpretar
por que o VaR falha mais em certos períodos (ver `analytics/backtesting.py`).

Referência: Hamilton, J. D. (1989). A New Approach to the Economic
Analysis of Nonstationary Time Series and the Business Cycle.
Econometrica, 57(2), 357-384.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from analytics.metrics import daily_returns


def _gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = max(sigma, 1e-8)
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def fit_hmm_2state(returns: np.ndarray, n_iter: int = 100, tol: float = 1e-6,
                    seed: int = 42) -> dict:
    """Ajusta um HMM Gaussiano de 2 estados via Baum-Welch (EM).

    Inicialização: estado 0 = baixa volatilidade (menor std), estado 1 =
    alta volatilidade (maior std) — reordenado ao final para garantir essa
    convenção independente de onde o EM convergiu.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(returns, dtype=float)
    n = len(x)
    if n < 30:
        raise ValueError("Série muito curta para ajustar HMM (mínimo ~30 observações).")

    # Inicialização: split simples pela mediana de |retorno| para dar um
    # chute inicial razoável de qual estado é "calmo" vs "volátil".
    abs_x = np.abs(x - x.mean())
    median_split = np.median(abs_x)
    low_mask = abs_x <= median_split

    mu = np.array([x[low_mask].mean(), x[~low_mask].mean()])
    sigma = np.array([x[low_mask].std(ddof=1) + 1e-6, x[~low_mask].std(ddof=1) + 1e-6])
    # Garante sigma[0] < sigma[1] (estado 0 = baixa vol)
    if sigma[0] > sigma[1]:
        mu, sigma = mu[::-1].copy(), sigma[::-1].copy()

    A = np.array([[0.95, 0.05], [0.05, 0.95]])  # matriz de transição inicial (regimes persistentes)
    pi0 = np.array([0.5, 0.5])                    # distribuição inicial

    prev_loglik = -np.inf

    for _ in range(n_iter):
        # -------- E-step: forward-backward --------
        B = np.column_stack([_gaussian_pdf(x, mu[k], sigma[k]) for k in range(2)])
        B = np.maximum(B, 1e-300)

        # Forward (com escalonamento para estabilidade numérica)
        alpha = np.zeros((n, 2))
        c = np.zeros(n)
        alpha[0] = pi0 * B[0]
        c[0] = alpha[0].sum()
        alpha[0] /= c[0]
        for t in range(1, n):
            alpha[t] = (alpha[t - 1] @ A) * B[t]
            c[t] = alpha[t].sum()
            if c[t] <= 0:
                c[t] = 1e-300
            alpha[t] /= c[t]

        # Backward
        beta = np.zeros((n, 2))
        beta[-1] = 1.0
        for t in range(n - 2, -1, -1):
            beta[t] = (A @ (B[t + 1] * beta[t + 1])) / c[t + 1]

        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)

        xi_sum = np.zeros((2, 2))
        for t in range(n - 1):
            denom = c[t + 1]
            xi_t = np.outer(alpha[t], B[t + 1] * beta[t + 1]) * A / denom
            xi_sum += xi_t

        # -------- M-step --------
        pi0 = gamma[0].copy()
        A_new = xi_sum / xi_sum.sum(axis=1, keepdims=True)
        gamma_sum = gamma.sum(axis=0)
        mu_new = (gamma * x[:, None]).sum(axis=0) / gamma_sum
        sigma_new = np.sqrt((gamma * (x[:, None] - mu_new) ** 2).sum(axis=0) / gamma_sum)
        sigma_new = np.maximum(sigma_new, 1e-6)

        # Reordena para manter convenção estado 0 = baixa vol
        if sigma_new[0] > sigma_new[1]:
            mu_new, sigma_new = mu_new[::-1].copy(), sigma_new[::-1].copy()
            A_new = A_new[::-1][:, ::-1].copy()
            gamma = gamma[:, ::-1].copy()

        loglik = float(np.sum(np.log(c)))
        A, mu, sigma = A_new, mu_new, sigma_new

        if abs(loglik - prev_loglik) < tol:
            prev_loglik = loglik
            break
        prev_loglik = loglik

    # -------- Viterbi (sequência de estados mais provável) --------
    B = np.column_stack([_gaussian_pdf(x, mu[k], sigma[k]) for k in range(2)])
    B = np.maximum(B, 1e-300)
    log_A = np.log(np.maximum(A, 1e-300))
    log_B = np.log(B)
    log_pi0 = np.log(np.maximum(pi0, 1e-300))

    delta = np.zeros((n, 2))
    psi = np.zeros((n, 2), dtype=int)
    delta[0] = log_pi0 + log_B[0]
    for t in range(1, n):
        for k in range(2):
            scores = delta[t - 1] + log_A[:, k]
            psi[t, k] = int(np.argmax(scores))
            delta[t, k] = scores[psi[t, k]] + log_B[t, k]

    states = np.zeros(n, dtype=int)
    states[-1] = int(np.argmax(delta[-1]))
    for t in range(n - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]

    return {
        "transition_matrix": A,          # [[P(0->0), P(0->1)], [P(1->0), P(1->1)]]
        "means": mu,                       # retorno médio diário por regime
        "stds": sigma,                     # volatilidade diária por regime
        "state_probs": gamma,             # P(estado=k | toda a série) — smoothed
        "viterbi_states": states,          # sequência mais provável (0=baixa vol, 1=alta vol)
        "log_likelihood": prev_loglik,
        "n_iter_used": n_iter,
    }


def regime_summary(close: pd.Series, n_iter: int = 100) -> dict:
    """Wrapper de conveniência: recebe preços, ajusta o HMM sobre os
    retornos diários e devolve tudo já indexado por data, pronto para plot."""
    rets = daily_returns(close)
    result = fit_hmm_2state(rets.values, n_iter=n_iter)

    idx = rets.index
    state_probs_df = pd.DataFrame(result["state_probs"], index=idx,
                                    columns=["prob_baixa_vol", "prob_alta_vol"])
    viterbi = pd.Series(result["viterbi_states"], index=idx, name="regime")

    ann_factor = np.sqrt(252)
    regime_stats = pd.DataFrame({
        "Retorno médio diário": result["means"],
        "Volatilidade anualizada": result["stds"] * ann_factor,
        "Persistência (prob. de permanecer)": np.diag(result["transition_matrix"]),
    }, index=["Baixa Volatilidade", "Alta Volatilidade"])

    current_regime = int(viterbi.iloc[-1])
    current_regime_label = "Alta Volatilidade" if current_regime == 1 else "Baixa Volatilidade"
    current_prob = float(state_probs_df.iloc[-1, current_regime])

    return {
        "state_probs": state_probs_df,
        "viterbi_states": viterbi,
        "regime_stats": regime_stats,
        "transition_matrix": result["transition_matrix"],
        "current_regime_label": current_regime_label,
        "current_regime_prob": current_prob,
        "log_likelihood": result["log_likelihood"],
    }