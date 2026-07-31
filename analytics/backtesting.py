"""
Analytics — Backtesting de VaR (Kupiec / Christoffersen)
============================================================
Testes formais de adequação de modelos de VaR, padrão em risk management
institucional (Basel III traffic-light approach usa o mesmo princípio).

Referências:
- Kupiec, P. (1995). Techniques for Verifying the Accuracy of Risk
  Measurement Models. Journal of Derivatives, 3(2), 73-84.
- Christoffersen, P. (1998). Evaluating Interval Forecasts.
  International Economic Review, 39(4), 841-862.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats as _stats

from analytics.metrics import daily_returns


def rolling_var_forecast(close: pd.Series, confidence: float = 0.95,
                          window: int = 252, method: str = "historical") -> pd.Series:
    """VaR rolling out-of-sample: em cada dia t, estima o VaR usando apenas
    os `window` retornos ANTERIORES a t (nunca olha o próprio dia t —
    essencial para o backtest não ser otimista/enviesado)."""
    rets = daily_returns(close)
    var_series = pd.Series(index=rets.index, dtype=float)

    values = rets.values
    for i in range(window, len(values)):
        train = values[i - window:i]
        if method == "parametric":
            mu, sigma = train.mean(), train.std(ddof=1)
            z = _stats.norm.ppf(1 - confidence)
            var_t = -(mu + z * sigma)
        else:
            var_t = -np.percentile(train, (1 - confidence) * 100)
        var_series.iloc[i] = var_t

    return var_series.dropna()


def identify_breaches(close: pd.Series, var_series: pd.Series) -> pd.Series:
    """Retorna série booleana: True nos dias em que a perda realizada
    excedeu o VaR previsto (breach/exceção)."""
    rets = daily_returns(close)
    aligned = pd.concat([rets, var_series], axis=1, join="inner")
    aligned.columns = ["ret", "var"]
    breaches = aligned["ret"] < -aligned["var"]
    breaches.name = "breach"
    return breaches


def kupiec_pof_test(breaches: pd.Series, confidence: float = 0.95) -> dict:
    """Teste de Proporção de Falhas (POF) de Kupiec (1995).

    H0: a taxa de exceções observada é estatisticamente igual à taxa
    esperada (1 - confidence). Estatística LR segue qui-quadrado(1) sob H0.
    """
    n = len(breaches)
    x = int(breaches.sum())  # número de exceções observadas
    p = 1 - confidence        # taxa esperada de exceções

    if n == 0:
        return {"n_obs": 0, "n_breaches": 0, "breach_rate": np.nan,
                "expected_rate": p, "lr_stat": np.nan, "p_value": np.nan,
                "reject_h0": None}

    pi_hat = x / n

    # Log-likelihood sob H0 (taxa = p) vs H1 (taxa = pi_hat observada)
    def _loglik(prob, successes, trials):
        if prob <= 0 or prob >= 1:
            return -np.inf
        return successes * np.log(prob) + (trials - successes) * np.log(1 - prob)

    ll_null = _loglik(p, x, n)
    ll_alt = _loglik(pi_hat, x, n) if 0 < pi_hat < 1 else 0.0

    lr_stat = -2 * (ll_null - ll_alt)
    lr_stat = float(lr_stat) if np.isfinite(lr_stat) else np.nan
    p_value = float(1 - _stats.chi2.cdf(lr_stat, df=1)) if np.isfinite(lr_stat) else np.nan

    return {
        "n_obs": n, "n_breaches": x,
        "breach_rate": pi_hat, "expected_rate": p,
        "lr_stat": lr_stat, "p_value": p_value,
        "reject_h0": bool(p_value < 0.05) if np.isfinite(p_value) else None,
    }


def christoffersen_independence_test(breaches: pd.Series) -> dict:
    """Teste de Independência de Christoffersen (1998).

    H0: as exceções são independentes ao longo do tempo (não formam
    clusters). Um modelo de VaR "bom" não deve ter breaches concentrados
    em períodos de estresse — isso indicaria que o modelo não reage
    rápido o suficiente a mudanças de volatilidade.
    """
    b = breaches.astype(int).values
    n = len(b)
    if n < 2:
        return {"lr_stat": np.nan, "p_value": np.nan, "reject_h0": None}

    # Conta transições 0->0, 0->1, 1->0, 1->1
    n00 = n01 = n10 = n11 = 0
    for t in range(1, n):
        prev, curr = b[t - 1], b[t]
        if prev == 0 and curr == 0:
            n00 += 1
        elif prev == 0 and curr == 1:
            n01 += 1
        elif prev == 1 and curr == 0:
            n10 += 1
        else:
            n11 += 1

    n0, n1 = n00 + n01, n10 + n11
    pi01 = n01 / n0 if n0 > 0 else 0.0
    pi11 = n11 / n1 if n1 > 0 else 0.0
    pi = (n01 + n11) / (n0 + n1) if (n0 + n1) > 0 else 0.0

    def _safe_log(x, p):
        if p <= 0 or p >= 1:
            return 0.0
        return x * np.log(p)

    ll_null = _safe_log(n01 + n11, pi) + _safe_log(n00 + n10, 1 - pi)
    ll_alt = (_safe_log(n01, pi01) + _safe_log(n00, 1 - pi01)
              + _safe_log(n11, pi11) + _safe_log(n10, 1 - pi11))

    lr_stat = -2 * (ll_null - ll_alt)
    lr_stat = float(lr_stat) if np.isfinite(lr_stat) and lr_stat >= 0 else np.nan
    p_value = float(1 - _stats.chi2.cdf(lr_stat, df=1)) if np.isfinite(lr_stat) else np.nan

    return {
        "lr_stat": lr_stat, "p_value": p_value,
        "reject_h0": bool(p_value < 0.05) if np.isfinite(p_value) else None,
    }


def joint_backtest(close: pd.Series, confidence: float = 0.95, window: int = 252,
                    method: str = "historical") -> dict:
    """Roda o pipeline completo: VaR rolling -> breaches -> Kupiec +
    Christoffersen + teste conjunto (Christoffersen 1998, LR_cc = LR_pof + LR_ind ~ qui²(2))."""
    var_series = rolling_var_forecast(close, confidence, window, method)
    breaches = identify_breaches(close, var_series)

    pof = kupiec_pof_test(breaches, confidence)
    ind = christoffersen_independence_test(breaches)

    if np.isfinite(pof.get("lr_stat", np.nan)) and np.isfinite(ind.get("lr_stat", np.nan)):
        lr_cc = pof["lr_stat"] + ind["lr_stat"]
        p_cc = float(1 - _stats.chi2.cdf(lr_cc, df=2))
    else:
        lr_cc, p_cc = np.nan, np.nan

    return {
        "var_series": var_series,
        "breaches": breaches,
        "kupiec": pof,
        "christoffersen": ind,
        "joint": {"lr_stat": lr_cc, "p_value": p_cc,
                  "reject_h0": bool(p_cc < 0.05) if np.isfinite(p_cc) else None},
    }