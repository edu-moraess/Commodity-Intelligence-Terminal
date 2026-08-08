"""Monte Carlo engine — Stationary Bootstrap, GBM, Jump, GARCH-MC, Student-t."""
from __future__ import annotations
import warnings
from typing import Any
import numpy as np
import pandas as pd
from analytics.metrics import daily_returns

try:
    from forecasting.jump_calibration import calibrate_jump_diffusion, path_dependent_barrier_probs
    _HAS_JUMP_CAL = True
except ImportError:
    _HAS_JUMP_CAL = False


def _optimal_block_length(rets: np.ndarray, max_lag: int = 40) -> int:
    """Estima block length ótimo via regra prática baseada em ACF."""
    n = len(rets)
    if n < 30:
        return 5
    max_lag = min(max_lag, n // 4)
    rets_c = rets - rets.mean()
    var = np.dot(rets_c, rets_c) / n
    if var <= 0:
        return 5
    acf = np.array([
        np.dot(rets_c[:n - k], rets_c[k:]) / (n * var) for k in range(1, max_lag + 1)
    ])
    threshold = 2.0 / np.sqrt(n)
    below = np.where(np.abs(acf) < threshold)[0]
    if len(below) > 0:
        opt = int(below[0] + 1)
    else:
        opt = int(np.argmax(np.abs(acf)) + 1)
    return int(np.clip(opt, 3, 20))


def _stationary_bootstrap_path(
    rets: np.ndarray,
    horizon: int,
    p: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Stationary Bootstrap (Politis & Romano 1994)."""
    n = len(rets)
    path = np.empty(horizon)
    t = 0
    while t < horizon:
        start = rng.integers(0, n)
        length = rng.geometric(p)
        for j in range(length):
            if t >= horizon:
                break
            path[t] = rets[(start + j) % n]
            t += 1
    return path


def _fit_garch11_simple(rets: np.ndarray) -> tuple[float, float, float, float]:
    """Ajuste rápido GARCH(1,1) por MLE simplificado."""
    from scipy.optimize import minimize

    rets = rets - rets.mean()
    n = len(rets)

    def neg_ll(params):
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
            return 1e12
        sigma2 = np.empty(n)
        sigma2[0] = np.var(rets)
        for t in range(1, n):
            sigma2[t] = omega + alpha * rets[t - 1] ** 2 + beta * sigma2[t - 1]
        sigma2 = np.maximum(sigma2, 1e-12)
        return 0.5 * np.sum(np.log(2 * np.pi * sigma2) + rets**2 / sigma2)

    x0 = [1e-6, 0.08, 0.90]
    bounds = [(1e-8, None), (0.0, 0.5), (0.0, 0.99)]
    res = minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds)
    omega, alpha, beta = res.x
    sigma2 = np.var(rets)
    for t in range(1, n):
        sigma2 = omega + alpha * rets[t - 1] ** 2 + beta * sigma2
    last_sigma = float(np.sqrt(max(sigma2, 1e-12)))
    return float(omega), float(alpha), float(beta), last_sigma


def monte_carlo_paths(
    close: pd.Series,
    horizon_days: int,
    n_sims: int = 2000,
    lookback: int = 504,
    method: str = "block_bootstrap",
    seed: int = 42,
    block_size: int | None = None,
    mu: float | None = None,
    sigma: float | None = None,
    jump_lambda: float = 0.1,
    jump_mu: float = -0.02,
    jump_sigma: float = 0.05,
    df_student: float = 5.0,
) -> np.ndarray:
    """Simula trajetórias de preço (Stationary Bootstrap / GBM / Jump / Student-t / GARCH-MC)."""
    rets = daily_returns(close).tail(lookback).dropna().values
    if len(rets) < 20:
        rets = daily_returns(close).dropna().values
    if len(rets) < 10:
        raise ValueError("Série de retornos insuficiente para Monte Carlo.")

    rng = np.random.default_rng(seed)
    s0 = float(close.iloc[-1])

    if mu is None:
        mu = float(np.mean(rets))
    if sigma is None:
        sigma = float(np.std(rets, ddof=1))

    paths = np.zeros((n_sims, horizon_days))

    if method == "block_bootstrap":
        if block_size is None or block_size <= 0:
            block_size = _optimal_block_length(rets)
        p = 1.0 / max(block_size, 1)
        for i in range(n_sims):
            path_rets = _stationary_bootstrap_path(rets, horizon_days, p, rng)
            paths[i] = s0 * np.exp(np.cumsum(path_rets))

    elif method == "gbm":
        dt = 1.0
        z = rng.standard_normal((n_sims, horizon_days))
        path_rets = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
        paths = s0 * np.exp(np.cumsum(path_rets, axis=1))

    elif method == "jump_diffusion":
        if _HAS_JUMP_CAL and jump_lambda == 0.1 and jump_mu == -0.02 and jump_sigma == 0.05:
            cal = calibrate_jump_diffusion(rets)
            jump_lambda = cal["jump_lambda"]
            jump_mu = cal["jump_mu"]
            jump_sigma = cal["jump_sigma"]
            mu = cal["mu"]
            sigma = cal["sigma"]
        dt = 1.0
        z = rng.standard_normal((n_sims, horizon_days))
        n_jumps = rng.poisson(jump_lambda * dt, size=(n_sims, horizon_days))
        jumps = n_jumps * (jump_mu + jump_sigma * rng.standard_normal((n_sims, horizon_days)))
        path_rets = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z + jumps
        paths = s0 * np.exp(np.cumsum(path_rets, axis=1))

    elif method == "student_t":
        dt = 1.0
        z = rng.standard_t(df_student, size=(n_sims, horizon_days))
        z = z / (np.std(z, axis=1, keepdims=True) + 1e-12) * sigma
        path_rets = (mu - 0.5 * sigma**2) * dt + z
        paths = s0 * np.exp(np.cumsum(path_rets, axis=1))

    elif method == "garch_mc":
        omega, alpha, beta, last_sigma = _fit_garch11_simple(rets)
        for i in range(n_sims):
            z = rng.standard_normal(horizon_days)
            sigma_t = last_sigma
            path_rets = np.empty(horizon_days)
            for t in range(horizon_days):
                path_rets[t] = mu + sigma_t * z[t]
                sigma_t = np.sqrt(omega + alpha * path_rets[t] ** 2 + beta * sigma_t ** 2)
            paths[i] = s0 * np.exp(np.cumsum(path_rets))

    else:
        raise ValueError(f"Método Monte Carlo desconhecido: {method}")

    return paths


def scenario_summary(
    close: pd.Series,
    horizon_days: int,
    n_sims: int = 2000,
    method: str = "block_bootstrap",
    seed: int = 42,
    block_size: int | None = None,
) -> dict:
    """Gera cenários + métricas + probabilidades de rompimento (terminal + path-dependent)."""
    paths = monte_carlo_paths(
        close, horizon_days, n_sims=n_sims, method=method, seed=seed, block_size=block_size
    )
    future_dates = pd.bdate_range(start=close.index[-1] + pd.Timedelta(days=1), periods=horizon_days)

    percentiles = {p: np.percentile(paths, p, axis=0) for p in [5, 10, 25, 50, 75, 90, 95]}
    fan_chart = pd.DataFrame(percentiles, index=future_dates)
    fan_chart.columns = [f"p{p}" for p in fan_chart.columns]

    final_prices = paths[:, -1]
    last_price = float(close.iloc[-1])
    rets_final = final_prices / last_price - 1.0

    sma20 = float(close.tail(20).mean()) if len(close) >= 20 else last_price
    sma50 = float(close.tail(50).mean()) if len(close) >= 50 else last_price
    std20 = float(close.tail(20).std()) if len(close) >= 20 else 0.0
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    support = float(close.tail(60).min()) if len(close) >= 60 else last_price * 0.9
    resistance = float(close.tail(60).max()) if len(close) >= 60 else last_price * 1.1

    path_barriers = {}
    if _HAS_JUMP_CAL:
        path_barriers = path_dependent_barrier_probs(
            paths, support=support, resistance=resistance,
            sma20=sma20, bb_upper=bb_upper, bb_lower=bb_lower,
        )

    from scipy import stats as _stats
    skew = float(_stats.skew(rets_final))
    kurt = float(_stats.kurtosis(rets_final))
    jb_stat, jb_p = _stats.jarque_bera(rets_final)

    effective_block = block_size
    if method == "block_bootstrap" and block_size is None:
        rets = daily_returns(close).tail(504).dropna().values
        if len(rets) >= 20:
            effective_block = _optimal_block_length(rets)

    return {
        "fan_chart": fan_chart,
        "cenario_base": float(np.percentile(final_prices, 50)),
        "cenario_otimista": float(np.percentile(final_prices, 90)),
        "cenario_pessimista": float(np.percentile(final_prices, 10)),
        "expected_price": float(np.mean(final_prices)),
        "expected_return": float(np.mean(rets_final)),
        "prob_alta": float(np.mean(final_prices > last_price)),
        "prob_baixa": float(np.mean(final_prices < last_price)),
        "prob_rompe_suporte": float(np.mean(final_prices < support)),
        "prob_rompe_resistencia": float(np.mean(final_prices > resistance)),
        "prob_acima_sma20": float(np.mean(final_prices > sma20)),
        "prob_acima_sma50": float(np.mean(final_prices > sma50)),
        "prob_acima_bb_upper": float(np.mean(final_prices > bb_upper)),
        "prob_abaixo_bb_lower": float(np.mean(final_prices < bb_lower)),
        "prob_rompe_suporte_path": path_barriers.get("prob_rompe_suporte_path", float(np.mean(final_prices < support))),
        "prob_rompe_resistencia_path": path_barriers.get("prob_rompe_resistencia_path", float(np.mean(final_prices > resistance))),
        "prob_acima_bb_upper_path": path_barriers.get("prob_acima_bb_upper_path"),
        "prob_abaixo_bb_lower_path": path_barriers.get("prob_abaixo_bb_lower_path"),
        "preco_atual": last_price,
        "final_prices_dist": final_prices,
        "intervalo_confianca_90": (
            float(np.percentile(final_prices, 5)),
            float(np.percentile(final_prices, 95)),
        ),
        "skewness": skew,
        "kurtosis": kurt,
        "jarque_bera_stat": float(jb_stat),
        "jarque_bera_pvalue": float(jb_p),
        "method": method,
        "seed": seed,
        "block_size_used": effective_block,
        "support": support,
        "resistance": resistance,
        "sma20": sma20,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
    }


def compare_monte_carlo_methods(
    close: pd.Series,
    horizon_days: int = 30,
    n_sims: int = 1500,
    seed: int = 42,
) -> pd.DataFrame:
    """Compara métodos de Monte Carlo lado a lado."""
    methods = ["block_bootstrap", "gbm", "jump_diffusion", "student_t", "garch_mc"]
    rows = []
    for m in methods:
        try:
            s = scenario_summary(close, horizon_days, n_sims=n_sims, method=m, seed=seed)
            rows.append({
                "Método": m,
                "Expected Price": s["expected_price"],
                "Expected Return": s["expected_return"],
                "P(Alta)": s["prob_alta"],
                "P(Baixa)": s["prob_baixa"],
                "P10": s["cenario_pessimista"],
                "P50": s["cenario_base"],
                "P90": s["cenario_otimista"],
                "Skewness": s["skewness"],
                "Kurtosis": s["kurtosis"],
                "JB p-value": s["jarque_bera_pvalue"],
                "Block Size": s.get("block_size_used"),
            })
        except Exception as exc:
            rows.append({"Método": m, "Erro": str(exc)[:80]})
    return pd.DataFrame(rows)
