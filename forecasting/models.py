"""
Forecasting — Modelos de Projeção
=====================================
Implementa:
  - Regressão em tendência log-linear (Linear / Ridge / Lasso) como
    baseline determinístico
  - Simulação Monte Carlo (GBM com bootstrap de retornos históricos) para
    gerar a distribuição de probabilidade e os cenários Base/Otimista/
    Pessimista com fan chart

Cada horizonte em `config.settings.FORECAST_HORIZONS` é suportado.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso

from analytics.metrics import daily_returns

MODEL_REGISTRY = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.01, max_iter=5000),
}


def trend_forecast(close: pd.Series, horizon_days: int, model_name: str = "Linear",
                    lookback: int = 252) -> pd.Series:
    """Ajusta log(preço) ~ t em uma janela recente e projeta `horizon_days`
    à frente. Simples, interpretável, serve de baseline para comparação
    com os cenários probabilísticos de Monte Carlo."""
    hist = close.tail(lookback)
    y = np.log(hist.values).reshape(-1, 1)
    X = np.arange(len(hist)).reshape(-1, 1)

    model = MODEL_REGISTRY.get(model_name, LinearRegression())
    model.fit(X, y.ravel())

    future_X = np.arange(len(hist), len(hist) + horizon_days).reshape(-1, 1)
    pred_log = model.predict(future_X)
    pred = np.exp(pred_log)

    future_dates = pd.bdate_range(start=hist.index[-1] + pd.Timedelta(days=1), periods=horizon_days)
    return pd.Series(pred, index=future_dates, name=f"forecast_{model_name.lower()}")


def monte_carlo_paths(close: pd.Series, horizon_days: int, n_sims: int = 2000,
                       lookback: int = 504, block_bootstrap: bool = True,
                       seed: int = 42) -> np.ndarray:
    """Simula `n_sims` trajetórias de preço via bootstrap de blocos dos
    retornos históricos (captura autocorrelação/clusters de vol melhor que
    um GBM puramente i.i.d.). Retorna array (n_sims, horizon_days)."""
    rets = daily_returns(close).tail(lookback).values
    if len(rets) < 20:
        rets = daily_returns(close).values
    rng = np.random.default_rng(seed)
    s0 = float(close.iloc[-1])

    block_size = 5
    paths = np.zeros((n_sims, horizon_days))
    for i in range(n_sims):
        if block_bootstrap:
            n_blocks = int(np.ceil(horizon_days / block_size))
            sampled = []
            for _ in range(n_blocks):
                start = rng.integers(0, max(len(rets) - block_size, 1))
                sampled.append(rets[start:start + block_size])
            path_rets = np.concatenate(sampled)[:horizon_days]
        else:
            path_rets = rng.choice(rets, size=horizon_days, replace=True)
        paths[i] = s0 * np.exp(np.cumsum(path_rets))
    return paths


def scenario_summary(close: pd.Series, horizon_days: int, n_sims: int = 2000) -> dict:
    """Gera cenários Base (mediana), Otimista (p90) e Pessimista (p10) mais
    o fan chart (bandas p10/p25/p50/p75/p90) a partir das trajetórias de
    Monte Carlo."""
    paths = monte_carlo_paths(close, horizon_days, n_sims=n_sims)
    future_dates = pd.bdate_range(start=close.index[-1] + pd.Timedelta(days=1), periods=horizon_days)

    percentiles = {p: np.percentile(paths, p, axis=0) for p in [10, 25, 50, 75, 90]}
    fan_chart = pd.DataFrame(percentiles, index=future_dates)
    fan_chart.columns = [f"p{p}" for p in fan_chart.columns]

    final_prices = paths[:, -1]
    last_price = float(close.iloc[-1])

    return {
        "fan_chart": fan_chart,
        "cenario_base": float(np.percentile(final_prices, 50)),
        "cenario_otimista": float(np.percentile(final_prices, 90)),
        "cenario_pessimista": float(np.percentile(final_prices, 10)),
        "prob_alta": float(np.mean(final_prices > last_price)),
        "preco_atual": last_price,
        "final_prices_dist": final_prices,
        "intervalo_confianca_90": (
            float(np.percentile(final_prices, 5)),
            float(np.percentile(final_prices, 95)),
        ),
    }