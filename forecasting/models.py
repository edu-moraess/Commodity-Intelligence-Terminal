"""
Forecasting — Ensemble de Modelos de Projeção (Institucional)
=============================================================
Baseline determinístico + ensemble de modelos clássicos e de ML +
Monte Carlo avançado (Stationary Block Bootstrap, GBM, Jump Diffusion,
GARCH-MC real, Student-t).

Cada modelo produz: RMSE, MAE, MAPE, R², erro fora da amostra.
Walk-forward validation automático com seleção do melhor modelo.

Modelos:
  Linear, Ridge, Lasso, ElasticNet, RandomForest, XGBoost, LightGBM,
  CatBoost, ARIMA, SARIMA, Prophet (quando disponível).

Monte Carlo:
  Stationary Block Bootstrap (Politis & Romano), GBM, Merton Jump-Diffusion,
  GARCH(1,1)-MC real, Student-t innovations.

Referências:
  - Hyndman & Athanasopoulos (Forecasting Principles)
  - Politis & Romano (1994) Stationary Bootstrap
  - Merton (1976) Jump Diffusion
  - Bollerslev GARCH
"""

from __future__ import annotations
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from analytics.metrics import daily_returns

# Optional heavy dependencies
try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

try:
    from catboost import CatBoostRegressor
    _HAS_CAT = True
except ImportError:
    _HAS_CAT = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False

try:
    from prophet import Prophet
    _HAS_PROPHET = True
except ImportError:
    _HAS_PROPHET = False

try:
    from arch import arch_model
    _HAS_ARCH = True
except ImportError:
    _HAS_ARCH = False


# ---------------------------------------------------------------------------
# Registry de modelos de tendência / regressão
# ---------------------------------------------------------------------------

def _build_model_registry() -> dict[str, Any]:
    reg: dict[str, Any] = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01, max_iter=8000),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=8000),
        "RandomForest": RandomForestRegressor(
            n_estimators=120, max_depth=6, min_samples_leaf=5, random_state=42, n_jobs=-1
        ),
    }
    if _HAS_XGB:
        reg["XGBoost"] = XGBRegressor(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            verbosity=0, n_jobs=-1,
        )
    if _HAS_LGBM:
        reg["LightGBM"] = LGBMRegressor(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            verbosity=-1, n_jobs=-1,
        )
    if _HAS_CAT:
        reg["CatBoost"] = CatBoostRegressor(
            iterations=150, depth=4, learning_rate=0.05,
            random_seed=42, verbose=0,
        )
    return reg


MODEL_REGISTRY = _build_model_registry()


# ---------------------------------------------------------------------------
# Trend forecast (legado + expandido)
# ---------------------------------------------------------------------------

def trend_forecast(
    close: pd.Series,
    horizon_days: int,
    model_name: str = "Linear",
    lookback: int = 252,
) -> pd.Series:
    """Ajusta log(preço) ~ t e projeta horizon_days à frente.

    Nota Quant Dev: modelos de árvore/boosting aqui usam apenas o índice
    temporal como feature (legado). Para uso preditivo real de RF/XGB,
    prefira pipelines com lags + vol features (roadmap Fase 2).
    """
    hist = close.tail(lookback).dropna()
    if len(hist) < 10:
        raise ValueError("Histórico insuficiente para trend_forecast.")

    y = np.log(hist.values)
    X = np.arange(len(hist)).reshape(-1, 1)

    registry = _build_model_registry()
    model = registry.get(model_name, LinearRegression())
    if hasattr(model, "get_params"):
        from sklearn.base import clone
        try:
            model = clone(model)
        except Exception:
            pass

    model.fit(X, y)
    future_X = np.arange(len(hist), len(hist) + horizon_days).reshape(-1, 1)
    pred_log = model.predict(future_X)
    pred = np.exp(pred_log)

    future_dates = pd.bdate_range(start=hist.index[-1] + pd.Timedelta(days=1), periods=horizon_days)
    return pd.Series(pred, index=future_dates, name=f"forecast_{model_name.lower()}")


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate_point_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Métricas padrão de forecast de ponto."""
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": _mape(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
    }


def walk_forward_validation(
    close: pd.Series,
    train_window: int = 120,
    n_folds: int = 15,
    models: list[str] | None = None,
) -> pd.DataFrame:
    """Walk-forward 1-step ahead. Retorna ranking de modelos por RMSE."""
    log_close = np.log(close.dropna().values)
    total_len = len(log_close)
    if total_len < train_window + n_folds + 5:
        n_folds = max(3, total_len - train_window - 2)

    fold_starts = np.linspace(train_window, total_len - 2, n_folds, dtype=int)
    registry = _build_model_registry()
    if models is None:
        models = list(registry.keys())

    results: dict[str, dict[str, list]] = {m: {"mae": [], "rmse": [], "preds": [], "trues": []} for m in models}

    for start in fold_starts:
        y_train = log_close[start - train_window:start]
        X_train = np.arange(train_window).reshape(-1, 1)
        y_true_next = log_close[start]
        X_next = np.array([[train_window]])

        for name in models:
            if name not in registry:
                continue
            from sklearn.base import clone
            try:
                model = clone(registry[name])
            except Exception:
                model = registry[name]
            try:
                model.fit(X_train, y_train)
                pred = model.predict(X_next)[0]
                err = np.exp(pred) - np.exp(y_true_next)
                results[name]["mae"].append(abs(err))
                results[name]["rmse"].append(err ** 2)
                results[name]["preds"].append(np.exp(pred))
                results[name]["trues"].append(np.exp(y_true_next))
            except Exception:
                continue

    summary = []
    for name, r in results.items():
        if not r["mae"]:
            continue
        y_t = np.array(r["trues"])
        y_p = np.array(r["preds"])
        metrics = evaluate_point_forecast(y_t, y_p)
        summary.append({
            "Modelo": name,
            "MAE (1-step)": metrics["MAE"],
            "RMSE (1-step)": metrics["RMSE"],
            "MAPE (1-step)": metrics["MAPE"],
            "R2 (1-step)": metrics["R2"],
            "n_folds": len(r["mae"]),
        })
    df = pd.DataFrame(summary)
    if not df.empty:
        df = df.sort_values("RMSE (1-step)").reset_index(drop=True)
    return df


def select_best_trend_model(close: pd.Series, train_window: int = 120, n_folds: int = 15) -> tuple[str, pd.DataFrame]:
    """Retorna (melhor_modelo, ranking_completo)."""
    ranking = walk_forward_validation(close, train_window=train_window, n_folds=n_folds)
    if ranking.empty:
        return "Linear", ranking
    return str(ranking.iloc[0]["Modelo"]), ranking


# ---------------------------------------------------------------------------
# ARIMA / SARIMA / Prophet helpers
# ---------------------------------------------------------------------------

def arima_forecast(close: pd.Series, horizon_days: int, order: tuple = (1, 1, 1), lookback: int = 504) -> pd.Series:
    if not _HAS_STATSMODELS:
        return trend_forecast(close, horizon_days, "Linear", lookback)
    hist = close.tail(lookback).dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMA(hist.values, order=order)
        res = model.fit()
        pred = res.forecast(steps=horizon_days)
    future_dates = pd.bdate_range(start=hist.index[-1] + pd.Timedelta(days=1), periods=horizon_days)
    return pd.Series(pred, index=future_dates, name="forecast_arima")


def prophet_forecast(close: pd.Series, horizon_days: int, lookback: int = 756) -> pd.Series:
    if not _HAS_PROPHET:
        return trend_forecast(close, horizon_days, "Linear", lookback)
    hist = close.tail(lookback).dropna().reset_index()
    hist.columns = ["ds", "y"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
        m.fit(hist)
        future = m.make_future_dataframe(periods=horizon_days, freq="B")
        fc = m.predict(future)
    pred = fc["yhat"].tail(horizon_days).values
    future_dates = pd.bdate_range(start=close.index[-1] + pd.Timedelta(days=1), periods=horizon_days)
    return pd.Series(pred, index=future_dates, name="forecast_prophet")


# ---------------------------------------------------------------------------
# Helpers Monte Carlo — Stationary Bootstrap + auto block length
# ---------------------------------------------------------------------------

def _optimal_block_length(rets: np.ndarray, max_lag: int = 40) -> int:
    """Estima block length ótimo via regra prática baseada em ACF.

    Usa o primeiro lag em que |ACF| cai abaixo de 2/sqrt(n) (limite 95%
    aproximado) ou o lag de máxima ACF residual. Fallback = 5.
    Referência: Politis & White (2004) / regras práticas de stationary bootstrap.
    """
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
    """Gera uma trajetória de retornos via Stationary Bootstrap (Politis & Romano 1994).

    p = probabilidade de iniciar novo bloco (esperado de comprimento do bloco = 1/p).
    """
    n = len(rets)
    path = np.empty(horizon)
    t = 0
    while t < horizon:
        start = rng.integers(0, n)
        length = rng.geometric(p)
        for j in range(length):
            if t >= horizon:
                break
            path[t] = rets[(start + j) % n]  # circular
            t += 1
    return path


def _fit_garch11_simple(rets: np.ndarray) -> tuple[float, float, float, float]:
    """Ajuste rápido GARCH(1,1) por MLE simplificado (fallback sem arch).

    Retorna omega, alpha, beta, last_sigma.
    """
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


# ---------------------------------------------------------------------------
# Monte Carlo avançado (vetorizado onde possível)
# ---------------------------------------------------------------------------

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
    """
    Simula trajetórias de preço.

    method:
      - 'block_bootstrap'  → Stationary Bootstrap (Politis & Romano)
      - 'gbm'
      - 'jump_diffusion'   → Merton
      - 'student_t'
      - 'garch_mc'         → GARCH(1,1) real (recursão de σ²)

    block_size:
      - None → estimado automaticamente via ACF
      - int  → força comprimento médio do bloco (p = 1/block_size)
    """
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
    """Gera cenários + métricas de distribuição + probabilidades de rompimento.

    Parâmetros extras (seed, block_size) agora são expostos para reprodutibilidade.
    """
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
    """Compara métodos de Monte Carlo lado a lado (seed controlável)."""
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
