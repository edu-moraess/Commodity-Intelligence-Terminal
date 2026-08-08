"""
Forecasting — Ensemble de Modelos de Projeção (Institucional)
=============================================================
Baseline determinístico + ensemble de modelos clássicos e de ML +
Monte Carlo avançado (via forecasting.mc_engine).

Monte Carlo API is re-exported from forecasting.mc_engine for backward compatibility.
Density backtest (Fase 4) re-exported from forecasting.density_backtest.
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

try:
    from forecasting.jump_calibration import calibrate_jump_diffusion, path_dependent_barrier_probs
    _HAS_JUMP_CAL = True
except ImportError:
    _HAS_JUMP_CAL = False

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

_TREE_MODELS = {"RandomForest", "XGBoost", "LightGBM", "CatBoost"}

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

def _make_trend_features(log_prices: np.ndarray, max_lag: int = 5):
    n = len(log_prices)
    min_hist = max(20, max_lag + 1)
    if n < min_hist + 5:
        return np.arange(n).reshape(-1, 1).astype(float)
    rets = np.diff(log_prices, prepend=log_prices[0])
    features, valid_idx = [], []
    for t in range(min_hist, n):
        row = []
        for lag in range(1, max_lag + 1):
            row.append(log_prices[t - lag])
        row.append(rets[t])
        row.append(log_prices[t] - log_prices[t - 5])
        row.append(np.std(rets[t - 5:t + 1]) + 1e-8)
        row.append(np.std(rets[max(0, t - 20):t + 1]) + 1e-8)
        row.append(log_prices[t] - log_prices[t - 10] if t >= 10 else 0.0)
        row.append(t / n)
        features.append(row)
        valid_idx.append(t)
    return np.asarray(features, dtype=float), np.asarray(valid_idx)

def _is_tree_model(name: str) -> bool:
    return name in _TREE_MODELS

def trend_forecast(close: pd.Series, horizon_days: int, model_name: str = "Linear", lookback: int = 252) -> pd.Series:
    hist = close.tail(lookback).dropna()
    if len(hist) < 30:
        raise ValueError("Histórico insuficiente para trend_forecast.")
    log_hist = np.log(hist.values)
    registry = _build_model_registry()
    model = registry.get(model_name, LinearRegression())
    if hasattr(model, "get_params"):
        from sklearn.base import clone
        try:
            model = clone(model)
        except Exception:
            pass
    future_dates = pd.bdate_range(start=hist.index[-1] + pd.Timedelta(days=1), periods=horizon_days)
    if not _is_tree_model(model_name):
        y = log_hist
        X = np.arange(len(hist)).reshape(-1, 1)
        model.fit(X, y)
        future_X = np.arange(len(hist), len(hist) + horizon_days).reshape(-1, 1)
        pred = np.exp(model.predict(future_X))
        return pd.Series(pred, index=future_dates, name=f"forecast_{model_name.lower()}")
    feat_result = _make_trend_features(log_hist)
    if isinstance(feat_result, tuple):
        X_feat, valid_idx = feat_result
        y_feat = log_hist[valid_idx]
    else:
        X_feat = np.arange(len(hist)).reshape(-1, 1)
        y_feat = log_hist
    model.fit(X_feat, y_feat)
    log_series = list(log_hist)
    preds = []
    for h in range(horizon_days):
        current = np.array(log_series)
        feat_result_h = _make_trend_features(current)
        if isinstance(feat_result_h, tuple):
            X_h, _ = feat_result_h
            x_next = X_h[-1:].copy()
        else:
            x_next = np.array([[len(log_series)]])
        pred_log = float(model.predict(x_next)[0])
        preds.append(np.exp(pred_log))
        log_series.append(pred_log)
    return pd.Series(preds, index=future_dates, name=f"forecast_{model_name.lower()}")

def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def evaluate_point_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": _mape(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
    }

def walk_forward_validation(close: pd.Series, train_window: int = 120, n_folds: int = 15, models: list[str] | None = None) -> pd.DataFrame:
    prices = close.dropna().values
    log_close = np.log(prices)
    total_len = len(log_close)
    if total_len < train_window + n_folds + 5:
        n_folds = max(3, total_len - train_window - 2)
    fold_starts = np.linspace(train_window, total_len - 2, n_folds, dtype=int)
    registry = _build_model_registry()
    if models is None:
        models = list(registry.keys())
    results: dict[str, dict[str, list]] = {m: {"mae": [], "rmse": [], "preds": [], "trues": []} for m in models}
    for start in fold_starts:
        true_price = np.exp(log_close[start])
        for name in models:
            if name not in registry:
                continue
            from sklearn.base import clone
            try:
                model = clone(registry[name])
            except Exception:
                model = registry[name]
            try:
                if _is_tree_model(name):
                    train_log = log_close[start - train_window:start]
                    feat_result = _make_trend_features(train_log)
                    if isinstance(feat_result, tuple):
                        X_train, valid_idx = feat_result
                        y_train = train_log[valid_idx]
                        if len(y_train) < 10:
                            continue
                        model.fit(X_train, y_train)
                        pred_log = float(model.predict(X_train[-1:].copy())[0])
                    else:
                        continue
                else:
                    y_train = log_close[start - train_window:start]
                    X_train = np.arange(train_window).reshape(-1, 1)
                    model.fit(X_train, y_train)
                    pred_log = float(model.predict(np.array([[train_window]]))[0])
                pred_price = np.exp(pred_log)
                err = pred_price - true_price
                results[name]["mae"].append(abs(err))
                results[name]["rmse"].append(err ** 2)
                results[name]["preds"].append(pred_price)
                results[name]["trues"].append(true_price)
            except Exception:
                continue
    summary = []
    for name, r in results.items():
        if not r["mae"]:
            continue
        metrics = evaluate_point_forecast(np.array(r["trues"]), np.array(r["preds"]))
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
    ranking = walk_forward_validation(close, train_window=train_window, n_folds=n_folds)
    if ranking.empty:
        return "Linear", ranking
    return str(ranking.iloc[0]["Modelo"]), ranking

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

# Re-export Monte Carlo API from engine (backward compatible)
from forecasting.mc_engine import (
    monte_carlo_paths,
    scenario_summary,
    compare_monte_carlo_methods,
    _optimal_block_length,
)

# Fase 4 — density backtest
try:
    from forecasting.density_backtest import (
        walk_forward_density_backtest,
        rank_mc_methods_by_crps,
    )
except ImportError:
    pass
