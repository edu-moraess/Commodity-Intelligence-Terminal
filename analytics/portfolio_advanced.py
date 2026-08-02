"""
Portfolio Optimization — Advanced
====================================
Extensões para otimização de portfólio com validação out-of-sample,
rolling weights, diagnóstico de séries e comparação avançada de métodos.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import zscore
import warnings
warnings.filterwarnings("ignore")

from analytics import portfolio as port
from utils.logger import get_logger

logger = get_logger("portfolio_advanced")


def validate_returns(returns: pd.DataFrame, z_threshold: float = 3.0) -> pd.DataFrame:
    """Remove outliers extremos (|z-score| > threshold) por coluna e
    interpola os buracos resultantes.

    BUGFIX: `scipy.stats.zscore` retorna um ndarray puro (sem `.index`),
    mesmo recebendo uma Series como entrada. O código original fazia
    `outliers.index[outliers]`, o que quebrava com
    `AttributeError: 'numpy.ndarray' object has no attribute 'index'`
    toda vez que a checkbox "Remover outliers" era marcada na UI.
    Corrigido usando o índice da própria Series de dados (`col_data`).
    """
    clean = returns.copy()
    for col in clean.columns:
        col_data = clean[col].dropna()
        if len(col_data) < 2:
            continue
        z = np.asarray(zscore(col_data.values))
        outlier_mask = np.abs(z) > z_threshold
        if outlier_mask.any():
            outlier_index = col_data.index[outlier_mask]
            n_out = int(outlier_mask.sum())
            logger.info(
                "validate_returns: removendo %d outlier(s) em '%s' (z>%s)",
                n_out, col, z_threshold,
            )
            clean.loc[outlier_index, col] = np.nan
            clean[col] = clean[col].interpolate(method="linear", limit=5)
    return clean.dropna(axis=0, how="any")


def rolling_weights(
    panel: pd.DataFrame,
    window: int = 252,
    method: str = "max_sharpe",
    risk_free: float = 0.045,
    long_only: bool = True,
    rebalance_freq: int = 21,
) -> pd.DataFrame:
    dates = panel.index
    weights_history = []
    for i in range(window, len(dates), rebalance_freq):
        train = panel.iloc[i - window:i]
        try:
            res = port.optimize_portfolio(
                train,
                method=method,
                window=window,
                risk_free=risk_free,
                long_only=long_only,
            )
            w = res["weights"].reindex(panel.columns, fill_value=0.0)
            weights_history.append((dates[i], w))
        except Exception as exc:
            logger.warning(
                "Rebalanceamento falhou em %s (method=%s, window=%s): %s",
                dates[i], method, window, exc,
            )
    if not weights_history:
        logger.warning(
            "rolling_weights: nenhum rebalanceamento bem-sucedido "
            "(method=%s, window=%s, rebalance_freq=%s, n_dates=%s)",
            method, window, rebalance_freq, len(dates),
        )
        return pd.DataFrame()
    df = pd.DataFrame({dt: w for dt, w in weights_history}).T
    df.index = pd.DatetimeIndex(df.index)
    return df


def walk_forward_backtest(
    panel: pd.DataFrame,
    method: str = "max_sharpe",
    window: int = 252,
    risk_free: float = 0.045,
    long_only: bool = True,
    rebalance_freq: int = 21,
) -> dict:
    weights_df = rolling_weights(
        panel, window, method, risk_free, long_only, rebalance_freq
    )
    if weights_df.empty:
        logger.warning(
            "walk_forward_backtest: weights vazios — retornando equity vazia "
            "(method=%s, window=%s)",
            method, window,
        )
        return {"equity_curve": pd.Series(dtype=float), "stats": {}}

    rets = port._returns_matrix(panel, window=None)
    rets = rets.reindex(panel.index)

    equity = pd.Series(1.0, index=rets.index)
    current_weights = None

    for date in rets.index:
        if date in weights_df.index:
            current_weights = weights_df.loc[date]
        if current_weights is None:
            continue
        daily_ret = (rets.loc[date] * current_weights).sum()
        if pd.notna(daily_ret):
            equity.loc[date] = equity.shift(1).fillna(1.0).loc[date] * (1 + daily_ret)

    # BUGFIX: `equity` já é o nível acumulado — não aplicar .cumprod() de novo.
    equity = equity.fillna(1.0)

    rets_series = equity.pct_change().dropna()
    if len(rets_series) == 0:
        logger.warning("walk_forward_backtest: série de retornos vazia após backtest")
        return {
            "equity_curve": equity,
            "stats": {},
            "rebalance_dates": weights_df.index.tolist(),
            "n_rebalances": len(weights_df),
        }

    ann_return = float((rets_series.mean() + 1) ** 252 - 1)
    ann_vol = float(rets_series.std() * np.sqrt(252))
    sharpe = (ann_return - risk_free) / ann_vol if ann_vol > 0 else 0.0
    max_dd = float((equity / equity.cummax() - 1).min())

    logger.info(
        "walk_forward_backtest OK: method=%s n_rebalances=%d "
        "ann_return=%.4f ann_vol=%.4f sharpe=%.3f max_dd=%.4f",
        method, len(weights_df), ann_return, ann_vol, sharpe, max_dd,
    )

    return {
        "equity_curve": equity,
        "stats": {
            "expected_return": ann_return,
            "volatility": ann_vol,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
        },
        "rebalance_dates": weights_df.index.tolist(),
        "n_rebalances": len(weights_df),
    }


def compare_methods_advanced(
    panel: pd.DataFrame,
    window: int = 252,
    risk_free: float = 0.045,
    long_only: bool = True,
) -> pd.DataFrame:
    methods = {
        "Max Sharpe": "max_sharpe",
        "Min Variance": "min_variance",
        "Risk Parity (ERC)": "risk_parity",
        "Max Diversification": "max_diversification",
        "Min CVaR (95%)": "min_cvar",
        "Equal Weight": "equal_weight",
    }
    results = []
    for label, m in methods.items():
        try:
            res = port.optimize_portfolio(
                panel,
                method=m,
                window=window,
                risk_free=risk_free,
                long_only=long_only,
            )
            w = res["weights"]
            w_pos = w[w > 0.01]
            hhi = float((w ** 2).sum())
            eq_weight = pd.Series(1 / len(panel.columns), index=panel.columns)
            turnover = float((w - eq_weight).abs().sum() / 2)
            results.append({
                "Método": label,
                "Retorno": res["stats"]["expected_return"],
                "Vol": res["stats"]["volatility"],
                "Sharpe": res["stats"]["sharpe"],
                "Max DD": res["max_drawdown"],
                "N ativos > 1%": len(w_pos),
                "HHI (concentração)": hhi,
                "Turnover (vs Equal)": turnover,
                "Erro": "",
            })
        except Exception as e:
            logger.warning("compare_methods_advanced: método '%s' falhou: %s", label, e)
            results.append({
                "Método": label,
                "Retorno": np.nan,
                "Vol": np.nan,
                "Sharpe": np.nan,
                "Max DD": np.nan,
                "N ativos > 1%": 0,
                "HHI (concentração)": np.nan,
                "Turnover (vs Equal)": np.nan,
                "Erro": str(e)[:50],
            })
    return pd.DataFrame(results).set_index("Método")


def asset_diagnostics(panel: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    ytd_start = pd.Timestamp(panel.index[-1].year, 1, 1)
    ytd_mask = panel.index >= ytd_start
    diag = []
    for col in panel.columns:
        prices = panel[col]
        rets = prices.pct_change().dropna()
        if len(prices) >= window:
            window_prices = prices.iloc[-window:]
            ret_window = window_prices.iloc[-1] / window_prices.iloc[0] - 1
            vol_window = rets.tail(window).std() * np.sqrt(252)
        else:
            ret_window = np.nan
            vol_window = np.nan
        if ytd_mask.any():
            ytd_prices = prices[ytd_mask]
            ret_ytd = (
                ytd_prices.iloc[-1] / ytd_prices.iloc[0] - 1
                if len(ytd_prices) > 1
                else np.nan
            )
        else:
            ret_ytd = np.nan
        dd = (prices / prices.cummax() - 1).min()
        skew = rets.skew()
        kurt = rets.kurtosis()
        anomaly = ""
        if pd.notna(ret_window) and pd.notna(ret_ytd) and abs(ret_window - ret_ytd) > 0.5:
            anomaly = f"Divergência: {ret_window:.1%} vs YTD {ret_ytd:.1%}"
        diag.append({
            "Ativo": col,
            "Retorno Janela (anual.)": ret_window,
            "Retorno YTD": ret_ytd,
            "Vol. Janela": vol_window,
            "Max DD": dd,
            "Skew": skew,
            "Kurtosis": kurt,
            "Anomalia": anomaly,
        })
    return pd.DataFrame(diag).set_index("Ativo")


def get_window_data(
    panel: pd.DataFrame, option: str, custom_window: int = 252
) -> pd.DataFrame:
    if option == "YTD (desde jan/2026)":
        ytd_start = pd.Timestamp(panel.index[-1].year, 1, 1)
        return panel[panel.index >= ytd_start]
    elif option == "Últimos 63 pregões":
        return panel.iloc[-63:]
    elif option == "Personalizado":
        return panel.iloc[-custom_window:]
    else:
        return panel.iloc[-252:]