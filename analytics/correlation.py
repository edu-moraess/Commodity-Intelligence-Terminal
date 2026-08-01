"""Analytics — Correlação, Correlação Rolante, PCA, Beta e Lead-Lag

CHANGELOG:
- correlation_matrix: dropna robusto + min_periods
- pca_components: filtra colunas com <20 obs
- rolling_correlation: valida tamanho mínimo
- rolling_beta: beta rolante commodity vs fator macro
- lead_lag_correlation: correlação com defasagens ±max_lag
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def returns_panel(price_panel: pd.DataFrame) -> pd.DataFrame:
    return price_panel.pct_change().dropna(how="all")


def correlation_matrix(price_panel: pd.DataFrame, window: int | None = None) -> pd.DataFrame:
    rets = returns_panel(price_panel)
    if window:
        rets = rets.tail(window)
    rets = rets.dropna(how="all", axis=1)
    if rets.shape[1] < 2:
        return pd.DataFrame()
    min_p = max(10, len(rets) // 4)
    return rets.corr(min_periods=min_p)


def rolling_correlation(series_a: pd.Series, series_b: pd.Series, window: int = 63) -> pd.Series:
    ra = series_a.pct_change()
    rb = series_b.pct_change()
    joined = pd.concat([ra, rb], axis=1, join="inner").dropna()
    if len(joined) < window + 5:
        return pd.Series(dtype=float)
    joined.columns = ["a", "b"]
    return joined["a"].rolling(window, min_periods=max(window // 2, 5)).corr(joined["b"]).dropna()


def pca_components(price_panel: pd.DataFrame, n_components: int = 3) -> dict:
    """PCA simples via SVD com validação robusta de dados."""
    rets = returns_panel(price_panel).dropna(how="all", axis=1)
    rets = rets.loc[:, rets.count() >= 20]
    rets = rets.dropna()
    if rets.empty or rets.shape[1] < 2:
        return {"explained_variance_ratio": [], "loadings": pd.DataFrame(), "scores": pd.DataFrame()}

    stds = rets.std(ddof=0).replace(0, 1)
    X = ((rets - rets.mean()) / stds).fillna(0).values

    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    n = min(n_components, Vt.shape[0])
    explained = (S**2) / np.sum(S**2)
    loadings = pd.DataFrame(
        Vt[:n].T, index=rets.columns, columns=[f"PC{i+1}" for i in range(n)]
    )
    scores = pd.DataFrame(
        U[:, :n] * S[:n], index=rets.index, columns=[f"PC{i+1}" for i in range(n)]
    )
    return {
        "explained_variance_ratio": explained[:n].tolist(),
        "loadings": loadings,
        "scores": scores,
    }


def rolling_beta(asset: pd.Series, factor: pd.Series, window: int = 63) -> pd.Series:
    """Beta rolante de asset vs factor (preços → retornos)."""
    ra = asset.pct_change()
    rf = factor.pct_change()
    joined = pd.concat([ra, rf], axis=1, join="inner").dropna()
    if len(joined) < window + 5:
        return pd.Series(dtype=float)
    joined.columns = ["a", "f"]
    cov = joined["a"].rolling(window, min_periods=max(window // 2, 5)).cov(joined["f"])
    var = joined["f"].rolling(window, min_periods=max(window // 2, 5)).var()
    beta = (cov / var.replace(0, np.nan)).dropna()
    return beta


def lead_lag_correlation(
    series_a: pd.Series,
    series_b: pd.Series,
    max_lag: int = 20,
) -> pd.DataFrame:
    """
    Correlação de retornos de A com B defasado de -max_lag … +max_lag.
    Lag > 0: B lidera A. Lag < 0: A lidera B.
    """
    ra = series_a.pct_change().dropna()
    rb = series_b.pct_change().dropna()
    joined = pd.concat([ra, rb], axis=1, join="inner").dropna()
    if len(joined) < max_lag * 3:
        return pd.DataFrame(columns=["lag", "correlation"])
    joined.columns = ["a", "b"]
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        if lag == 0:
            corr = joined["a"].corr(joined["b"])
        elif lag > 0:
            corr = joined["a"].iloc[lag:].reset_index(drop=True).corr(
                joined["b"].iloc[:-lag].reset_index(drop=True)
            )
        else:
            lag_abs = abs(lag)
            corr = joined["a"].iloc[:-lag_abs].reset_index(drop=True).corr(
                joined["b"].iloc[lag_abs:].reset_index(drop=True)
            )
        rows.append({"lag": lag, "correlation": corr})
    return pd.DataFrame(rows)