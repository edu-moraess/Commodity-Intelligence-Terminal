"""Analytics — Correlação, Correlação Rolante e PCA sobre painel de ativos."""

from __future__ import annotations
import numpy as np
import pandas as pd


def returns_panel(price_panel: pd.DataFrame) -> pd.DataFrame:
    return price_panel.pct_change().dropna(how="all")


def correlation_matrix(price_panel: pd.DataFrame, window: int | None = None) -> pd.DataFrame:
    rets = returns_panel(price_panel)
    if window:
        rets = rets.tail(window)
    return rets.corr()


def rolling_correlation(series_a: pd.Series, series_b: pd.Series, window: int = 63) -> pd.Series:
    ra = series_a.pct_change()
    rb = series_b.pct_change()
    joined = pd.concat([ra, rb], axis=1, join="inner").dropna()
    joined.columns = ["a", "b"]
    return joined["a"].rolling(window).corr(joined["b"]).dropna()


def pca_components(price_panel: pd.DataFrame, n_components: int = 3) -> dict:
    """PCA simples via SVD (evita dependência obrigatória de sklearn para
    este cálculo específico, embora sklearn já seja usado em forecasting)."""
    rets = returns_panel(price_panel).dropna()
    if rets.empty or rets.shape[1] < 2:
        return {"explained_variance_ratio": [], "loadings": pd.DataFrame(), "scores": pd.DataFrame()}

    X = (rets - rets.mean()) / rets.std(ddof=0).replace(0, 1)
    X = X.fillna(0).values
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