"""Analytics — Correlação, Correlação Rolante e PCA (v4.4.0)

CHANGELOG v4.4.0:
- correlation_matrix: dropna(how='all', axis=1) remove colunas 100% NaN
  e usa min_periods para evitar matriz inteira de NaN.
- pca_components: remove colunas com <20 observações válidas antes do SVD
  evitando componentes degenerados (ex: PC1=PC2=0.5).
- rolling_correlation: valida se há dados suficientes antes de calcular.
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
    # Remove colunas 100% NaN (ativos sem dados)
    rets = rets.dropna(how="all", axis=1)
    if rets.shape[1] < 2:
        return pd.DataFrame()
    # min_periods evita NaN quando há poucos dados em comum
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
    # Remove colunas com poucos dados válidos (menos de 20 obs)
    rets = rets.loc[:, rets.count() >= 20]
    # Dropna how='any' — só linhas onde TODAS as colunas têm valor
    rets = rets.dropna()
    if rets.empty or rets.shape[1] < 2:
        return {"explained_variance_ratio": [], "loadings": pd.DataFrame(), "scores": pd.DataFrame()}

    # Padronização
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
