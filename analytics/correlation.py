# analytics/correlation.py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def correlation_matrix(data: pd.DataFrame, window: int) -> pd.DataFrame:
    """Matriz de correlação móvel (sobre retornos diários) nas últimas 'window' linhas."""
    if window > len(data):
        window = len(data)
    returns = data.pct_change().iloc[-window:]
    corr = returns.corr()
    return corr


def rolling_correlation(series_a: pd.Series, series_b: pd.Series, window: int) -> pd.Series:
    """Correlação rolante (Pearson) entre duas séries de preços."""
    ra = series_a.pct_change()
    rb = series_b.pct_change()
    joined = pd.DataFrame({"a": ra, "b": rb}).dropna()
    if len(joined) < window:
        return pd.Series(dtype=float)
    roll_corr = joined["a"].rolling(window, min_periods=max(5, window // 2)).corr(joined["b"])
    return roll_corr.dropna()


def pca_components(panel: pd.DataFrame, n_components: int = 3) -> dict:
    """PCA sobre retornos diários. Devolve explained_variance_ratio e loadings."""
    returns = panel.pct_change().dropna()
    if returns.shape[1] < 2:
        return {"explained_variance_ratio": [], "loadings": pd.DataFrame()}

    scaler = StandardScaler()
    scaled = scaler.fit_transform(returns)
    pca = PCA(n_components=min(n_components, scaled.shape[1]))
    pca.fit(scaled)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=returns.columns,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)],
    )
    return {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "loadings": loadings,
    }


def rolling_beta(asset: pd.Series, factor: pd.Series, window: int = 63) -> pd.Series:
    """Beta rolante: Cov(ret_asset, ret_factor) / Var(ret_factor)."""
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


def lead_lag_correlation(series_a: pd.Series, series_b: pd.Series, max_lag: int = 20) -> pd.DataFrame:
    """Correlação para diferentes desfasamentos (lag > 0 → B lidera A)."""
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


# Se quiseres manter algum teste rápido, coloca‑o dentro deste bloco (não será executado ao importar)
if __name__ == "__main__":
    # Exemplo de uso local
    pass