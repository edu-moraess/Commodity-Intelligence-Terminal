"""Fase 4 — density backtest out-of-sample."""
import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from forecasting.density_backtest import (
    walk_forward_density_backtest,
    rank_mc_methods_by_crps,
)


@pytest.fixture
def sample_close():
    dates = pd.bdate_range("2022-01-01", periods=350)
    rng = np.random.default_rng(21)
    prices = 75 * np.exp(np.cumsum(rng.normal(0.0001, 0.018, size=350)))
    return pd.Series(prices, index=dates)


def test_density_backtest_returns_ranking(sample_close):
    result = walk_forward_density_backtest(
        sample_close,
        horizon_days=5,
        n_folds=6,
        min_train=100,
        n_sims=80,
        methods=("block_bootstrap", "gbm"),
        seed=3,
    )
    assert result["n_folds"] >= 3
    ranking = result["ranking"]
    assert not ranking.empty
    assert "Mean CRPS" in ranking.columns
    assert ranking["Mean CRPS"].is_monotonic_increasing or len(ranking) == 1
    assert result["best_method"] in ("block_bootstrap", "gbm")


def test_density_backtest_no_lookahead_structure(sample_close):
    """Cada fold deve ter realized definido e CRPS finito para pelo menos um método."""
    result = walk_forward_density_backtest(
        sample_close,
        horizon_days=3,
        n_folds=5,
        min_train=120,
        n_sims=50,
        methods=("gbm",),
        seed=0,
    )
    details = result["fold_details"]
    assert not details.empty
    assert details["crps"].notna().any()
    # realized deve ser preço positivo
    assert (details["realized"] > 0).all()


def test_rank_mc_methods_by_crps_shortcut(sample_close):
    df = rank_mc_methods_by_crps(
        sample_close, horizon_days=5, n_folds=5, n_sims=60, seed=1
    )
    assert isinstance(df, pd.DataFrame)
    if not df.empty:
        assert df.iloc[0]["Mean CRPS"] <= df.iloc[-1]["Mean CRPS"]
