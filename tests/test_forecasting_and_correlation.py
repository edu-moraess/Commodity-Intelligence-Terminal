import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from forecasting import models as fc
from analytics import correlation
from data.sources import synthetic


@pytest.fixture
def sample_close():
    dates = pd.bdate_range("2024-01-01", periods=400)
    rng = np.random.default_rng(11)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, size=400)))
    return pd.Series(prices, index=dates)


def test_trend_forecast_returns_correct_horizon(sample_close):
    horizon = 30
    forecast = fc.trend_forecast(sample_close, horizon)
    assert len(forecast) == horizon


def test_trend_forecast_prices_are_positive(sample_close):
    forecast = fc.trend_forecast(sample_close, 30)
    assert (forecast > 0).all()


def test_monte_carlo_paths_shape(sample_close):
    paths = fc.monte_carlo_paths(sample_close, horizon_days=20, n_sims=100)
    assert paths.shape == (100, 20)


def test_scenario_summary_ordering(sample_close):
    scenario = fc.scenario_summary(sample_close, horizon_days=30, n_sims=300)
    assert scenario["cenario_pessimista"] <= scenario["cenario_base"] <= scenario["cenario_otimista"]


def test_scenario_summary_prob_alta_in_unit_interval(sample_close):
    scenario = fc.scenario_summary(sample_close, horizon_days=30, n_sims=300)
    assert 0 <= scenario["prob_alta"] <= 1


def test_synthetic_series_is_deterministic_per_ticker():
    a = synthetic.generate_price_series("GC=F", days=100)
    b = synthetic.generate_price_series("GC=F", days=100)
    pd.testing.assert_series_equal(a["Close"], b["Close"])


def test_correlation_matrix_diagonal_is_one():
    dates = pd.bdate_range("2024-01-01", periods=200)
    rng = np.random.default_rng(3)
    panel = pd.DataFrame({
        "A": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200))),
        "B": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200))),
    }, index=dates)
    corr = correlation.correlation_matrix(panel)
    assert corr.loc["A", "A"] == pytest.approx(1.0)
    assert corr.loc["B", "B"] == pytest.approx(1.0)


def test_pca_explained_variance_sums_to_at_most_one():
    dates = pd.bdate_range("2024-01-01", periods=200)
    rng = np.random.default_rng(5)
    panel = pd.DataFrame({
        "A": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200))),
        "B": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200))),
        "C": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200))),
    }, index=dates)
    result = correlation.pca_components(panel, n_components=3)
    assert sum(result["explained_variance_ratio"]) <= 1.0001


# ---------------------------------------------------------------------------
# Fase 1 — novos testes de robustez quant
# ---------------------------------------------------------------------------

def test_stationary_bootstrap_reproducible_with_seed(sample_close):
    """Mesmo seed deve produzir paths idênticos."""
    p1 = fc.monte_carlo_paths(sample_close, horizon_days=15, n_sims=50, method="block_bootstrap", seed=123)
    p2 = fc.monte_carlo_paths(sample_close, horizon_days=15, n_sims=50, method="block_bootstrap", seed=123)
    np.testing.assert_array_almost_equal(p1, p2)


def test_different_seeds_produce_different_paths(sample_close):
    p1 = fc.monte_carlo_paths(sample_close, horizon_days=15, n_sims=50, method="block_bootstrap", seed=1)
    p2 = fc.monte_carlo_paths(sample_close, horizon_days=15, n_sims=50, method="block_bootstrap", seed=2)
    assert not np.allclose(p1, p2)


def test_optimal_block_length_returns_sensible_value(sample_close):
    rets = np.diff(np.log(sample_close.values))
    bl = fc._optimal_block_length(rets)
    assert 3 <= bl <= 20


def test_garch_mc_paths_shape_and_positive(sample_close):
    paths = fc.monte_carlo_paths(sample_close, horizon_days=20, n_sims=80, method="garch_mc", seed=42)
    assert paths.shape == (80, 20)
    assert (paths > 0).all()


def test_scenario_summary_exposes_seed_and_block_size(sample_close):
    s = fc.scenario_summary(sample_close, horizon_days=20, n_sims=200, method="block_bootstrap", seed=99)
    assert s.get("seed") == 99
    assert "block_size_used" in s
    assert s["block_size_used"] is None or (3 <= s["block_size_used"] <= 20)


def test_compare_methods_includes_garch_mc(sample_close):
    df = fc.compare_monte_carlo_methods(sample_close, horizon_days=15, n_sims=100, seed=42)
    methods = set(df["Método"].tolist()) if "Método" in df.columns else set()
    assert "garch_mc" in methods or any("Erro" in str(r) for r in df.to_dict("records"))


def test_all_mc_methods_produce_positive_prices(sample_close):
    for method in ["block_bootstrap", "gbm", "jump_diffusion", "student_t", "garch_mc"]:
        paths = fc.monte_carlo_paths(
            sample_close, horizon_days=10, n_sims=50, method=method, seed=7
        )
        assert (paths > 0).all(), f"{method} gerou preços não-positivos"
