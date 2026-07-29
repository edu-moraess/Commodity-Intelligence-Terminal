import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analytics import metrics


@pytest.fixture
def sample_close():
    dates = pd.bdate_range("2024-01-01", periods=300)
    rng = np.random.default_rng(42)
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, size=300)))
    return pd.Series(prices, index=dates)


def test_daily_returns_length(sample_close):
    rets = metrics.daily_returns(sample_close)
    assert len(rets) == len(sample_close) - 1


def test_pct_change_over_matches_manual_calc(sample_close):
    result = metrics.pct_change_over(sample_close, 10)
    expected = sample_close.iloc[-1] / sample_close.iloc[-11] - 1
    assert result == pytest.approx(expected)


def test_pct_change_over_insufficient_history_returns_none():
    short_series = pd.Series([100, 101, 102])
    assert metrics.pct_change_over(short_series, 10) is None


def test_annualized_volatility_positive(sample_close):
    vol = metrics.annualized_volatility(sample_close)
    assert vol > 0


def test_max_drawdown_is_non_positive(sample_close):
    mdd = metrics.max_drawdown(sample_close)
    assert mdd <= 0


def test_max_drawdown_monotonic_series_is_zero():
    monotonic = pd.Series(np.linspace(100, 200, 50))
    assert metrics.max_drawdown(monotonic) == pytest.approx(0.0)


def test_sharpe_ratio_runs_without_error(sample_close):
    sharpe = metrics.sharpe_ratio(sample_close)
    assert isinstance(sharpe, float)


def test_beta_of_series_with_itself_is_one(sample_close):
    b = metrics.beta(sample_close, sample_close)
    assert b == pytest.approx(1.0, abs=1e-6)


def test_trend_label_uptrend():
    dates = pd.bdate_range("2024-01-01", periods=150)
    prices = pd.Series(np.linspace(100, 200, 150), index=dates)
    assert metrics.trend_label(prices) == "alta"


def test_summary_row_has_expected_keys(sample_close):
    row = metrics.summary_row(sample_close)
    expected_keys = {
        "last_price", "chg_1d", "chg_1w", "chg_1m", "chg_ytd",
        "vol_annual", "sharpe", "sortino", "max_drawdown", "calmar",
        "momentum", "trend",
    }
    assert expected_keys.issubset(row.keys())