import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analytics import signals


def test_volatility_percentile_returns_none_for_short_series():
    close = pd.Series([100, 101, 102, 103])
    assert signals.volatility_percentile(close) is None


def test_volatility_percentile_detects_recent_vol_shock():
    """Série com vol baixa constante seguida de choque nos últimos 25 dias
    -> o percentil da vol atual deve ficar próximo do topo da distribuição."""
    rng = np.random.default_rng(1)
    n = 400
    dates = pd.bdate_range("2023-01-01", periods=n)
    rets = rng.normal(0, 0.008, n)
    rets[-25:] = rng.normal(0, 0.035, 25)
    close = pd.Series(100 * np.exp(np.cumsum(rets)), index=dates)

    pct = signals.volatility_percentile(close)
    assert pct is not None
    assert pct > 0.85


def test_volatility_percentile_low_for_calm_recent_period():
    """Série com choque de vol no meio, mas calma nos últimos 21 dias ->
    percentil atual deve ficar baixo."""
    rng = np.random.default_rng(2)
    n = 400
    dates = pd.bdate_range("2023-01-01", periods=n)
    rets = rng.normal(0, 0.008, n)
    rets[150:200] = rng.normal(0, 0.04, 50)  # choque no meio, não no fim
    close = pd.Series(100 * np.exp(np.cumsum(rets)), index=dates)

    pct = signals.volatility_percentile(close)
    assert pct is not None
    assert pct < 0.5


@pytest.mark.parametrize("pct,expected_substr", [
    (0.95, "Elevada"),
    (0.5, "Normal"),
    (0.05, "Baixa"),
    (None, "N/D"),
])
def test_vol_regime_label_thresholds(pct, expected_substr):
    assert expected_substr in signals.vol_regime_label(pct)


def test_return_zscore_none_for_short_series():
    close = pd.Series(np.arange(10.0))
    assert signals.return_zscore(close, window=63) is None


def test_return_zscore_flags_extreme_move():
    """Retorno do último dia muito maior que os anteriores -> z-score
    grande em magnitude."""
    rng = np.random.default_rng(3)
    n = 200
    dates = pd.bdate_range("2023-01-01", periods=n)
    rets = rng.normal(0, 0.01, n)
    rets[-1] = 0.15  # choque isolado no último dia
    close = pd.Series(100 * np.exp(np.cumsum(rets)), index=dates)

    z = signals.return_zscore(close, window=63)
    assert z is not None
    assert abs(z) > 3


@pytest.mark.parametrize("momentum,expected_substr", [
    (0.10, "alta"),
    (-0.10, "baixa"),
    (0.0, "Neutro"),
])
def test_momentum_label(momentum, expected_substr):
    assert expected_substr in signals.momentum_label(momentum)


def test_build_signal_row_has_expected_keys():
    rng = np.random.default_rng(4)
    n = 300
    dates = pd.bdate_range("2023-01-01", periods=n)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=dates)

    row = signals.build_signal_row(close, momentum=0.03)
    expected_keys = {"vol_percentile", "vol_regime", "return_zscore", "extreme_move", "momentum_label"}
    assert expected_keys.issubset(row.keys())
    assert isinstance(row["extreme_move"], bool)