import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analytics import risk


@pytest.fixture
def sample_close():
    dates = pd.bdate_range("2024-01-01", periods=300)
    rng = np.random.default_rng(7)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, size=300)))
    return pd.Series(prices, index=dates)


def test_historical_var_is_positive_for_normal_series(sample_close):
    v = risk.historical_var(sample_close)
    assert v > 0  # convenção: VaR reportado como perda positiva


def test_cvar_greater_or_equal_than_var(sample_close):
    var = risk.historical_var(sample_close)
    cvar = risk.historical_cvar(sample_close)
    assert cvar >= var - 1e-9


def test_parametric_var_reasonable_range(sample_close):
    v = risk.parametric_var(sample_close)
    assert 0 < v < 1  # perda diária de 0-100% é o range logicamente válido


def test_stress_test_default_shocks_row_count(sample_close):
    df = risk.stress_test(sample_close)
    assert len(df) == 7
    assert set(df.columns) == {"choque", "preco_atual", "preco_stress", "variacao_absoluta"}


def test_stress_test_negative_shock_reduces_price(sample_close):
    df = risk.stress_test(sample_close, shocks_pct=[-0.5])
    row = df.iloc[0]
    assert row["preco_stress"] < row["preco_atual"]


def test_risk_summary_contains_expected_keys(sample_close):
    summary = risk.risk_summary(sample_close)
    assert {"var_historico", "var_parametrico", "cvar", "confianca", "janela_dias"} <= summary.keys()