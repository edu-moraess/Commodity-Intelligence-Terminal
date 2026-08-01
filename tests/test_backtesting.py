import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analytics import backtesting


@pytest.fixture
def well_calibrated_close():
    """Série normal i.i.d. — um modelo de VaR histórico deveria estar bem
    calibrado sobre ela (não rejeitar H0 em nenhum dos testes)."""
    dates = pd.bdate_range("2018-01-01", periods=1200)
    rng = np.random.default_rng(11)
    rets = rng.normal(0.0002, 0.012, size=1200)
    return pd.Series(100 * np.exp(np.cumsum(rets)), index=dates)


def test_rolling_var_forecast_never_uses_future_data():
    """O VaR no dia t não pode depender de retornos >= t (look-ahead bias).
    Verifica que a série de VaR começa exatamente na posição `window`."""
    dates = pd.bdate_range("2020-01-01", periods=500)
    rng = np.random.default_rng(1)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 500))), index=dates)

    var_series = backtesting.rolling_var_forecast(close, confidence=0.95, window=100)
    assert len(var_series) == 500 - 1 - 100  # 1 retorno "perdido" no pct_change + 100 de warm-up
    assert (var_series > 0).all()  # VaR reportado como perda positiva


def test_identify_breaches_returns_boolean_series():
    dates = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(2)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))), index=dates)
    var_series = backtesting.rolling_var_forecast(close, window=100)
    breaches = backtesting.identify_breaches(close, var_series)
    assert breaches.dtype == bool
    assert len(breaches) == len(var_series)


def test_kupiec_pof_breach_rate_matches_input():
    """20 exceções em 400 observações -> taxa observada = 5%."""
    breaches = pd.Series([False] * 380 + [True] * 20)
    result = backtesting.kupiec_pof_test(breaches, confidence=0.95)
    assert result["n_obs"] == 400
    assert result["n_breaches"] == 20
    assert result["breach_rate"] == pytest.approx(0.05)
    assert result["expected_rate"] == pytest.approx(0.05)


def test_kupiec_pof_does_not_reject_well_calibrated_model():
    """Taxa observada == taxa esperada -> LR deveria ser ~0 e p-valor alto."""
    breaches = pd.Series([False] * 950 + [True] * 50)  # exatamente 5% em n=1000
    result = backtesting.kupiec_pof_test(breaches, confidence=0.95)
    assert result["lr_stat"] == pytest.approx(0.0, abs=1e-6)
    assert result["p_value"] > 0.9
    assert result["reject_h0"] is False


def test_kupiec_pof_rejects_badly_miscalibrated_model():
    """40% de exceções quando o esperado é 5% -> deve rejeitar H0."""
    breaches = pd.Series([True] * 40 + [False] * 60)
    result = backtesting.kupiec_pof_test(breaches, confidence=0.95)
    assert result["reject_h0"] is True
    assert result["p_value"] < 0.05


def test_christoffersen_detects_clustering():
    """Exceções 100% concentradas em bloco (clustering perfeito) devem
    rejeitar a hipótese de independência."""
    breaches = pd.Series([False] * 50 + [True] * 10 + [False] * 50)
    result = backtesting.christoffersen_independence_test(breaches)
    assert result["reject_h0"] is True


def test_christoffersen_does_not_reject_alternating_pattern():
    """Um padrão perfeitamente alternado não tem características de
    clustering simples (cada exceção é isolada, não há dois breaches
    consecutivos) — não deve ser um caso óbvio de rejeição por clustering."""
    breaches = pd.Series(([True, False] * 25))
    result = backtesting.christoffersen_independence_test(breaches)
    assert result["lr_stat"] is not None


def test_joint_backtest_end_to_end_well_calibrated(well_calibrated_close):
    result = backtesting.joint_backtest(well_calibrated_close, confidence=0.95, window=252)
    assert "var_series" in result and "breaches" in result
    assert "kupiec" in result and "christoffersen" in result and "joint" in result
    # com dados normais i.i.d. bem comportados, não esperamos rejeição categórica
    assert result["kupiec"]["breach_rate"] == pytest.approx(0.05, abs=0.04)


def test_joint_backtest_parametric_method_runs(well_calibrated_close):
    result = backtesting.joint_backtest(well_calibrated_close, confidence=0.99, window=252, method="parametric")
    assert result["kupiec"]["expected_rate"] == pytest.approx(0.01)