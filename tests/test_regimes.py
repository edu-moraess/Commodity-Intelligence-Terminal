import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analytics import regimes


def _make_two_regime_series(n=1200, seed=5):
    """Gera uma série com dois regimes de volatilidade reais e persistentes
    — usada em vários testes para validar que o HMM separa corretamente."""
    rng = np.random.default_rng(seed)
    regime_true = np.zeros(n, dtype=int)
    returns_sim = np.zeros(n)
    state = 0
    for t in range(n):
        if state == 0 and rng.random() < 0.02:
            state = 1
        elif state == 1 and rng.random() < 0.05:
            state = 0
        regime_true[t] = state
        returns_sim[t] = rng.normal(0.0005, 0.008) if state == 0 else rng.normal(-0.001, 0.030)

    dates = pd.bdate_range("2019-01-01", periods=n + 1)
    close = pd.Series(np.concatenate([[100], 100 * np.exp(np.cumsum(returns_sim))]), index=dates)
    return close, regime_true


def test_fit_hmm_2state_raises_on_short_series():
    with pytest.raises(ValueError):
        regimes.fit_hmm_2state(np.random.default_rng(0).normal(0, 0.01, 10))


def test_fit_hmm_2state_sigma_ordering_convention():
    """Convenção do módulo: estado 0 = baixa vol, estado 1 = alta vol
    (sigma[0] < sigma[1]) — sempre, independente de onde o EM converge."""
    close, _ = _make_two_regime_series(n=800, seed=1)
    rets = close.pct_change().dropna().values
    result = regimes.fit_hmm_2state(rets, n_iter=100)
    assert result["stds"][0] < result["stds"][1]


def test_hmm_transition_matrix_is_valid_stochastic_matrix():
    close, _ = _make_two_regime_series(n=800, seed=2)
    rets = close.pct_change().dropna().values
    result = regimes.fit_hmm_2state(rets, n_iter=100)
    A = result["transition_matrix"]
    assert A.shape == (2, 2)
    # cada linha deve somar 1 (matriz de transição estocástica válida)
    np.testing.assert_allclose(A.sum(axis=1), [1.0, 1.0], atol=1e-6)
    assert (A >= 0).all() and (A <= 1).all()


def test_hmm_correctly_separates_known_regimes():
    """Teste de recuperação de parâmetro: injeta 2 regimes conhecidos e
    verifica que o Viterbi recupera a sequência real com alta acurácia."""
    close, regime_true = _make_two_regime_series(n=1500, seed=3)
    result = regimes.regime_summary(close, n_iter=150)
    viterbi = result["viterbi_states"].values
    acc = max((viterbi == regime_true).mean(), (viterbi == (1 - regime_true)).mean())
    assert acc > 0.85, f"HMM não separou os regimes adequadamente: acc={acc:.3f}"


def test_regime_summary_state_probs_sum_to_one():
    close, _ = _make_two_regime_series(n=600, seed=4)
    result = regimes.regime_summary(close, n_iter=80)
    row_sums = result["state_probs"].sum(axis=1)
    np.testing.assert_allclose(row_sums.values, np.ones(len(row_sums)), atol=1e-6)


def test_regime_summary_current_regime_label_is_valid():
    close, _ = _make_two_regime_series(n=600, seed=6)
    result = regimes.regime_summary(close, n_iter=80)
    assert result["current_regime_label"] in ("Baixa Volatilidade", "Alta Volatilidade")
    assert 0.0 <= result["current_regime_prob"] <= 1.0


def test_regime_stats_high_vol_regime_has_higher_volatility():
    close, _ = _make_two_regime_series(n=1000, seed=7)
    result = regimes.regime_summary(close, n_iter=100)
    stats = result["regime_stats"]
    vol_baixa = stats.loc["Baixa Volatilidade", "Volatilidade anualizada"]
    vol_alta = stats.loc["Alta Volatilidade", "Volatilidade anualizada"]
    assert vol_alta > vol_baixa