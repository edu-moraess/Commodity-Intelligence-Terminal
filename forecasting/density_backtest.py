"""
Forecasting — Density Backtest (Quant Research)
===============================================
Validação out-of-sample de forecasts densitários via walk-forward.

Para cada fold:
  1. Usa apenas histórico até t (sem leakage)
  2. Gera amostras preditivas h-passos à frente com cada método MC
  3. Observa preço realizado em t+h
  4. Calcula CRPS, PIT e cobertura de intervalo

Ao final:
  - Ranking de métodos por mean CRPS (menor = melhor)
  - Diagnóstico agregado de calibração (PIT KS)
  - Coverage empírica vs nominal

Referências:
  - Gneiting & Raftery (2007) Strictly Proper Scoring Rules
  - Diebold et al. (1998) Evaluating Density Forecasts
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from forecasting.mc_engine import monte_carlo_paths
from analytics.probabilistic_validation import (
    pit_values,
    pit_diagnostics,
    crps_empirical,
    interval_coverage,
)

DEFAULT_METHODS = ("block_bootstrap", "gbm", "jump_diffusion", "student_t", "garch_mc")


def _fold_starts(
    n: int,
    min_train: int,
    horizon: int,
    n_folds: int,
) -> np.ndarray:
    """Índices t onde o forecast usa close[:t] e realiza close[t+horizon-1]."""
    last_valid = n - horizon
    if last_valid <= min_train:
        return np.array([], dtype=int)
    n_folds = min(n_folds, max(1, last_valid - min_train))
    return np.linspace(min_train, last_valid, n_folds, dtype=int)


def walk_forward_density_backtest(
    close: pd.Series,
    horizon_days: int = 5,
    n_folds: int = 12,
    min_train: int = 120,
    n_sims: int = 400,
    methods: Sequence[str] = DEFAULT_METHODS,
    seed: int = 42,
) -> dict:
    """Walk-forward densitário para múltiplos métodos MC.

    Returns
    -------
    dict com:
      ranking : DataFrame ordenado por mean_crps
      pit_by_method : dict method -> pit diagnostics
      fold_details : DataFrame longo (método, fold, crps, realized, ...)
    """
    prices = close.dropna()
    n = len(prices)
    starts = _fold_starts(n, min_train, horizon_days, n_folds)
    if len(starts) == 0:
        return {
            "ranking": pd.DataFrame(),
            "pit_by_method": {},
            "fold_details": pd.DataFrame(),
            "n_folds": 0,
            "horizon_days": horizon_days,
        }

    # samples_by_method[m] = list of (n_sims,) arrays; realized list
    samples_by_method: dict[str, list[np.ndarray]] = {m: [] for m in methods}
    realized_list: list[float] = []
    fold_rows: list[dict] = []

    for fold_i, t in enumerate(starts):
        hist = prices.iloc[:t]
        # preço realizado h pregões à frente (índice t+h-1 no array completo)
        realized_px = float(prices.iloc[t + horizon_days - 1])
        realized_list.append(realized_px)

        for m in methods:
            try:
                paths = monte_carlo_paths(
                    hist,
                    horizon_days=horizon_days,
                    n_sims=n_sims,
                    method=m,
                    seed=seed + fold_i * 17,
                )
                terminal = paths[:, -1]
                samples_by_method[m].append(terminal)

                crps_i = float(crps_empirical(realized_px, terminal))
                p10 = float(np.percentile(terminal, 10))
                p90 = float(np.percentile(terminal, 90))
                p05 = float(np.percentile(terminal, 5))
                p95 = float(np.percentile(terminal, 95))
                fold_rows.append({
                    "fold": fold_i,
                    "t_index": int(t),
                    "method": m,
                    "realized": realized_px,
                    "crps": crps_i,
                    "p50": float(np.percentile(terminal, 50)),
                    "expected": float(np.mean(terminal)),
                    "in_80": int(p10 <= realized_px <= p90),
                    "in_90": int(p05 <= realized_px <= p95),
                })
            except Exception as exc:
                fold_rows.append({
                    "fold": fold_i,
                    "t_index": int(t),
                    "method": m,
                    "realized": realized_px,
                    "crps": np.nan,
                    "p50": np.nan,
                    "expected": np.nan,
                    "in_80": np.nan,
                    "in_90": np.nan,
                    "error": str(exc)[:80],
                })

    realized_arr = np.asarray(realized_list, dtype=float)
    ranking_rows = []
    pit_by_method = {}

    for m in methods:
        samples_list = samples_by_method[m]
        if len(samples_list) == 0:
            continue
        # alinhar: só folds com sucesso
        n_ok = len(samples_list)
        if n_ok != len(realized_arr):
            # alguns métodos falharam em folds — usar apenas folds OK
            # reconstrói realized correspondente a partir de fold_rows
            ok_realized = [
                r["realized"] for r in fold_rows
                if r["method"] == m and np.isfinite(r.get("crps", np.nan))
            ]
            realized_m = np.asarray(ok_realized, dtype=float)
        else:
            realized_m = realized_arr

        if len(samples_list) == 0 or len(realized_m) == 0:
            continue

        sample_mat = np.vstack(samples_list)
        if sample_mat.shape[0] != len(realized_m):
            n_use = min(sample_mat.shape[0], len(realized_m))
            sample_mat = sample_mat[:n_use]
            realized_m = realized_m[:n_use]

        crps_vals = crps_empirical(realized_m, sample_mat)
        mean_crps = float(np.mean(crps_vals))

        pit = pit_values(realized_m, sample_mat)
        pit_diag = pit_diagnostics(pit)
        pit_by_method[m] = pit_diag

        q10 = np.percentile(sample_mat, 10, axis=1)
        q90 = np.percentile(sample_mat, 90, axis=1)
        q05 = np.percentile(sample_mat, 5, axis=1)
        q95 = np.percentile(sample_mat, 95, axis=1)
        cov80 = interval_coverage(realized_m, q10, q90, 0.80)
        cov90 = interval_coverage(realized_m, q05, q95, 0.90)

        ranking_rows.append({
            "Método": m,
            "Mean CRPS": mean_crps,
            "Median CRPS": float(np.median(crps_vals)),
            "Coverage 80%": cov80["coverage"],
            "Coverage 90%": cov90["coverage"],
            "Avg Width 90%": cov90["avg_width"],
            "PIT mean": pit_diag.get("mean", np.nan),
            "PIT KS p-value": pit_diag.get("ks_pvalue", np.nan),
            "PIT calibrated": pit_diag.get("uniform_ok", None),
            "n_folds": int(len(realized_m)),
        })

    ranking = pd.DataFrame(ranking_rows)
    if not ranking.empty:
        ranking = ranking.sort_values("Mean CRPS").reset_index(drop=True)

    return {
        "ranking": ranking,
        "pit_by_method": pit_by_method,
        "fold_details": pd.DataFrame(fold_rows),
        "n_folds": int(len(starts)),
        "horizon_days": horizon_days,
        "n_sims": n_sims,
        "min_train": min_train,
        "best_method": str(ranking.iloc[0]["Método"]) if not ranking.empty else None,
    }


def rank_mc_methods_by_crps(
    close: pd.Series,
    horizon_days: int = 5,
    n_folds: int = 10,
    n_sims: int = 300,
    seed: int = 42,
) -> pd.DataFrame:
    """Atalho: retorna apenas o ranking ordenado por CRPS."""
    result = walk_forward_density_backtest(
        close,
        horizon_days=horizon_days,
        n_folds=n_folds,
        n_sims=n_sims,
        seed=seed,
    )
    return result["ranking"]
