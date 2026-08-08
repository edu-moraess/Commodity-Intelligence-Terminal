"""
Utils — Health Check (Produção)
===============================
Verifica disponibilidade de fontes de dados, pacotes opcionais e
smoke test do motor Monte Carlo. Usado no sidebar e para diagnóstico.

Não bloqueia a aplicação — apenas reporta status.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""
    latency_ms: float | None = None


@dataclass
class HealthReport:
    timestamp: str
    overall_ok: bool
    checks: list[CheckResult] = field(default_factory=list)
    version: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "overall_ok": self.overall_ok,
            "version": self.version,
            "checks": [
                {
                    "name": c.name,
                    "ok": c.ok,
                    "detail": c.detail,
                    "latency_ms": c.latency_ms,
                }
                for c in self.checks
            ],
        }


def _check_package(name: str, import_name: str | None = None) -> CheckResult:
    mod = import_name or name
    try:
        __import__(mod)
        return CheckResult(name=f"pkg:{name}", ok=True, detail="installed")
    except ImportError:
        return CheckResult(name=f"pkg:{name}", ok=False, detail="not installed (optional)")


def _check_mc_engine() -> CheckResult:
    import time
    t0 = time.perf_counter()
    try:
        from forecasting.mc_engine import monte_carlo_paths, scenario_summary
        dates = pd.bdate_range("2024-01-01", periods=120)
        rng = np.random.default_rng(0)
        close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 120))), index=dates)
        paths = monte_carlo_paths(close, horizon_days=5, n_sims=30, method="gbm", seed=1)
        assert paths.shape == (30, 5)
        assert (paths > 0).all()
        s = scenario_summary(close, horizon_days=5, n_sims=30, method="gbm", seed=1)
        assert "expected_price" in s
        dt = (time.perf_counter() - t0) * 1000
        return CheckResult(name="mc_engine", ok=True, detail="smoke ok", latency_ms=round(dt, 1))
    except Exception as exc:
        dt = (time.perf_counter() - t0) * 1000
        return CheckResult(name="mc_engine", ok=False, detail=str(exc)[:120], latency_ms=round(dt, 1))


def _check_data_layer() -> CheckResult:
    try:
        from data.sources.synthetic import generate_price_series
        df = generate_price_series("CL=F", days=30)
        if df is None or df.empty or "Close" not in df.columns:
            return CheckResult(name="data_synthetic", ok=False, detail="empty synthetic")
        return CheckResult(name="data_synthetic", ok=True, detail=f"rows={len(df)}")
    except Exception as exc:
        return CheckResult(name="data_synthetic", ok=False, detail=str(exc)[:120])


def run_health_checks(version: str = "unknown") -> HealthReport:
    """Executa checks leves (sem rede externa obrigatória)."""
    checks: list[CheckResult] = []

    checks.append(_check_data_layer())
    checks.append(_check_mc_engine())

    for pkg, mod in [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scipy", "scipy"),
        ("sklearn", "sklearn"),
        ("arch", "arch"),
        ("yfinance", "yfinance"),
    ]:
        checks.append(_check_package(pkg, mod))

    # overall: críticos devem passar (data + mc + numpy/pandas)
    critical = {"data_synthetic", "mc_engine", "pkg:numpy", "pkg:pandas"}
    overall = all(c.ok for c in checks if c.name in critical)

    return HealthReport(
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        overall_ok=overall,
        checks=checks,
        version=version,
    )
