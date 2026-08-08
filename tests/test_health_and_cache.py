"""Fase 5 — health + production cache smoke tests."""
import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.health import run_health_checks
from config.settings import APP_VERSION


def test_health_checks_critical_pass():
    report = run_health_checks(version=APP_VERSION)
    assert report.version == APP_VERSION
    assert report.overall_ok is True
    names = {c.name for c in report.checks}
    assert "mc_engine" in names
    assert "data_synthetic" in names
    assert "pkg:numpy" in names


def test_health_report_to_dict():
    report = run_health_checks(version="test")
    d = report.to_dict()
    assert "checks" in d
    assert d["overall_ok"] is True


def test_scenario_fingerprint_stable():
    from forecasting.scenario_cache import _fingerprint
    a = _fingerprint("CL=F", 70.0, "2024-01-01", 30, 500, "gbm", 42, None)
    b = _fingerprint("CL=F", 70.0, "2024-01-01", 30, 500, "gbm", 42, None)
    c = _fingerprint("CL=F", 70.1, "2024-01-01", 30, 500, "gbm", 42, None)
    assert a == b
    assert a != c
