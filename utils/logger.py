"""Logging estruturado para o Commodity Intelligence Terminal.

Fase 3: decorator `timed` para medir latência de funções quant críticas
(Monte Carlo, walk-forward, data load) sem acoplar a Streamlit.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from functools import wraps
from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable)

_CONFIGURED = False


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root = logging.getLogger("commodity_terminal")
    root.setLevel(level)
    if not root.handlers:
        root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    _configure_root()
    return logging.getLogger(f"commodity_terminal.{name}")


def timed(label: str | None = None) -> Callable[[F], F]:
    """Decorator: loga duração de execução em INFO.

    Uso:
        @timed("monte_carlo_paths")
        def monte_carlo_paths(...): ...
    """
    def decorator(fn: F) -> F:
        log = get_logger(fn.__module__)
        name = label or fn.__name__

        @wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                dt_ms = (time.perf_counter() - t0) * 1000
                log.info("%s completed in %.1f ms", name, dt_ms)

        return wrapper  # type: ignore[return-value]

    return decorator
