"""The optimizer interface and its registry.

A backend is a callable

    run(objective, x0, bounds, rng, **options) -> OptimizerResult

``objective`` is a :class:`~pymodest.objective.ModuleObjective`: call it for a
scalar cost, or use ``objective.residuals`` for a least-squares vector.
Register a new backend with :func:`register`, then name it in the TOML config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["OptimizerResult", "OptimizerError", "register", "get_optimizer", "available", "run"]


class OptimizerError(RuntimeError):
    """Raised when a backend is unknown or misconfigured."""


@dataclass
class OptimizerResult:
    """Outcome of one module fit, in search space."""

    x: np.ndarray
    fun: float
    nfev: int = 0
    nit: int = 0
    success: bool = True
    message: str = ""
    backend: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=float)
        self.fun = float(self.fun)


_REGISTRY: Dict[str, Callable[..., OptimizerResult]] = {}
_ALIASES: Dict[str, str] = {}


def register(name: str, *aliases: str) -> Callable:
    """Decorator registering a backend under ``name`` and any aliases."""

    def decorator(fn: Callable[..., OptimizerResult]) -> Callable[..., OptimizerResult]:
        _REGISTRY[name] = fn
        for alias in aliases:
            _ALIASES[alias] = name
        return fn

    return decorator


def _canonical(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    return _ALIASES.get(key, key)


def get_optimizer(name: str) -> Callable[..., OptimizerResult]:
    key = _canonical(name)
    if key not in _REGISTRY:
        raise OptimizerError(
            f"unknown optimizer '{name}'; available: {available()}"
        )
    return _REGISTRY[key]


def available() -> List[str]:
    """Names of all registered backends."""
    return sorted(_REGISTRY)


def run(
    name: str,
    objective,
    x0: Optional[Sequence[float]] = None,
    bounds: Optional[Sequence[Tuple[float, float]]] = None,
    rng: Optional[np.random.Generator] = None,
    **options: Any,
) -> OptimizerResult:
    """Dispatch to a registered backend, filling in defaults from the objective."""
    backend = get_optimizer(name)
    x0 = np.asarray(objective.x0 if x0 is None else x0, dtype=float)
    bounds = list(objective.bounds if bounds is None else bounds)
    rng = np.random.default_rng() if rng is None else rng
    result = backend(objective, x0=x0, bounds=bounds, rng=rng, **options)
    result.backend = _canonical(name)
    lower = np.array([b[0] for b in bounds], dtype=float)
    upper = np.array([b[1] for b in bounds], dtype=float)
    result.x = np.clip(result.x, lower, upper)
    return result


def sample_uniform(
    rng: np.random.Generator, bounds: Sequence[Tuple[float, float]], size: int
) -> np.ndarray:
    """``size`` points drawn uniformly inside ``bounds`` (search space)."""
    lower = np.array([b[0] for b in bounds], dtype=float)
    upper = np.array([b[1] for b in bounds], dtype=float)
    return lower + rng.random((size, lower.size)) * (upper - lower)


def latin_hypercube(
    rng: np.random.Generator, bounds: Sequence[Tuple[float, float]], size: int
) -> np.ndarray:
    """A latin-hypercube sample, for better coverage than plain uniform draws."""
    lower = np.array([b[0] for b in bounds], dtype=float)
    upper = np.array([b[1] for b in bounds], dtype=float)
    d = lower.size
    cut = np.linspace(0.0, 1.0, size + 1)
    points = np.empty((size, d))
    for j in range(d):
        u = rng.random(size)
        strata = cut[:size] + u * (cut[1] - cut[0])
        points[:, j] = rng.permutation(strata)
    return lower + points * (upper - lower)
