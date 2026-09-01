"""SciPy-backed optimizers: differential evolution, least squares, minimize."""

from __future__ import annotations

from typing import Any, Sequence, Tuple

import numpy as np
from scipy.optimize import OptimizeResult, differential_evolution, least_squares, minimize

from .base import OptimizerResult, register

__all__ = ["run_differential_evolution", "run_least_squares", "run_minimize"]


def _seed_from(rng: np.random.Generator) -> int:
    return int(rng.integers(0, 2**31 - 1))


@register("differential_evolution", "de", "diffevo")
def run_differential_evolution(
    objective,
    x0: np.ndarray,
    bounds: Sequence[Tuple[float, float]],
    rng: np.random.Generator,
    maxiter: int = 100,
    popsize: int = 15,
    tol: float = 1e-6,
    mutation: Any = (0.5, 1.0),
    recombination: float = 0.7,
    polish: bool = True,
    init: str = "latinhypercube",
    strategy: str = "best1bin",
    workers: int = 1,
    use_x0: bool = True,
    **_: Any,
) -> OptimizerResult:
    """Bounded global search. The robust default for wide biological ranges.

    ``use_x0`` seeds the population with the current parameter values, so a
    later loop starts from where the previous one left off.
    """
    kwargs = dict(
        bounds=list(bounds),
        maxiter=int(maxiter),
        popsize=int(popsize),
        tol=float(tol),
        mutation=mutation,
        recombination=float(recombination),
        polish=bool(polish),
        init=init,
        strategy=strategy,
        workers=int(workers),
        seed=_seed_from(rng),
        updating="deferred" if int(workers) != 1 else "immediate",
    )
    if use_x0:
        kwargs["x0"] = np.asarray(x0, dtype=float)
    result: OptimizeResult = differential_evolution(objective, **kwargs)
    return OptimizerResult(
        x=result.x,
        fun=result.fun,
        nfev=int(getattr(result, "nfev", 0)),
        nit=int(getattr(result, "nit", 0)),
        success=bool(getattr(result, "success", True)),
        message=str(getattr(result, "message", "")),
    )


@register("least_squares", "trf", "lsq")
def run_least_squares(
    objective,
    x0: np.ndarray,
    bounds: Sequence[Tuple[float, float]],
    rng: np.random.Generator,
    method: str = "trf",
    max_nfev: int = 200,
    ftol: float = 1e-10,
    xtol: float = 1e-10,
    gtol: float = 1e-10,
    loss: str = "linear",
    f_scale: float = 1.0,
    diff_step: float = 1e-4,
    **_: Any,
) -> OptimizerResult:
    """Local gradient refinement on the residual vector. A good polish step."""
    lower = np.array([b[0] for b in bounds], dtype=float)
    upper = np.array([b[1] for b in bounds], dtype=float)
    start = np.clip(np.asarray(x0, dtype=float), lower, upper)
    # keep the start strictly inside the box: trf rejects points on the bound
    span = upper - lower
    start = np.clip(start, lower + 1e-9 * span, upper - 1e-9 * span)

    result = least_squares(
        objective.residuals,
        start,
        bounds=(lower, upper),
        method=method,
        max_nfev=int(max_nfev),
        ftol=float(ftol),
        xtol=float(xtol),
        gtol=float(gtol),
        loss=loss,
        f_scale=float(f_scale),
        diff_step=float(diff_step),
    )
    # report the module's own aggregated cost, so values are directly
    # comparable with what the other backends return
    cost = objective(result.x)
    return OptimizerResult(
        x=result.x,
        fun=cost,
        nfev=int(getattr(result, "nfev", 0)),
        nit=int(getattr(result, "njev", 0) or 0),
        success=bool(result.success),
        message=str(result.message),
    )


@register("minimize", "lbfgsb", "l_bfgs_b", "nelder_mead", "neldermead")
def run_minimize(
    objective,
    x0: np.ndarray,
    bounds: Sequence[Tuple[float, float]],
    rng: np.random.Generator,
    method: str = "L-BFGS-B",
    maxiter: int = 200,
    tol: float = 1e-10,
    eps: float = 1e-5,
    **options: Any,
) -> OptimizerResult:
    """General local minimizer on the scalar objective.

    Gradient-free methods (Nelder-Mead, Powell) and bounded quasi-Newton
    methods (L-BFGS-B, TNC) are both reachable through ``method``.
    """
    lower = np.array([b[0] for b in bounds], dtype=float)
    upper = np.array([b[1] for b in bounds], dtype=float)
    start = np.clip(np.asarray(x0, dtype=float), lower, upper)

    opts = {"maxiter": int(maxiter)}
    if method.upper() in ("L-BFGS-B", "TNC"):
        opts["eps"] = float(eps)
    opts.update({k: v for k, v in options.items() if not k.startswith("_")})

    result: OptimizeResult = minimize(
        objective,
        start,
        method=method,
        bounds=list(zip(lower, upper)),
        tol=float(tol),
        options=opts,
    )
    return OptimizerResult(
        x=result.x,
        fun=float(result.fun),
        nfev=int(getattr(result, "nfev", 0)),
        nit=int(getattr(result, "nit", 0)),
        success=bool(getattr(result, "success", True)),
        message=str(getattr(result, "message", "")),
    )
