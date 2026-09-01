"""Pluggable optimizer backends for module fits.

Built in: ``differential_evolution``, ``least_squares``, ``minimize``,
``particle_swarm``, ``scatter_search``. Add your own with

    from pymodest.optimizers import register, OptimizerResult

    @register("my_method")
    def my_method(objective, x0, bounds, rng, **options):
        ...
        return OptimizerResult(x=best_x, fun=best_cost)
"""

from .base import (  # noqa: F401
    OptimizerError,
    OptimizerResult,
    available,
    get_optimizer,
    latin_hypercube,
    register,
    run,
    sample_uniform,
)
from . import scatter, scipy_backends, swarm  # noqa: F401  (import registers them)

__all__ = [
    "OptimizerError",
    "OptimizerResult",
    "available",
    "get_optimizer",
    "latin_hypercube",
    "register",
    "run",
    "sample_uniform",
]
