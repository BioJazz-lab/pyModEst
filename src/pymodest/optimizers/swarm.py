"""Particle swarm optimization, implemented directly on numpy.

Standard constriction-free PSO with inertia damping, velocity clamping and
reflecting walls at the parameter bounds. Suits module fits where the cost
surface is rough but each evaluation is a single ODE integration.
"""

from __future__ import annotations

from typing import Any, Sequence, Tuple

import numpy as np

from .base import OptimizerResult, latin_hypercube, register

__all__ = ["run_particle_swarm"]


@register("particle_swarm", "pso", "swarm")
def run_particle_swarm(
    objective,
    x0: np.ndarray,
    bounds: Sequence[Tuple[float, float]],
    rng: np.random.Generator,
    n_particles: int = 24,
    maxiter: int = 60,
    inertia: float = 0.72,
    inertia_final: float = 0.35,
    cognitive: float = 1.5,
    social: float = 1.5,
    velocity_fraction: float = 0.25,
    tol: float = 1e-10,
    patience: int = 15,
    seed_with_x0: bool = True,
    **_: Any,
) -> OptimizerResult:
    """Run PSO in search space and return the global best.

    ``inertia`` decays linearly to ``inertia_final`` so the swarm explores
    early and converges late; the run stops after ``patience`` iterations
    without an improvement greater than ``tol``.
    """
    lower = np.array([b[0] for b in bounds], dtype=float)
    upper = np.array([b[1] for b in bounds], dtype=float)
    span = upper - lower
    dim = lower.size
    n = max(int(n_particles), 4)

    position = latin_hypercube(rng, bounds, n)
    if seed_with_x0:
        position[0] = np.clip(np.asarray(x0, dtype=float), lower, upper)

    vmax = float(velocity_fraction) * span
    velocity = rng.uniform(-1.0, 1.0, size=(n, dim)) * vmax

    cost = np.array([objective(p) for p in position], dtype=float)
    nfev = n

    best_pos = position.copy()
    best_cost = cost.copy()
    g = int(np.argmin(best_cost))
    gbest_pos = best_pos[g].copy()
    gbest_cost = float(best_cost[g])

    stalled = 0
    iterations = 0
    for it in range(int(maxiter)):
        iterations = it + 1
        w = inertia + (inertia_final - inertia) * (it / max(int(maxiter) - 1, 1))
        r1 = rng.random((n, dim))
        r2 = rng.random((n, dim))
        velocity = (
            w * velocity
            + cognitive * r1 * (best_pos - position)
            + social * r2 * (gbest_pos - position)
        )
        velocity = np.clip(velocity, -vmax, vmax)
        position = position + velocity

        # reflect off the walls so particles stay inside the declared range
        below = position < lower
        above = position > upper
        position = np.where(below, lower + (lower - position), position)
        position = np.where(above, upper - (position - upper), position)
        position = np.clip(position, lower, upper)
        velocity[below | above] *= -0.5

        cost = np.array([objective(p) for p in position], dtype=float)
        nfev += n

        improved = cost < best_cost
        best_pos[improved] = position[improved]
        best_cost[improved] = cost[improved]

        g = int(np.argmin(best_cost))
        if best_cost[g] < gbest_cost - tol:
            gbest_cost = float(best_cost[g])
            gbest_pos = best_pos[g].copy()
            stalled = 0
        else:
            stalled += 1
            if stalled >= int(patience):
                break

    return OptimizerResult(
        x=gbest_pos,
        fun=gbest_cost,
        nfev=nfev,
        nit=iterations,
        success=True,
        message=f"particle swarm finished after {iterations} iteration(s)",
        extra={"n_particles": n},
    )
