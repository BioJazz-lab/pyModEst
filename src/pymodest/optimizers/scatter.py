"""Scatter search, in the form that has proven effective for kinetic models.

The scheme follows the standard five-method template (Glover; Egea et al.'s
eSS for systems biology):

1. *Diversification* -- a large latin-hypercube pool covering the search box.
2. *Reference set* -- half by quality (lowest cost), half by diversity
   (maximum-minimum distance from those already chosen).
3. *Subset generation and combination* -- every ordered pair produces trial
   points in three hyper-rectangles around the segment joining them.
4. *Improvement* -- a short local search from promising trials.
5. *Reference set update* -- keep the best, and replace stagnant members with
   fresh diverse points so the search does not collapse prematurely.
"""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from .base import OptimizerResult, latin_hypercube, register

__all__ = ["run_scatter_search"]


def _distance_matrix(points: np.ndarray, span: np.ndarray) -> np.ndarray:
    scaled = points / np.where(span > 0, span, 1.0)
    diff = scaled[:, None, :] - scaled[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def _build_reference_set(
    pool: np.ndarray, costs: np.ndarray, size: int, span: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Half the reference set by quality, half by diversity."""
    size = min(size, pool.shape[0])
    n_quality = max(size // 2, 1)
    order = np.argsort(costs)
    chosen = list(order[:n_quality])

    remaining = [i for i in order[n_quality:]]
    while len(chosen) < size and remaining:
        dist = _distance_matrix(pool[remaining + chosen], span)
        block = dist[: len(remaining), len(remaining):]
        min_dist = block.min(axis=1)
        pick = int(np.argmax(min_dist))
        chosen.append(remaining.pop(pick))

    idx = np.array(chosen, dtype=int)
    return pool[idx].copy(), costs[idx].copy()


@register("scatter_search", "ss", "scatter", "ess")
def run_scatter_search(
    objective,
    x0: np.ndarray,
    bounds: Sequence[Tuple[float, float]],
    rng: np.random.Generator,
    refset_size: int = 10,
    pool_multiplier: int = 10,
    maxiter: int = 30,
    max_nfev: int = 4000,
    local_search: str = "Nelder-Mead",
    local_maxiter: int = 40,
    local_frequency: int = 2,
    tol: float = 1e-10,
    patience: int = 8,
    seed_with_x0: bool = True,
    **_: Any,
) -> OptimizerResult:
    """Population-based global search with periodic local refinement."""
    lower = np.array([b[0] for b in bounds], dtype=float)
    upper = np.array([b[1] for b in bounds], dtype=float)
    span = upper - lower
    dim = lower.size

    budget = {"n": 0}

    def evaluate(point: np.ndarray) -> float:
        budget["n"] += 1
        return float(objective(np.clip(point, lower, upper)))

    def exhausted() -> bool:
        return budget["n"] >= int(max_nfev)

    # 1. diversification --------------------------------------------------
    pool_size = max(int(refset_size) * int(pool_multiplier), int(refset_size) + 1)
    pool = latin_hypercube(rng, bounds, pool_size)
    if seed_with_x0:
        pool[0] = np.clip(np.asarray(x0, dtype=float), lower, upper)
    pool_costs = np.array([evaluate(p) for p in pool], dtype=float)

    # 2. reference set ----------------------------------------------------
    refset, ref_costs = _build_reference_set(pool, pool_costs, int(refset_size), span)
    b = refset.shape[0]

    best_idx = int(np.argmin(ref_costs))
    best_x = refset[best_idx].copy()
    best_cost = float(ref_costs[best_idx])

    stalled = 0
    iterations = 0

    for it in range(int(maxiter)):
        iterations = it + 1
        if exhausted():
            break

        # 3. subset generation and combination ----------------------------
        trials: List[np.ndarray] = []
        for i in range(b):
            for j in range(b):
                if i == j:
                    continue
                xi, xj = refset[i], refset[j]
                d = (xj - xi) / 2.0
                alpha = 1.0 if ref_costs[j] < ref_costs[i] else -1.0
                beta = abs(ref_costs[j] - ref_costs[i]) / (
                    abs(ref_costs[i]) + abs(ref_costs[j]) + 1e-30
                )
                r = rng.random(dim)
                # three regions: before xi, between the pair, beyond xj
                trials.append(xi - d * (1.0 + alpha * beta) * r)
                trials.append(xi + d * (1.0 + r))
                trials.append(xj + d * (1.0 + alpha * beta) * r)

        if not trials:
            break
        candidates = np.clip(np.array(trials), lower, upper)
        # cap the work per iteration when the reference set is large
        max_trials = max(4 * b, 20)
        if candidates.shape[0] > max_trials:
            keep = rng.choice(candidates.shape[0], size=max_trials, replace=False)
            candidates = candidates[keep]

        cand_costs = []
        for point in candidates:
            if exhausted():
                break
            cand_costs.append(evaluate(point))
        candidates = candidates[: len(cand_costs)]
        cand_costs = np.array(cand_costs, dtype=float)
        if candidates.size == 0:
            break

        # 4. improvement --------------------------------------------------
        if local_search and (it % max(int(local_frequency), 1) == 0) and not exhausted():
            start = candidates[int(np.argmin(cand_costs))]
            try:
                local = minimize(
                    lambda p: evaluate(p),
                    start,
                    method=local_search,
                    bounds=list(zip(lower, upper)) if local_search != "Nelder-Mead" else None,
                    options={"maxiter": int(local_maxiter), "maxfev": int(local_maxiter)},
                )
                refined = np.clip(local.x, lower, upper)
                candidates = np.vstack([candidates, refined])
                cand_costs = np.append(cand_costs, float(local.fun))
            except Exception:  # pragma: no cover - local search is best effort
                pass

        # 5. reference set update -----------------------------------------
        merged = np.vstack([refset, candidates])
        merged_costs = np.concatenate([ref_costs, cand_costs])
        # drop duplicates before rebuilding, so the set keeps its spread
        _, unique = np.unique(np.round(merged / np.where(span > 0, span, 1.0), 9),
                              axis=0, return_index=True)
        unique = np.sort(unique)
        refset, ref_costs = _build_reference_set(
            merged[unique], merged_costs[unique], b, span
        )

        current = float(np.min(ref_costs))
        if current < best_cost - tol:
            best_cost = current
            best_x = refset[int(np.argmin(ref_costs))].copy()
            stalled = 0
        else:
            stalled += 1
            if stalled >= int(patience):
                break
            # restart the diverse half to escape stagnation
            if stalled % 3 == 0 and not exhausted():
                n_keep = max(b // 2, 1)
                fresh = latin_hypercube(rng, bounds, b - n_keep)
                fresh_costs = np.array([evaluate(p) for p in fresh], dtype=float)
                order = np.argsort(ref_costs)[:n_keep]
                refset = np.vstack([refset[order], fresh])
                ref_costs = np.concatenate([ref_costs[order], fresh_costs])

    return OptimizerResult(
        x=best_x,
        fun=best_cost,
        nfev=budget["n"],
        nit=iterations,
        success=True,
        message=f"scatter search finished after {iterations} iteration(s)",
        extra={"refset_size": int(refset.shape[0])},
    )
