"""The divide-and-conquer estimator.

One *loop* is a sweep over every module. Fitting a module means optimizing only
that module's parameters, scored only on that module's variables, while every
other module's parameters stay fixed at the values reached so far. Loops repeat
until the total cost stops improving or ``max_loops`` is reached.

    theta <- initial values
    repeat up to max_loops times:
        for each module m (in the configured order):
            theta[m] <- argmin  cost_m(theta[m] ; theta[not m] fixed)
        stop if the total cost has stopped improving

Because each module fit sees only a handful of free parameters, the individual
optimizations stay tractable even when the full model has dozens of parameters.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from . import optimizers
from .config import ModuleSpec, StudyConfig, load_config
from .data import ExperimentData
from .model import SimulationModel
from .objective import Problem
from .result import FitResult, LoopRecord, ModuleStep

__all__ = ["ModularEstimator", "fit", "fit_from_file"]

logger = logging.getLogger("pymodest")

ProgressCallback = Callable[[ModuleStep], None]


class ModularEstimator:
    """Fit a shared parameter set module by module, over repeated loops."""

    def __init__(
        self,
        config: StudyConfig,
        problem: Optional[Problem] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.config = config
        self.problem = problem if problem is not None else Problem(config)
        seed = config.fitting.seed
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        self._order_rng = np.random.default_rng(None if seed is None else seed + 1)

    # -- convenience accessors --------------------------------------------
    @property
    def models(self) -> Dict[str, SimulationModel]:
        return self.problem.models

    @property
    def data(self) -> Dict[str, ExperimentData]:
        return self.problem.data

    @property
    def values(self) -> Dict[str, float]:
        return self.problem.snapshot()

    # -- module ordering ---------------------------------------------------
    def module_order(self, loop: int) -> List[ModuleSpec]:
        setting = self.config.fitting.module_order
        modules = self.config.modules
        if isinstance(setting, list):
            index = {m.id: m for m in modules}
            return [index[mid] for mid in setting]
        if setting == "random":
            order = self._order_rng.permutation(len(modules))
            return [modules[i] for i in order]
        if setting == "round_robin_reversed":
            return list(modules) if loop % 2 == 1 else list(reversed(modules))
        return list(modules)

    # -- one module fit ----------------------------------------------------
    def fit_module(self, module: ModuleSpec, loop: int = 1) -> ModuleStep:
        """Optimize one module's parameters, holding all others fixed."""
        objective = self.problem.objective_for(module)
        spec = self.config.optimizer_for(module, loop)

        cost_before = self.problem.module_cost(module)
        total_before = self.problem.total_cost()
        before_values = self.problem.snapshot()

        started = time.perf_counter()
        result = optimizers.run(
            spec.name,
            objective,
            x0=objective.x0,
            bounds=objective.bounds,
            rng=self.rng,
            **spec.options,
        )
        elapsed = time.perf_counter() - started

        # A backend may return a point worse than the best it saw, or worse than
        # where it started. Prefer the best point actually evaluated.
        candidate_x, candidate_cost = result.x, result.fun
        if objective.best_x is not None and objective.best_cost < candidate_cost:
            candidate_x, candidate_cost = objective.best_x, objective.best_cost

        accepted = candidate_cost < cost_before
        reason = ""
        if accepted:
            objective.commit(candidate_x)
            if self.config.fitting.accept == "total":
                # modules are coupled through the shared model, so a step that
                # helps one module can hurt another; under this policy, don't
                # take it unless the study as a whole improves
                total_after = self.problem.total_cost()
                if total_after > total_before:
                    self.problem.set_values(before_values)
                    accepted = False
                    candidate_cost = cost_before
                    reason = " (rejected: total cost would rise)"
        else:
            self.problem.set_values(before_values)
            candidate_cost = cost_before
            reason = " (rejected: no improvement)"

        step = ModuleStep(
            loop=loop,
            module=module.id,
            optimizer=result.backend or spec.name,
            cost_before=cost_before,
            cost_after=candidate_cost,
            total_before=total_before,
            total_after=self.problem.total_cost(),
            parameters={name: self.problem.values[name] for name in objective.names},
            nfev=int(result.nfev),
            seconds=elapsed,
            accepted=accepted,
            message=result.message,
        )
        logger.info(
            "loop %d | module %-16s | %-22s | cost %.6g -> %.6g%s | %5d evals | %.2fs",
            loop, module.id, step.optimizer, cost_before, candidate_cost,
            reason, step.nfev, elapsed,
        )
        return step

    # -- the outer loop ----------------------------------------------------
    def run(
        self,
        max_loops: Optional[int] = None,
        callback: Optional[ProgressCallback] = None,
    ) -> FitResult:
        """Run the divide-and-conquer loop and return the fitted result."""
        fitting = self.config.fitting
        max_loops = int(fitting.max_loops if max_loops is None else max_loops)

        initial_values = self.problem.snapshot()
        initial_cost = self.problem.total_cost()
        logger.info(
            "starting %s: %d module(s), %d parameter(s), %d dataset(s); initial cost %.6g",
            self.config.name, len(self.config.modules),
            len(self.problem.parameter_specs), len(self.data), initial_cost,
        )

        loops: List[LoopRecord] = []
        best_values = dict(initial_values)
        best_cost = initial_cost
        previous_cost = initial_cost
        stalled = 0
        stop_reason = f"reached max_loops ({max_loops})"
        converged = False
        started = time.perf_counter()

        for loop in range(1, max_loops + 1):
            loop_started = time.perf_counter()
            best_before_loop = best_cost
            record = LoopRecord(loop=loop)

            for module in self.module_order(loop):
                if not module.free_parameters:
                    logger.debug(
                        "loop %d | module %s has no free parameters; skipping", loop, module.id
                    )
                    continue
                step = self.fit_module(module, loop)
                record.steps.append(step)
                if callback is not None:
                    callback(step)

            record.module_costs = {
                m.id: self.problem.module_cost(m) for m in self.config.modules
            }
            record.total_cost = self.problem.total_cost()
            record.parameters = self.problem.snapshot()
            record.seconds = time.perf_counter() - loop_started

            denominator = max(abs(previous_cost), 1e-30)
            record.relative_improvement = (previous_cost - record.total_cost) / denominator
            loops.append(record)

            if record.total_cost < best_cost:
                best_cost = record.total_cost
                best_values = dict(record.parameters)

            logger.info(
                "loop %d complete | total cost %.6g (relative improvement %.3g) | %.2fs",
                loop, record.total_cost, record.relative_improvement, record.seconds,
            )

            # Progress is judged against the best cost seen so far, not the
            # previous loop: with coupled modules the total can oscillate, and
            # recovering from a bad loop is not the same as making progress.
            gain = best_before_loop - record.total_cost
            threshold = max(fitting.tol * abs(best_before_loop), fitting.atol)
            if gain < threshold:
                stalled += 1
                if stalled >= fitting.patience:
                    converged = True
                    stop_reason = (
                        f"converged: the total cost improved by less than "
                        f"{threshold:.3g} for {stalled} consecutive loop(s)"
                    )
                    break
            else:
                stalled = 0
            previous_cost = record.total_cost

        # leave the problem holding the best parameters found, not the last ones
        self.problem.set_values(best_values)
        elapsed = time.perf_counter() - started

        result = FitResult(
            parameters=best_values,
            cost=best_cost,
            module_costs={m.id: self.problem.module_cost(m) for m in self.config.modules},
            loops=loops,
            converged=converged,
            stop_reason=stop_reason,
            n_evaluations=self.problem.n_evaluations,
            seconds=elapsed,
            initial_parameters=initial_values,
            initial_cost=initial_cost,
            study=self.config.name,
        )
        logger.info(
            "finished: cost %.6g -> %.6g after %d loop(s), %d evaluation(s) in %.1fs (%s)",
            initial_cost, best_cost, len(loops), result.n_evaluations, elapsed, stop_reason,
        )
        return result

    # -- exports -----------------------------------------------------------
    def write_predictions(
        self, directory: Path, values: Optional[Dict[str, float]] = None,
        n_points: int = 200,
    ) -> List[Path]:
        """Write a dense simulated trace per dataset, beside the observations."""
        import pandas as pd

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        written: List[Path] = []
        for data in self.data.values():
            dense = self.problem.predictions(data, values, n_points=n_points)
            frame = pd.DataFrame({"time": dense["time"]})
            for variable in data.variables:
                if variable in dense:
                    frame[f"{variable}_fit"] = dense[variable]
            observed = pd.DataFrame({"time": data.time_grid()})
            for variable, measurement in data.measurements.items():
                lookup = dict(zip(measurement.times, measurement.values))
                observed[f"{variable}_obs"] = [
                    lookup.get(t, np.nan) for t in observed["time"]
                ]
            merged = pd.merge(frame, observed, on="time", how="outer").sort_values("time")
            path = directory / f"predictions_{data.id}.csv"
            merged.to_csv(path, index=False)
            written.append(path)
        return written


# --------------------------------------------------------------------------
# functional entry points
# --------------------------------------------------------------------------

def fit(
    config: StudyConfig,
    max_loops: Optional[int] = None,
    callback: Optional[ProgressCallback] = None,
) -> FitResult:
    """Run a study that has already been loaded."""
    return ModularEstimator(config).run(max_loops=max_loops, callback=callback)


def fit_from_file(path, max_loops: Optional[int] = None) -> FitResult:
    """Load a TOML study configuration and fit it."""
    return fit(load_config(path), max_loops=max_loops)
