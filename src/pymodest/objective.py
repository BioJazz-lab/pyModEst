"""Objective functions for module-wise fitting.

The estimator holds one :class:`Problem`, which owns the loaded models and
datasets and the current value of every fitted parameter. For each module it
hands out a :class:`ModuleObjective`: a callable over just that module's free
parameters, in optimizer (search) space, that

1. writes the trial values into the shared parameter vector,
2. simulates every relevant model/dataset pair once,
3. collects residuals for only that module's variables, and
4. scales and aggregates them into a scalar cost.

Parameters outside the module keep whatever value the loop has reached, which
is what makes the procedure divide-and-conquer rather than a joint fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .config import ModuleSpec, ObjectiveSpec, ParameterSpec, StudyConfig
from .data import ExperimentData, Measurement, load_datasets
from .model import SimulationFailure, SimulationModel, build_models

__all__ = ["ProblemError", "ResidualBlock", "Problem", "ModuleObjective"]

INFEASIBLE = 1e12


class ProblemError(ValueError):
    """Raised when models, data and modules do not fit together."""


@dataclass(frozen=True)
class ResidualBlock:
    """Residuals contributed by one variable of one dataset."""

    dataset: str
    variable: str
    times: np.ndarray
    observed: np.ndarray
    simulated: np.ndarray
    residuals: np.ndarray
    weight: float


def _interpolate(sim_times: np.ndarray, sim_values: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Values at the measurement times; exact when the grids already match."""
    if sim_times.shape == at.shape and np.allclose(sim_times, at):
        return sim_values
    return np.interp(at, sim_times, sim_values)


def _scale_residuals(
    simulated: np.ndarray,
    measurement: Measurement,
    objective: ObjectiveSpec,
) -> np.ndarray:
    """Turn a raw difference into a comparable residual.

    ``relative``       divide by the observed magnitude, so variables with very
                       different units contribute comparably
    ``absolute``       plain difference
    ``sigma``          divide by the measurement error (chi-square residuals)
    ``max_normalized`` divide by that variable's peak observed value
    """
    diff = simulated - measurement.values
    mode = objective.scaling
    if mode == "absolute":
        return diff
    if mode == "relative":
        return diff / (np.abs(measurement.values) + objective.epsilon)
    if mode == "max_normalized":
        return diff / measurement.scale
    if mode == "sigma":
        sigma = measurement.sigma
        if sigma is None:
            sigma = np.full_like(diff, float(objective.default_sigma))
        return diff / sigma
    raise ProblemError(f"unknown residual scaling '{mode}'")


class Problem:
    """Loaded models and data plus the current shared parameter vector."""

    def __init__(
        self,
        config: StudyConfig,
        models: Optional[Mapping[str, SimulationModel]] = None,
        datasets: Optional[Mapping[str, ExperimentData]] = None,
    ) -> None:
        self.config = config
        self.objective_spec = config.fitting.objective
        self.models: Dict[str, SimulationModel] = dict(
            models if models is not None else build_models(config.models, config.fitting.simulation)
        )
        self.data: Dict[str, ExperimentData] = dict(
            datasets if datasets is not None else load_datasets(config.datasets)
        )
        self.parameter_specs: Dict[str, ParameterSpec] = {
            p.name: p for p in config.all_parameters()
        }
        self.values: Dict[str, float] = config.initial_parameter_values()
        self.n_evaluations = 0
        self.check()

    # -- consistency -------------------------------------------------------
    def check(self) -> None:
        """Verify that every declared name exists somewhere it can be used."""
        problems: List[str] = []

        for name in self.parameter_specs:
            owners = [mid for mid, m in self.models.items() if m.has_parameter(name)]
            if not owners:
                problems.append(
                    f"fitted parameter '{name}' is not a settable parameter in any model"
                )

        for module in self.config.modules:
            for variable in module.variables:
                models_with = {
                    d.model for d in self.datasets_for(module) if d.has(variable)
                }
                if not models_with:
                    problems.append(
                        f"module '{module.id}': variable '{variable}' is not measured in any "
                        f"of its datasets"
                    )
                    continue
                for model_id in models_with:
                    model = self.models.get(model_id)
                    if model is not None and not model.has_variable(variable):
                        problems.append(
                            f"module '{module.id}': variable '{variable}' is measured in a "
                            f"dataset for model '{model_id}', which has no such variable "
                            f"(available: {model.available_variables})"
                        )

        for data in self.data.values():
            model = self.models.get(data.model)
            if model is None:
                problems.append(f"dataset '{data.id}': unknown model '{data.model}'")
                continue
            unknown = model.missing_parameters(
                list(data.conditions) + list(self.config.dataset(data.id).initial_conditions)
            )
            unknown = [
                n for n in unknown if not model.has_variable(n)
            ]
            if unknown:
                problems.append(
                    f"dataset '{data.id}': condition(s) {unknown} are not settable in "
                    f"model '{data.model}'"
                )

        if problems:
            raise ProblemError(
                "configuration is inconsistent with the models or data:\n  - "
                + "\n  - ".join(problems)
            )

    def datasets_for(self, module: ModuleSpec) -> List[ExperimentData]:
        """Datasets a module scores against: its own list, or all of them."""
        specs = self.config.datasets_for_module(module)
        return [self.data[s.id] for s in specs if s.id in self.data]

    # -- parameter vector --------------------------------------------------
    def set_values(self, values: Mapping[str, float]) -> None:
        unknown = sorted(set(values) - set(self.parameter_specs))
        if unknown:
            raise ProblemError(f"cannot set unknown parameter(s) {unknown}")
        self.values.update({k: float(v) for k, v in values.items()})

    def snapshot(self) -> Dict[str, float]:
        return dict(self.values)

    # -- simulation and residuals -----------------------------------------
    def parameters_for(
        self, model: SimulationModel, values: Optional[Mapping[str, float]] = None
    ) -> Dict[str, float]:
        """The subset of the shared parameter vector this model actually has.

        Models in one study need not be structurally identical: a parameter
        such as a feedback constant may exist in only some of them, and is then
        identified from only those datasets.
        """
        source = self.values if values is None else values
        return {k: float(v) for k, v in source.items() if model.has_parameter(k)}

    def simulate_dataset(
        self,
        data: ExperimentData,
        variables: Sequence[str],
        values: Optional[Mapping[str, float]] = None,
    ) -> Dict[str, np.ndarray]:
        model = self.models[data.model]
        grid = data.time_grid(variables)
        return model.simulate(
            times=grid,
            parameters=self.parameters_for(model, values),
            conditions=data.conditions,
            initial_conditions=data.initial_conditions,
        )

    def residual_blocks(
        self,
        module: ModuleSpec,
        values: Optional[Mapping[str, float]] = None,
    ) -> List[ResidualBlock]:
        """Residuals for one module's variables across all its datasets."""
        blocks: List[ResidualBlock] = []
        for data in self.datasets_for(module):
            present = [v for v in module.variables if data.has(v)]
            if not present:
                continue
            simulated = self.simulate_dataset(data, present, values)
            sim_times = simulated["time"]
            for variable in present:
                measurement = data.measurements[variable]
                sim = _interpolate(sim_times, simulated[variable], measurement.times)
                residuals = _scale_residuals(sim, measurement, self.objective_spec)
                weight = float(data.weight) * module.weight_for(variable)
                blocks.append(
                    ResidualBlock(
                        dataset=data.id,
                        variable=variable,
                        times=measurement.times,
                        observed=measurement.values,
                        simulated=sim,
                        residuals=residuals * np.sqrt(weight),
                        weight=weight,
                    )
                )
        return blocks

    def module_cost(
        self, module: ModuleSpec, values: Optional[Mapping[str, float]] = None
    ) -> float:
        """Scalar cost of one module's objective at the given parameter values."""
        try:
            blocks = self.residual_blocks(module, values)
        except SimulationFailure:
            return INFEASIBLE
        if not blocks:
            return 0.0
        stacked = np.concatenate([b.residuals for b in blocks])
        return self._aggregate(stacked)

    def total_cost(self, values: Optional[Mapping[str, float]] = None) -> float:
        """Cost over every module -- the quantity the outer loop should reduce."""
        parts = [self.module_cost(m, values) for m in self.config.modules]
        if any(p >= INFEASIBLE for p in parts):
            return INFEASIBLE
        return float(np.mean(parts)) if self.objective_spec.aggregation == "mean" else float(sum(parts))

    def _aggregate(self, residuals: np.ndarray) -> float:
        squared = np.square(residuals)
        if not np.all(np.isfinite(squared)):
            return INFEASIBLE
        value = float(np.mean(squared)) if self.objective_spec.aggregation == "mean" else float(
            np.sum(squared)
        )
        return min(value, INFEASIBLE)

    def objective_for(self, module: ModuleSpec) -> "ModuleObjective":
        return ModuleObjective(self, module)

    # -- reporting ---------------------------------------------------------
    def cost_report(self, values: Optional[Mapping[str, float]] = None) -> Dict[str, float]:
        report = {m.id: self.module_cost(m, values) for m in self.config.modules}
        report["__total__"] = self.total_cost(values)
        return report

    def predictions(
        self, data: ExperimentData, values: Optional[Mapping[str, float]] = None,
        n_points: int = 200,
    ) -> Dict[str, np.ndarray]:
        """A dense simulated trace for plotting or export."""
        model = self.models[data.model]
        grid = data.time_grid()
        dense = np.linspace(float(grid[0]), float(grid[-1]), n_points) if grid.size > 1 else grid
        return model.simulate(
            times=dense,
            parameters=self.parameters_for(model, values),
            conditions=data.conditions,
            initial_conditions=data.initial_conditions,
        )


class ModuleObjective:
    """A module's cost as a function of its free parameters, in search space.

    Optimizers see a plain ``f(x) -> float`` over ``bounds``; log-scaled
    parameters are exponentiated on the way back into the model. ``residuals``
    exposes the same evaluation as a vector for least-squares backends.
    """

    def __init__(self, problem: Problem, module: ModuleSpec) -> None:
        self.problem = problem
        self.module = module
        self.free: List[ParameterSpec] = module.free_parameters
        if not self.free:
            raise ProblemError(f"module '{module.id}': every parameter is marked fixed")
        self.names: List[str] = [p.name for p in self.free]
        self.bounds: List[Tuple[float, float]] = [p.search_bounds for p in self.free]
        self.n_calls = 0
        self.best_cost = np.inf
        self.best_x: Optional[np.ndarray] = None

    # -- space transforms --------------------------------------------------
    def to_search(self, values: Mapping[str, float]) -> np.ndarray:
        return np.array([p.to_search(float(values[p.name])) for p in self.free], dtype=float)

    def to_model(self, x: Sequence[float]) -> Dict[str, float]:
        return {p.name: p.to_model(float(v)) for p, v in zip(self.free, x)}

    @property
    def x0(self) -> np.ndarray:
        return self.to_search(self.problem.values)

    @property
    def lower(self) -> np.ndarray:
        return np.array([b[0] for b in self.bounds], dtype=float)

    @property
    def upper(self) -> np.ndarray:
        return np.array([b[1] for b in self.bounds], dtype=float)

    def clip(self, x: Sequence[float]) -> np.ndarray:
        return np.clip(np.asarray(x, dtype=float), self.lower, self.upper)

    def _trial_values(self, x: Sequence[float]) -> Dict[str, float]:
        """Full parameter set: the module's trial values over the current vector."""
        values = self.problem.snapshot()
        values.update(self.to_model(x))
        return values

    # -- evaluation --------------------------------------------------------
    def __call__(self, x: Sequence[float]) -> float:
        self.n_calls += 1
        self.problem.n_evaluations += 1
        cost = self.problem.module_cost(self.module, self._trial_values(x))
        if cost < self.best_cost:
            self.best_cost = cost
            self.best_x = np.asarray(x, dtype=float).copy()
        return cost

    def residuals(self, x: Sequence[float]) -> np.ndarray:
        """Residual vector for least-squares backends.

        The length is fixed for a given module, so a failed simulation is
        reported as a large but finite residual rather than an exception.
        """
        self.n_calls += 1
        self.problem.n_evaluations += 1
        values = self._trial_values(x)
        try:
            blocks = self.problem.residual_blocks(self.module, values)
        except SimulationFailure:
            return np.full(self.residual_size, np.sqrt(INFEASIBLE / max(self.residual_size, 1)))
        if not blocks:
            return np.zeros(1)
        stacked = np.concatenate([b.residuals for b in blocks])
        if not np.all(np.isfinite(stacked)):
            stacked = np.nan_to_num(stacked, nan=1e6, posinf=1e6, neginf=-1e6)
        cost = self.problem._aggregate(stacked)
        if cost < self.best_cost:
            self.best_cost = cost
            self.best_x = np.asarray(x, dtype=float).copy()
        if self.problem.objective_spec.aggregation == "mean":
            stacked = stacked / np.sqrt(stacked.size)
        return stacked

    @property
    def residual_size(self) -> int:
        if not hasattr(self, "_residual_size"):
            blocks = self.problem.residual_blocks(self.module, self.problem.values)
            self._residual_size = sum(b.residuals.size for b in blocks) or 1
        return self._residual_size

    def commit(self, x: Sequence[float]) -> Dict[str, float]:
        """Write a solution back into the shared parameter vector."""
        update = self.to_model(x)
        self.problem.set_values(update)
        return update

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ModuleObjective(module={self.module.id!r}, free={self.names})"
