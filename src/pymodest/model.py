"""Antimony -> SBML -> roadrunner simulation, with a small caching layer.

One :class:`SimulationModel` wraps a single Antimony model. It knows how to

* set the shared fitted parameters,
* apply per-dataset conditions and initial conditions,
* integrate to a requested set of time points, and
* return the model's own species plus any derived observables.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from .config import ModelSpec, SimulationSpec

__all__ = ["ModelError", "SimulationFailure", "antimony_to_sbml", "SimulationModel"]


class ModelError(RuntimeError):
    """Raised when a model cannot be translated or loaded."""


class SimulationFailure(RuntimeError):
    """Raised when an integration fails for a given parameter set.

    The estimator catches this and scores the parameter set as infeasible
    rather than aborting the whole fit.
    """


def antimony_to_sbml(text: str, label: str = "model") -> str:
    """Translate Antimony source to SBML, raising a readable error on failure."""
    import antimony

    antimony.clearPreviousLoads()
    code = antimony.loadAntimonyString(text)
    if code < 0:
        message = antimony.getLastError()
        raise ModelError(f"{label}: Antimony could not parse the model:\n{message}")
    main = antimony.getMainModuleName()
    sbml = antimony.getSBMLString(main)
    if not sbml:
        raise ModelError(f"{label}: Antimony produced no SBML ({antimony.getLastError()})")
    return sbml


def _ids(rr, getter: str) -> List[str]:
    """Fetch an id list from roadrunner, tolerating API differences across versions."""
    for holder in (getattr(rr, "model", None), rr):
        if holder is None:
            continue
        fn = getattr(holder, getter, None)
        if callable(fn):
            try:
                return [str(x) for x in fn()]
            except Exception:  # pragma: no cover - defensive
                continue
    return []


class SimulationModel:
    """A loaded, reusable roadrunner instance for one Antimony model."""

    def __init__(
        self,
        spec: ModelSpec,
        simulation: Optional[SimulationSpec] = None,
        cache_size: int = 256,
    ) -> None:
        import roadrunner

        self.spec = spec
        self.simulation = simulation or SimulationSpec()
        self.sbml = antimony_to_sbml(spec.antimony, f"model '{spec.id}'")
        try:
            self.rr = roadrunner.RoadRunner(self.sbml)
        except Exception as exc:  # pragma: no cover - depends on roadrunner internals
            raise ModelError(f"model '{spec.id}': roadrunner could not load the SBML ({exc})") from exc

        self._configure_integrator()

        self.floating_species: List[str] = _ids(self.rr, "getFloatingSpeciesIds")
        self.boundary_species: List[str] = _ids(self.rr, "getBoundarySpeciesIds")
        self.global_parameters: List[str] = _ids(self.rr, "getGlobalParameterIds")
        self.compartments: List[str] = _ids(self.rr, "getCompartmentIds")
        self.observables: Dict[str, str] = dict(spec.observables)

        self._settable = set(self.global_parameters) | set(self.compartments) | set(
            self.boundary_species
        )
        self._species = set(self.floating_species) | set(self.boundary_species)
        self._target_cache: Dict[str, str] = {}
        self._cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._cache_order: List[str] = []
        self._cache_size = int(cache_size)
        self.n_simulations = 0

    # -- introspection -----------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"SimulationModel(id={self.spec.id!r}, species={len(self.floating_species)}, "
            f"parameters={len(self.global_parameters)})"
        )

    @property
    def available_variables(self) -> List[str]:
        """Everything a dataset or module may legitimately refer to."""
        return sorted(self._species | set(self.global_parameters) | set(self.observables))

    def has_variable(self, name: str) -> bool:
        return name in self._species or name in self.observables or name in self.global_parameters

    def has_parameter(self, name: str) -> bool:
        return name in self._settable

    def missing_parameters(self, names: Iterable[str]) -> List[str]:
        return [n for n in names if not self.has_parameter(n)]

    def missing_variables(self, names: Iterable[str]) -> List[str]:
        return [n for n in names if not self.has_variable(n)]

    # -- configuration -----------------------------------------------------
    def _configure_integrator(self) -> None:
        sim = self.simulation
        try:
            self.rr.setIntegrator(sim.integrator)
            integrator = self.rr.getIntegrator()
            integrator.setValue("relative_tolerance", sim.relative_tolerance)
            integrator.setValue("absolute_tolerance", sim.absolute_tolerance)
            if sim.integrator == "cvode":
                integrator.setValue("stiff", bool(sim.stiff))
                integrator.setValue("maximum_num_steps", int(sim.max_steps))
        except Exception as exc:  # pragma: no cover - integrator option names vary
            raise ModelError(f"model '{self.spec.id}': integrator setup failed ({exc})") from exc

    def _target_for(self, key: str) -> str:
        """The roadrunner selection used to write ``key`` (resolved once, cached)."""
        if key not in self._target_cache:
            target = key
            try:
                self.rr[key]
            except Exception:
                target = f"init({key})" if key in self._species else key
            self._target_cache[key] = target
        return self._target_cache[key]

    def _apply(self, assignments: Mapping[str, float], what: str) -> None:
        """Write parameter and species values onto the model.

        Species are set through their *current* value, not ``init(X)``. The
        resulting trajectories are identical because the model has just been
        reset to its SBML defaults, so the current value is the starting state
        -- but writing an initial value makes roadrunner re-initialise the whole
        model, which costs tens of milliseconds and would dominate a fit that
        varies the starting conditions between datasets.
        """
        for key, value in assignments.items():
            try:
                self.rr[self._target_for(key)] = float(value)
            except Exception as exc:
                raise ModelError(
                    f"model '{self.spec.id}': cannot set {what} '{key}' ({exc})"
                ) from exc

    # -- simulation --------------------------------------------------------
    def _cache_key(
        self,
        parameters: Mapping[str, float],
        conditions: Mapping[str, float],
        initial_conditions: Mapping[str, float],
        times: Sequence[float],
    ) -> str:
        digest = hashlib.blake2b(digest_size=16)
        for label, mapping in (
            ("p", parameters), ("c", conditions), ("i", initial_conditions),
        ):
            digest.update(label.encode())
            for key in sorted(mapping):
                digest.update(f"{key}={float(mapping[key]):.17g};".encode())
        digest.update(b"t")
        digest.update(np.asarray(times, dtype=float).tobytes())
        return digest.hexdigest()

    def _remember(self, key: str, value: Dict[str, np.ndarray]) -> None:
        self._cache[key] = value
        self._cache_order.append(key)
        while len(self._cache_order) > self._cache_size:
            self._cache.pop(self._cache_order.pop(0), None)

    def simulate(
        self,
        times: Sequence[float],
        parameters: Optional[Mapping[str, float]] = None,
        conditions: Optional[Mapping[str, float]] = None,
        initial_conditions: Optional[Mapping[str, float]] = None,
        variables: Optional[Sequence[str]] = None,
        use_cache: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Integrate the model and return ``{variable: values}`` at ``times``.

        ``parameters`` are the shared fitted parameters; ``conditions`` are
        per-dataset overrides (applied after them); ``initial_conditions`` set
        species starting amounts. Raises :class:`SimulationFailure` if the
        integrator cannot complete.
        """
        parameters = dict(parameters or {})
        conditions = {**self.spec.overrides, **(conditions or {})}
        initial_conditions = dict(initial_conditions or {})

        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or times.size == 0:
            raise ValueError("times must be a non-empty 1-D sequence")
        if np.any(np.diff(times) < 0):
            raise ValueError("times must be non-decreasing")

        key = self._cache_key(parameters, conditions, initial_conditions, times)
        if use_cache and key in self._cache:
            result = self._cache[key]
        else:
            result = self._run(times, parameters, conditions, initial_conditions)
            if use_cache:
                self._remember(key, result)

        if variables is None:
            return dict(result)
        missing = [v for v in variables if v not in result]
        if missing:
            raise KeyError(
                f"model '{self.spec.id}': no such variable(s) {missing}; "
                f"available: {self.available_variables}"
            )
        return {v: result[v] for v in variables}

    def _run(
        self,
        times: np.ndarray,
        parameters: Mapping[str, float],
        conditions: Mapping[str, float],
        initial_conditions: Mapping[str, float],
    ) -> Dict[str, np.ndarray]:
        for reset in ("resetAll", "resetToOrigin", "reset"):
            fn = getattr(self.rr, reset, None)
            if callable(fn):
                try:
                    fn()
                    break
                except Exception:  # pragma: no cover - defensive
                    continue
        # Later mappings win. resetAll() first returns every parameter and
        # species to its SBML default, so values left over from a previous
        # dataset's conditions cannot leak into this one.
        merged: Dict[str, float] = {}
        for mapping in (parameters, conditions, initial_conditions):
            for key, value in mapping.items():
                merged[key] = float(value)
        self._apply(merged, "assignment")

        selections = ["time"] + self.floating_species + self.boundary_species
        # de-duplicate while preserving order
        seen: set = set()
        selections = [s for s in selections if not (s in seen or seen.add(s))]

        start = float(times[0])
        needs_prefix = start > 0.0
        grid = np.concatenate(([0.0], times)) if needs_prefix else times

        self.n_simulations += 1
        try:
            raw = self.rr.simulate(times=grid, selections=selections)
        except Exception as exc:
            raise SimulationFailure(
                f"model '{self.spec.id}': integration failed ({exc})"
            ) from exc

        array = np.asarray(raw, dtype=float)
        if needs_prefix:
            array = array[1:]
        if array.shape[0] != times.size:
            raise SimulationFailure(
                f"model '{self.spec.id}': integrator returned {array.shape[0]} rows "
                f"for {times.size} requested time points"
            )
        if not np.all(np.isfinite(array)):
            raise SimulationFailure(f"model '{self.spec.id}': simulation produced non-finite values")

        out: Dict[str, np.ndarray] = {
            name: array[:, i] for i, name in enumerate(selections)
        }
        # constant parameters are available as flat traces
        for name in self.global_parameters:
            if name not in out:
                out[name] = np.full(times.size, float(self.rr[name]))
        self._add_observables(out, times.size)
        return out

    def _add_observables(self, values: Dict[str, np.ndarray], n: int) -> None:
        if not self.observables:
            return
        env = {"__builtins__": {}}
        for fn in ("exp", "log", "log10", "sqrt", "abs", "maximum", "minimum", "where"):
            env[fn] = getattr(np, fn)
        env.update(values)
        for name, expression in self.observables.items():
            try:
                result = eval(expression, env)  # noqa: S307 - expressions come from the user's own config
            except Exception as exc:
                raise ModelError(
                    f"model '{self.spec.id}': observable '{name}' = '{expression}' "
                    f"could not be evaluated ({exc})"
                ) from exc
            arr = np.asarray(result, dtype=float)
            if arr.ndim == 0:
                arr = np.full(n, float(arr))
            if arr.shape != (n,):
                raise ModelError(
                    f"model '{self.spec.id}': observable '{name}' produced shape {arr.shape}, "
                    f"expected ({n},)"
                )
            values[name] = arr
            env[name] = arr

    def clear_cache(self) -> None:
        self._cache.clear()
        self._cache_order.clear()


def build_models(
    specs: Sequence[ModelSpec], simulation: Optional[SimulationSpec] = None
) -> Dict[str, SimulationModel]:
    """Load every model in a study, keyed by model id."""
    return {spec.id: SimulationModel(spec, simulation) for spec in specs}
