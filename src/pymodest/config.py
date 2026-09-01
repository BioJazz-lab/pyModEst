"""TOML-backed configuration schema for pyModEst.

A pyModEst study is described by one TOML file with four sections:

``[[models]]``    one or more Antimony models that share a common parameter set
``[[datasets]]``  experimental measurements, each attached to one model
``[[modules]]``   the divide-and-conquer partition: which parameters are fitted
                  together, and which measured variables score that fit
``[fitting]``     loop control, objective definition, optimizer defaults

See ``examples/`` for a fully worked configuration.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:  # Python >= 3.11
    import tomllib as _toml
except ModuleNotFoundError:  # pragma: no cover - fallback for 3.9/3.10
    import tomli as _toml  # type: ignore[no-redef]

__all__ = [
    "ConfigError",
    "ParameterSpec",
    "ModuleSpec",
    "ModelSpec",
    "DatasetSpec",
    "ObjectiveSpec",
    "SimulationSpec",
    "OptimizerSpec",
    "FittingSpec",
    "StudyConfig",
    "load_config",
]

SCALINGS = ("relative", "absolute", "sigma", "max_normalized")
AGGREGATIONS = ("mean", "sum")
PARAM_SCALES = ("linear", "log")
MODULE_ORDERS = ("as_listed", "random", "round_robin_reversed")
ACCEPT_POLICIES = ("module", "total")


class ConfigError(ValueError):
    """Raised when a configuration file is malformed or inconsistent."""


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def _require(table: Dict[str, Any], key: str, where: str) -> Any:
    if key not in table:
        raise ConfigError(f"{where}: missing required key '{key}'")
    return table[key]


def _as_float(value: Any, where: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{where}: expected a number, got {value!r}") from None
    if math.isnan(out):
        raise ConfigError(f"{where}: value must not be NaN")
    return out


def _as_str_list(value: Any, where: str) -> List[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"{where}: expected a list of strings, got {value!r}")
    out = []
    for item in value:
        if not isinstance(item, str):
            raise ConfigError(f"{where}: expected a list of strings, got {item!r}")
        out.append(item)
    return out


def _check_unknown(table: Dict[str, Any], allowed: Sequence[str], where: str) -> None:
    unknown = sorted(set(table) - set(allowed))
    if unknown:
        raise ConfigError(
            f"{where}: unknown key(s) {unknown}; allowed keys are {sorted(allowed)}"
        )


def _resolve(base: Path, value: str) -> Path:
    path = Path(os.path.expanduser(value))
    return path if path.is_absolute() else (base / path).resolve()


# --------------------------------------------------------------------------
# parameters and modules
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ParameterSpec:
    """One fitted parameter: its search range and how it is sampled."""

    name: str
    lower: float
    upper: float
    init: Optional[float] = None
    scale: str = "linear"
    fixed: bool = False

    def __post_init__(self) -> None:
        where = f"parameter '{self.name}'"
        if self.scale not in PARAM_SCALES:
            raise ConfigError(f"{where}: scale must be one of {list(PARAM_SCALES)}")
        if not self.upper > self.lower:
            raise ConfigError(f"{where}: upper ({self.upper}) must exceed lower ({self.lower})")
        if self.scale == "log" and self.lower <= 0:
            raise ConfigError(f"{where}: log scale requires a strictly positive lower bound")
        if self.init is not None and not (self.lower <= self.init <= self.upper):
            raise ConfigError(
                f"{where}: init ({self.init}) lies outside [{self.lower}, {self.upper}]"
            )

    @property
    def initial_value(self) -> float:
        """Starting value: the declared ``init``, else the mid-point of the range."""
        if self.init is not None:
            return self.init
        if self.scale == "log":
            return math.sqrt(self.lower * self.upper)
        return 0.5 * (self.lower + self.upper)

    # -- transforms between model space and optimizer (search) space --------
    def to_search(self, value: float) -> float:
        return math.log10(value) if self.scale == "log" else value

    def to_model(self, value: float) -> float:
        return 10.0 ** value if self.scale == "log" else value

    @property
    def search_bounds(self) -> tuple:
        if self.scale == "log":
            return (math.log10(self.lower), math.log10(self.upper))
        return (self.lower, self.upper)

    @classmethod
    def from_toml(cls, table: Dict[str, Any], where: str) -> "ParameterSpec":
        _check_unknown(table, ("name", "lower", "upper", "init", "scale", "fixed"), where)
        name = _require(table, "name", where)
        if not isinstance(name, str):
            raise ConfigError(f"{where}: 'name' must be a string")
        where = f"{where} parameter '{name}'"
        init = table.get("init")
        return cls(
            name=name,
            lower=_as_float(_require(table, "lower", where), f"{where}.lower"),
            upper=_as_float(_require(table, "upper", where), f"{where}.upper"),
            init=None if init is None else _as_float(init, f"{where}.init"),
            scale=str(table.get("scale", "linear")),
            fixed=bool(table.get("fixed", False)),
        )


@dataclass(frozen=True)
class OptimizerSpec:
    """Which optimizer runs a module fit, and with what options."""

    name: str = "differential_evolution"
    options: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_toml(cls, table: Dict[str, Any], where: str) -> "OptimizerSpec":
        if not isinstance(table, dict):
            raise ConfigError(f"{where}: expected a table")
        opts = dict(table)
        name = opts.pop("name", "differential_evolution")
        if not isinstance(name, str):
            raise ConfigError(f"{where}.name: must be a string")
        return cls(name=name, options=opts)

    def merged_with(self, default: "OptimizerSpec") -> "OptimizerSpec":
        """Module-level settings override the study default, key by key."""
        if self.name == default.name:
            options = {**default.options, **self.options}
        else:
            options = dict(self.options)
        return OptimizerSpec(name=self.name, options=options)


@dataclass(frozen=True)
class ModuleSpec:
    """One block of the divide-and-conquer partition.

    ``parameters`` are fitted together while every other module's parameters are
    held fixed; ``variables`` are the measured species that score the fit.
    """

    id: str
    parameters: List[ParameterSpec]
    variables: List[str]
    weights: Dict[str, float] = field(default_factory=dict)
    datasets: Optional[List[str]] = None
    optimizer: Optional[OptimizerSpec] = None
    description: str = ""

    def __post_init__(self) -> None:
        where = f"module '{self.id}'"
        if not self.parameters:
            raise ConfigError(f"{where}: must declare at least one parameter")
        if not self.variables:
            raise ConfigError(f"{where}: must declare at least one variable")
        names = [p.name for p in self.parameters]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ConfigError(f"{where}: duplicate parameter(s) {dupes}")
        unknown_w = sorted(set(self.weights) - set(self.variables))
        if unknown_w:
            raise ConfigError(f"{where}: weights given for non-module variable(s) {unknown_w}")

    @property
    def free_parameters(self) -> List[ParameterSpec]:
        return [p for p in self.parameters if not p.fixed]

    def weight_for(self, variable: str) -> float:
        return float(self.weights.get(variable, 1.0))

    @classmethod
    def from_toml(cls, table: Dict[str, Any], where: str) -> "ModuleSpec":
        _check_unknown(
            table,
            ("id", "parameters", "variables", "weights", "datasets", "optimizer", "description"),
            where,
        )
        mid = _require(table, "id", where)
        if not isinstance(mid, str):
            raise ConfigError(f"{where}: 'id' must be a string")
        where = f"module '{mid}'"

        raw_params = _require(table, "parameters", where)
        if not isinstance(raw_params, list):
            raise ConfigError(f"{where}: 'parameters' must be an array of tables")
        parameters = [ParameterSpec.from_toml(p, where) for p in raw_params]

        weights_tbl = table.get("weights", {}) or {}
        if not isinstance(weights_tbl, dict):
            raise ConfigError(f"{where}.weights: expected a table")
        weights = {k: _as_float(v, f"{where}.weights.{k}") for k, v in weights_tbl.items()}

        opt = table.get("optimizer")
        datasets = table.get("datasets")
        return cls(
            id=mid,
            parameters=parameters,
            variables=_as_str_list(_require(table, "variables", where), f"{where}.variables"),
            weights=weights,
            datasets=None if datasets is None else _as_str_list(datasets, f"{where}.datasets"),
            optimizer=None if opt is None else OptimizerSpec.from_toml(opt, f"{where}.optimizer"),
            description=str(table.get("description", "")),
        )


# --------------------------------------------------------------------------
# models and datasets
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    """An Antimony model. Several models may share one fitted parameter set."""

    id: str
    antimony: str
    source: Optional[Path] = None
    overrides: Dict[str, float] = field(default_factory=dict)
    observables: Dict[str, str] = field(default_factory=dict)
    description: str = ""

    @classmethod
    def from_toml(cls, table: Dict[str, Any], base: Path, where: str) -> "ModelSpec":
        _check_unknown(
            table,
            ("id", "antimony_file", "antimony", "overrides", "observables", "description"),
            where,
        )
        mid = _require(table, "id", where)
        if not isinstance(mid, str):
            raise ConfigError(f"{where}: 'id' must be a string")
        where = f"model '{mid}'"

        has_file = "antimony_file" in table
        has_inline = "antimony" in table
        if has_file == has_inline:
            raise ConfigError(f"{where}: give exactly one of 'antimony_file' or 'antimony'")

        source: Optional[Path] = None
        if has_file:
            source = _resolve(base, str(table["antimony_file"]))
            if not source.is_file():
                raise ConfigError(f"{where}: antimony_file not found: {source}")
            text = source.read_text()
        else:
            text = str(table["antimony"])

        overrides_tbl = table.get("overrides", {}) or {}
        if not isinstance(overrides_tbl, dict):
            raise ConfigError(f"{where}.overrides: expected a table")
        overrides = {k: _as_float(v, f"{where}.overrides.{k}") for k, v in overrides_tbl.items()}

        obs_tbl = table.get("observables", {}) or {}
        if not isinstance(obs_tbl, dict):
            raise ConfigError(f"{where}.observables: expected a table")
        observables = {k: str(v) for k, v in obs_tbl.items()}

        return cls(
            id=mid,
            antimony=text,
            source=source,
            overrides=overrides,
            observables=observables,
            description=str(table.get("description", "")),
        )


@dataclass(frozen=True)
class DatasetSpec:
    """One experiment: measurements of shared variables under set conditions."""

    id: str
    model: str
    file: Optional[Path] = None
    format: str = "wide"
    weight: float = 1.0
    conditions: Dict[str, float] = field(default_factory=dict)
    initial_conditions: Dict[str, float] = field(default_factory=dict)
    time_column: str = "time"
    inline: Optional[Dict[str, List[float]]] = None
    description: str = ""

    def __post_init__(self) -> None:
        where = f"dataset '{self.id}'"
        if self.format not in ("wide", "long"):
            raise ConfigError(f"{where}: format must be 'wide' or 'long'")
        if (self.file is None) == (self.inline is None):
            raise ConfigError(f"{where}: give exactly one of 'file' or inline 'data'")
        if self.weight < 0:
            raise ConfigError(f"{where}: weight must be non-negative")

    @classmethod
    def from_toml(cls, table: Dict[str, Any], base: Path, where: str) -> "DatasetSpec":
        _check_unknown(
            table,
            (
                "id", "model", "file", "format", "weight", "conditions",
                "initial_conditions", "time_column", "data", "description",
            ),
            where,
        )
        did = _require(table, "id", where)
        if not isinstance(did, str):
            raise ConfigError(f"{where}: 'id' must be a string")
        where = f"dataset '{did}'"

        path: Optional[Path] = None
        if "file" in table:
            path = _resolve(base, str(table["file"]))
            if not path.is_file():
                raise ConfigError(f"{where}: data file not found: {path}")

        inline = None
        if "data" in table:
            raw = table["data"]
            if not isinstance(raw, dict):
                raise ConfigError(f"{where}.data: expected a table of columns")
            inline = {
                k: [_as_float(x, f"{where}.data.{k}") for x in v]
                for k, v in raw.items()
            }

        cond = table.get("conditions", {}) or {}
        init = table.get("initial_conditions", {}) or {}
        for name, tbl in (("conditions", cond), ("initial_conditions", init)):
            if not isinstance(tbl, dict):
                raise ConfigError(f"{where}.{name}: expected a table")

        return cls(
            id=did,
            model=str(_require(table, "model", where)),
            file=path,
            format=str(table.get("format", "wide")),
            weight=_as_float(table.get("weight", 1.0), f"{where}.weight"),
            conditions={k: _as_float(v, f"{where}.conditions.{k}") for k, v in cond.items()},
            initial_conditions={
                k: _as_float(v, f"{where}.initial_conditions.{k}") for k, v in init.items()
            },
            time_column=str(table.get("time_column", "time")),
            inline=inline,
            description=str(table.get("description", "")),
        )


# --------------------------------------------------------------------------
# fitting settings
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ObjectiveSpec:
    """How residuals are scaled and combined into a module cost."""

    scaling: str = "relative"
    aggregation: str = "mean"
    epsilon: float = 1e-8
    default_sigma: float = 1.0

    def __post_init__(self) -> None:
        if self.scaling not in SCALINGS:
            raise ConfigError(f"[fitting.objective]: scaling must be one of {list(SCALINGS)}")
        if self.aggregation not in AGGREGATIONS:
            raise ConfigError(
                f"[fitting.objective]: aggregation must be one of {list(AGGREGATIONS)}"
            )
        if self.epsilon <= 0:
            raise ConfigError("[fitting.objective]: epsilon must be positive")

    @classmethod
    def from_toml(cls, table: Dict[str, Any]) -> "ObjectiveSpec":
        where = "[fitting.objective]"
        _check_unknown(table, ("scaling", "aggregation", "epsilon", "default_sigma"), where)
        return cls(
            scaling=str(table.get("scaling", "relative")),
            aggregation=str(table.get("aggregation", "mean")),
            epsilon=_as_float(table.get("epsilon", 1e-8), f"{where}.epsilon"),
            default_sigma=_as_float(table.get("default_sigma", 1.0), f"{where}.default_sigma"),
        )


@dataclass(frozen=True)
class SimulationSpec:
    """Integrator settings handed to roadrunner."""

    integrator: str = "cvode"
    relative_tolerance: float = 1e-8
    absolute_tolerance: float = 1e-10
    stiff: bool = True
    max_steps: int = 20000

    @classmethod
    def from_toml(cls, table: Dict[str, Any]) -> "SimulationSpec":
        where = "[fitting.simulation]"
        _check_unknown(
            table,
            ("integrator", "relative_tolerance", "absolute_tolerance", "stiff", "max_steps"),
            where,
        )
        return cls(
            integrator=str(table.get("integrator", "cvode")),
            relative_tolerance=_as_float(
                table.get("relative_tolerance", 1e-8), f"{where}.relative_tolerance"
            ),
            absolute_tolerance=_as_float(
                table.get("absolute_tolerance", 1e-10), f"{where}.absolute_tolerance"
            ),
            stiff=bool(table.get("stiff", True)),
            max_steps=int(table.get("max_steps", 20000)),
        )


@dataclass(frozen=True)
class FittingSpec:
    """Control of the outer divide-and-conquer loop.

    ``accept`` decides when a module fit is kept:

    ``module``  keep it whenever the module's own cost improved. This is the
                plain divide-and-conquer rule; because modules are coupled
                through the shared model, the total cost can still rise.
    ``total``   additionally require that the total cost across all modules did
                not rise. Slower by one extra evaluation per step, but the
                total then decreases monotonically.

    A loop counts as making no progress when the total cost improved by less
    than ``tol`` relatively *and* less than ``atol`` absolutely. The absolute
    floor matters: a cost converging geometrically towards zero keeps producing
    large relative gains forever, so a relative test alone never terminates.

    ``refine`` names an optimizer used from loop ``refine_after`` + 1 onward.
    A global search in the first loop followed by a cheap local refinement
    afterwards is usually both faster and steadier than repeating a stochastic
    global search every loop.
    """

    max_loops: int = 10
    module_order: Any = "as_listed"
    tol: float = 1e-6
    atol: float = 1e-12
    patience: int = 2
    seed: Optional[int] = None
    accept: str = "module"
    optimizer: OptimizerSpec = field(default_factory=OptimizerSpec)
    refine: Optional[OptimizerSpec] = None
    refine_after: int = 1
    objective: ObjectiveSpec = field(default_factory=ObjectiveSpec)
    simulation: SimulationSpec = field(default_factory=SimulationSpec)

    def __post_init__(self) -> None:
        if self.max_loops < 1:
            raise ConfigError("[fitting]: max_loops must be at least 1")
        if self.patience < 1:
            raise ConfigError("[fitting]: patience must be at least 1")
        if self.accept not in ACCEPT_POLICIES:
            raise ConfigError(f"[fitting]: accept must be one of {list(ACCEPT_POLICIES)}")
        if self.refine_after < 0:
            raise ConfigError("[fitting]: refine_after must be non-negative")
        if isinstance(self.module_order, str) and self.module_order not in MODULE_ORDERS:
            raise ConfigError(
                f"[fitting]: module_order must be an explicit list or one of {list(MODULE_ORDERS)}"
            )

    @classmethod
    def from_toml(cls, table: Dict[str, Any]) -> "FittingSpec":
        where = "[fitting]"
        _check_unknown(
            table,
            (
                "max_loops", "module_order", "tol", "atol", "patience", "seed", "accept",
                "optimizer", "refine", "refine_after", "objective", "simulation",
            ),
            where,
        )
        order = table.get("module_order", "as_listed")
        if isinstance(order, list):
            order = _as_str_list(order, f"{where}.module_order")
        seed = table.get("seed")
        refine = table.get("refine")
        return cls(
            max_loops=int(table.get("max_loops", 10)),
            module_order=order,
            tol=_as_float(table.get("tol", 1e-6), f"{where}.tol"),
            atol=_as_float(table.get("atol", 1e-12), f"{where}.atol"),
            patience=int(table.get("patience", 2)),
            seed=None if seed is None else int(seed),
            accept=str(table.get("accept", "module")),
            optimizer=OptimizerSpec.from_toml(table.get("optimizer", {}) or {}, f"{where}.optimizer"),
            refine=None if refine is None else OptimizerSpec.from_toml(refine, f"{where}.refine"),
            refine_after=int(table.get("refine_after", 1)),
            objective=ObjectiveSpec.from_toml(table.get("objective", {}) or {}),
            simulation=SimulationSpec.from_toml(table.get("simulation", {}) or {}),
        )


# --------------------------------------------------------------------------
# top-level study
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class StudyConfig:
    """A complete parameter-estimation study."""

    models: List[ModelSpec]
    datasets: List[DatasetSpec]
    modules: List[ModuleSpec]
    fitting: FittingSpec = field(default_factory=FittingSpec)
    name: str = "pymodest-study"
    output_dir: Path = Path("results")
    base_dir: Path = Path(".")

    # -- lookups -----------------------------------------------------------
    @property
    def model_ids(self) -> List[str]:
        return [m.id for m in self.models]

    @property
    def dataset_ids(self) -> List[str]:
        return [d.id for d in self.datasets]

    @property
    def module_ids(self) -> List[str]:
        return [m.id for m in self.modules]

    def model(self, mid: str) -> ModelSpec:
        for m in self.models:
            if m.id == mid:
                return m
        raise KeyError(f"no model with id '{mid}'")

    def dataset(self, did: str) -> DatasetSpec:
        for d in self.datasets:
            if d.id == did:
                return d
        raise KeyError(f"no dataset with id '{did}'")

    def module(self, mid: str) -> ModuleSpec:
        for m in self.modules:
            if m.id == mid:
                return m
        raise KeyError(f"no module with id '{mid}'")

    def all_parameters(self) -> List[ParameterSpec]:
        """Every fitted parameter across all modules, in module order."""
        return [p for module in self.modules for p in module.parameters]

    def parameter_module(self, name: str) -> str:
        for module in self.modules:
            if any(p.name == name for p in module.parameters):
                return module.id
        raise KeyError(f"parameter '{name}' does not belong to any module")

    def initial_parameter_values(self) -> Dict[str, float]:
        return {p.name: p.initial_value for p in self.all_parameters()}

    def datasets_for_module(self, module: ModuleSpec) -> List[DatasetSpec]:
        if module.datasets is None:
            return list(self.datasets)
        chosen = set(module.datasets)
        return [d for d in self.datasets if d.id in chosen]

    def resolved_module_order(self) -> List[str]:
        order = self.fitting.module_order
        if isinstance(order, list):
            return list(order)
        return self.module_ids

    def optimizer_for(self, module: ModuleSpec, loop: int = 1) -> OptimizerSpec:
        """Which optimizer runs this module in this loop.

        A module's own ``[modules.optimizer]`` always wins. Otherwise the study
        default applies, replaced by ``[fitting.refine]`` once the loop number
        passes ``refine_after``.
        """
        if module.optimizer is not None:
            return module.optimizer.merged_with(self.fitting.optimizer)
        refine = self.fitting.refine
        if refine is not None and loop > self.fitting.refine_after:
            return refine
        return self.fitting.optimizer

    def with_output_dir(self, path: Path) -> "StudyConfig":
        return replace(self, output_dir=Path(path))

    # -- validation --------------------------------------------------------
    def validate(self) -> None:
        for label, ids in (
            ("model", self.model_ids),
            ("dataset", self.dataset_ids),
            ("module", self.module_ids),
        ):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            if dupes:
                raise ConfigError(f"duplicate {label} id(s): {dupes}")

        if not self.models:
            raise ConfigError("at least one [[models]] entry is required")
        if not self.datasets:
            raise ConfigError("at least one [[datasets]] entry is required")
        if not self.modules:
            raise ConfigError("at least one [[modules]] entry is required")

        known_models = set(self.model_ids)
        for d in self.datasets:
            if d.model not in known_models:
                raise ConfigError(
                    f"dataset '{d.id}' references unknown model '{d.model}'; "
                    f"known models: {sorted(known_models)}"
                )

        known_datasets = set(self.dataset_ids)
        for m in self.modules:
            if m.datasets is not None:
                missing = sorted(set(m.datasets) - known_datasets)
                if missing:
                    raise ConfigError(f"module '{m.id}' references unknown dataset(s) {missing}")

        # a parameter may be fitted in exactly one module: modules must partition
        seen: Dict[str, str] = {}
        for module in self.modules:
            for p in module.parameters:
                if p.name in seen:
                    raise ConfigError(
                        f"parameter '{p.name}' appears in both module '{seen[p.name]}' and "
                        f"module '{module.id}'; modules must partition the parameter set"
                    )
                seen[p.name] = module.id

        order = self.fitting.module_order
        if isinstance(order, list):
            unknown = sorted(set(order) - set(self.module_ids))
            if unknown:
                raise ConfigError(f"[fitting].module_order references unknown module(s) {unknown}")
            missing = sorted(set(self.module_ids) - set(order))
            if missing:
                raise ConfigError(
                    f"[fitting].module_order omits module(s) {missing}; list every module"
                )

    # -- construction ------------------------------------------------------
    @classmethod
    def from_toml(cls, table: Dict[str, Any], base_dir: Path) -> "StudyConfig":
        _check_unknown(
            table, ("study", "models", "datasets", "modules", "fitting"), "top level"
        )
        study = table.get("study", {}) or {}
        if not isinstance(study, dict):
            raise ConfigError("[study]: expected a table")
        _check_unknown(study, ("name", "output_dir"), "[study]")

        for key in ("models", "datasets", "modules"):
            if key in table and not isinstance(table[key], list):
                raise ConfigError(f"[[{key}]]: expected an array of tables")

        cfg = cls(
            models=[ModelSpec.from_toml(t, base_dir, "[[models]]") for t in table.get("models", [])],
            datasets=[
                DatasetSpec.from_toml(t, base_dir, "[[datasets]]") for t in table.get("datasets", [])
            ],
            modules=[ModuleSpec.from_toml(t, "[[modules]]") for t in table.get("modules", [])],
            fitting=FittingSpec.from_toml(table.get("fitting", {}) or {}),
            name=str(study.get("name", "pymodest-study")),
            output_dir=_resolve(base_dir, str(study.get("output_dir", "results"))),
            base_dir=base_dir,
        )
        cfg.validate()
        return cfg


def load_config(path: os.PathLike | str) -> StudyConfig:
    """Load and validate a study configuration from a TOML file.

    Relative paths inside the file are resolved against the file's directory.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ConfigError(f"configuration file not found: {path}")
    with open(path, "rb") as handle:
        try:
            table = _toml.load(handle)
        except Exception as exc:  # tomllib raises TOMLDecodeError
            raise ConfigError(f"{path}: could not parse TOML ({exc})") from exc
    return StudyConfig.from_toml(table, path.parent)
