# Python API

The CLI covers the common path. Use the library directly when you want to build
studies programmatically, drive the loop yourself, or reuse the simulation and
objective machinery for something else.

```python
from pymodest import load_config, fit

config = load_config("study.toml")
result = fit(config)
result.save(config.output_dir)
```

Signatures below are current; `pymodest.__all__` lists the public surface.

---

## Entry points

```python
fit(config: StudyConfig, max_loops: int | None = None,
    callback: Callable[[ModuleStep], None] | None = None) -> FitResult
fit_from_file(path, max_loops: int | None = None) -> FitResult
load_config(path) -> StudyConfig
```

`callback` fires after every module fit, receiving the `ModuleStep`.

---

## Building a study in code

Every part of a TOML config has a dataclass counterpart in `pymodest.config`,
so a study can be built without touching a file — useful for sweeps, and for
generating studies from a template.

```python
from pymodest.config import (
    DatasetSpec, FittingSpec, ModelSpec, ModuleSpec, ObjectiveSpec,
    OptimizerSpec, ParameterSpec, SimulationSpec, StudyConfig,
)

config = StudyConfig(
    name="study",
    models=[ModelSpec(id="wt", antimony=SOURCE)],
    datasets=[DatasetSpec(id="exp", model="wt",
                          inline={"time": [...], "A": [...]},
                          initial_conditions={"S": 10.0})],
    modules=[ModuleSpec(id="upstream", variables=["A"], parameters=[
        ParameterSpec("Vmax1", lower=0.1, upper=20.0, scale="log"),
    ])],
    fitting=FittingSpec(max_loops=6, seed=3),
)
config.validate()          # always call this; it catches wiring mistakes
```

These are frozen dataclasses, so vary a study with `dataclasses.replace`:

```python
from dataclasses import replace

faster = replace(config, fitting=replace(config.fitting, max_loops=3))
```

### `StudyConfig` helpers

| method | returns |
| --- | --- |
| `validate()` | raises `ConfigError` if the study is inconsistent |
| `model(id)` / `dataset(id)` / `module(id)` | one spec by id |
| `model_ids` / `dataset_ids` / `module_ids` | the ids, in order |
| `all_parameters()` | every `ParameterSpec`, in module order |
| `parameter_module(name)` | which module owns a parameter |
| `initial_parameter_values()` | the starting vector |
| `datasets_for_module(module)` | datasets that module scores |
| `optimizer_for(module, loop=1)` | the resolved `OptimizerSpec` |
| `with_output_dir(path)` | a copy writing elsewhere |

---

## `ModularEstimator`

Use it directly when you want the loop's parts rather than a single call.

```python
ModularEstimator(config, problem=None, rng=None)

.run(max_loops=None, callback=None) -> FitResult
.fit_module(module, loop=1) -> ModuleStep      # one module, once
.module_order(loop) -> list[ModuleSpec]        # the order for a given loop
.write_predictions(directory, values=None, n_points=200) -> list[Path]
.values                                        # current parameters
.models / .data                                # loaded models and datasets
.problem                                       # the underlying Problem
```

Fitting one module at a time, by hand:

```python
estimator = ModularEstimator(config)
for loop in range(1, 4):
    for module in estimator.module_order(loop):
        step = estimator.fit_module(module, loop)
        print(loop, module.id, step.cost_before, "->", step.cost_after)
```

---

## `Problem` — models, data, and the current parameter vector

```python
Problem(config, models=None, datasets=None)

.set_values(mapping)                       # update the shared vector
.snapshot() -> dict                        # a copy of it
.module_cost(module, values=None) -> float
.total_cost(values=None) -> float
.cost_report(values=None) -> dict          # per module, plus "__total__"
.residual_blocks(module, values=None) -> list[ResidualBlock]
.objective_for(module) -> ModuleObjective
.parameters_for(model, values=None) -> dict   # filtered to that model
.predictions(data, values=None, n_points=200) -> dict[str, ndarray]
.check()                                   # raises ProblemError if inconsistent
```

Evaluating a cost without fitting anything — useful for a noise floor, a
sensitivity scan, or a sanity check on published parameters:

```python
problem = Problem(config)
problem.set_values(PUBLISHED)
print(problem.cost_report())
# {'upstream': 0.0446, 'downstream': 0.0491, '__total__': 0.0469}
```

`ResidualBlock` carries `dataset`, `variable`, `times`, `observed`,
`simulated`, `residuals` and `weight` — everything needed for a residual plot.

---

## `ModuleObjective` — one module's cost as a function

What optimizers receive. All vectors are in **search space**, where a
`log`-scaled parameter appears as `log10(value)`.

```python
objective = problem.objective_for(config.module("upstream"))

objective(x) -> float                  # scalar cost
objective.residuals(x) -> ndarray      # residual vector, for least squares
objective.names                        # free parameter names, in order
objective.bounds                       # [(lo, hi), ...] in search space
objective.x0                           # current values, in search space
objective.to_model(x) -> dict          # search space -> parameter values
objective.to_search(values) -> ndarray # parameter values -> search space
objective.commit(x) -> dict            # write a solution back into the vector
objective.best_x / .best_cost          # best point actually evaluated
```

Evaluating a trial point does **not** commit it; only `commit` changes the
shared vector.

---

## `SimulationModel` — Antimony to trajectories

```python
SimulationModel(spec: ModelSpec, simulation: SimulationSpec | None = None,
                cache_size: int = 256)

.simulate(times, parameters=None, conditions=None, initial_conditions=None,
          variables=None, use_cache=True) -> dict[str, ndarray]
.has_parameter(name) / .has_variable(name) -> bool
.available_variables -> list[str]
.floating_species / .boundary_species / .global_parameters / .compartments
.sbml                                  # the translated SBML
.clear_cache()
```

```python
model = SimulationModel(ModelSpec(id="wt", antimony=SOURCE))
trace = model.simulate([0, 1, 2, 5], parameters={"Vmax1": 2.0},
                       initial_conditions={"S": 10.0})
trace["A"]        # ndarray at the requested times
```

Results are cached by their inputs, so repeated evaluation at the same point is
free. `antimony_to_sbml(text, label)` is available separately for translation
alone.

---

## Data

```python
load_dataset(spec: DatasetSpec) -> ExperimentData
load_datasets(specs) -> dict[str, ExperimentData]
```

`ExperimentData` has `id`, `model`, `measurements`, `weight`, `conditions`,
`initial_conditions`, plus `variables`, `n_points`, `has(v)`, `select(vs)` and
`time_grid(vs)`. Each `Measurement` carries `variable`, `times`, `values`,
optional `sigma`, and `n` / `scale`.

---

## Optimizers

```python
from pymodest import optimizers

optimizers.available() -> list[str]
optimizers.run(name, objective, x0=None, bounds=None, rng=None, **options)
optimizers.register(name, *aliases)          # decorator
optimizers.get_optimizer(name) -> callable
```

Running a backend against one module directly:

```python
import numpy as np

objective = problem.objective_for(config.module("upstream"))
result = optimizers.run("scatter_search", objective,
                        rng=np.random.default_rng(0), maxiter=20)
objective.commit(result.x)
```

`OptimizerResult` carries `x`, `fun`, `nfev`, `nit`, `success`, `message`,
`backend` and `extra`. See [optimizers.md](optimizers.md) for writing one.

---

## Results

`FitResult` fields: `parameters`, `cost`, `module_costs`, `loops`, `converged`,
`stop_reason`, `n_evaluations`, `seconds`, `initial_parameters`,
`initial_cost`, `study`.

```python
result.history()          # DataFrame, one row per module fit
result.loop_summary()     # DataFrame, one row per loop
result.parameter_table()  # DataFrame, initial beside estimate
result.steps              # list[ModuleStep], flattened
result.n_loops
result.to_dict()          # JSON-ready
result.save(directory)    # -> dict of written paths
```

`ModuleStep` fields: `loop`, `module`, `optimizer`, `cost_before`,
`cost_after`, `total_before`, `total_after`, `parameters`, `nfev`, `seconds`,
`accepted`, `message`. `LoopRecord` holds `loop`, `steps`, `total_cost`,
`module_costs`, `parameters`, `seconds`, `relative_improvement`.

---

## Exceptions

| exception | raised when |
| --- | --- |
| `ConfigError` | a configuration is malformed or inconsistent |
| `DataError` | a data table cannot be interpreted |
| `ProblemError` | models, data and modules do not fit together |
| `ModelError` | a model cannot be translated, loaded, or written to |
| `SimulationFailure` | an integration fails — caught by the estimator and scored |
| `OptimizerError` | an unknown or misconfigured backend |
