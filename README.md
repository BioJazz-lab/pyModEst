# pyModEst

Divide-and-conquer parameter estimation for biological models.

pyModEst reads models written in [Antimony](https://tellurium.readthedocs.io/en/latest/antimony.html),
a declaration of which parameters to fit, and one or more experimental datasets,
then estimates the parameters **module by module**: each module's parameters are
optimized against only that module's measured variables, while every other
parameter is held fixed. The procedure cycles through the modules for a finite
number of loops.

```
theta <- initial values
repeat up to max_loops times:
    for each module m:
        theta[m] <- argmin cost_m(theta[m] ; theta[not m] held fixed)
    stop when the total cost stops improving
```

One shared parameter set can be constrained by **several models** and **several
datasets** at once, which is the usual situation when wild-type and mutant
strains, or different experimental conditions, are described by related models.

---

## Documentation

Full reference in [`docs/`](https://github.com/BioJazz-Lab/pyModEst/tree/main/docs):

| | |
| --- | --- |
| [Concepts](https://github.com/BioJazz-Lab/pyModEst/blob/main/docs/concepts.md) | what module-wise fitting is, when it helps, how it fails |
| [Configuration](https://github.com/BioJazz-Lab/pyModEst/blob/main/docs/configuration.md) | the complete TOML reference |
| [Capabilities](https://github.com/BioJazz-Lab/pyModEst/blob/main/docs/capabilities.md) | what the package does, each pinned by a test |
| [Optimizers](https://github.com/BioJazz-Lab/pyModEst/blob/main/docs/optimizers.md) | the five backends, choosing one, adding one |
| [Python API](https://github.com/BioJazz-Lab/pyModEst/blob/main/docs/api.md) | using pyModEst as a library |
| [Troubleshooting](https://github.com/BioJazz-Lab/pyModEst/blob/main/docs/troubleshooting.md) | what goes wrong and what it means |
| [Examples](https://github.com/BioJazz-Lab/pyModEst/tree/main/docs/examples) | runnable scripts behind the documented output |
| [Releasing](https://github.com/BioJazz-Lab/pyModEst/blob/main/RELEASING.md) | how a version reaches PyPI |

## Why fit in modules

A joint fit over thirty parameters searches a thirty-dimensional space; a
module-wise fit searches several small ones in turn. Each module fit is small
enough for a global optimizer to solve reliably, and the biology usually
suggests the partition already: uptake parameters are constrained by upstream
metabolites, feedback constants by the species that does the inhibiting.

The trade-off is real and worth stating plainly. Because each module minimizes
its own objective rather than the joint one, the total cost is **not guaranteed
to decrease every loop** — modules are coupled through the shared model. pyModEst
handles this by tracking the best parameter set across the whole run and
returning that, not whatever the last loop happened to produce. If you want a
strictly monotone total, set `accept = "total"`, but read the warning under
[Acceptance policy](#acceptance-policy) first.

## Installation

The project is managed with [uv](https://docs.astral.sh/uv/). From a clone:

```bash
uv sync                   # create .venv and install from uv.lock, dev group included
```

`uv sync` installs the exact versions recorded in `uv.lock`, so everyone gets
the same environment. Commands then run through `uv run`, which keeps the
environment up to date for you — no manual activation needed:

```bash
uv run pymodest --version
uv run pytest
```

To add or change a dependency, edit `pyproject.toml` and run `uv lock` (or
`uv add <package>`, which does both), then commit the updated `uv.lock`.

Plain pip works too, if you would rather not use uv:

```bash
pip install -e .                       # runtime only
pip install -e . --group dev           # with pytest
```

Requires Python 3.11+ (the oldest version `libroadrunner` publishes wheels
for), and pulls in `antimony`, `libroadrunner`, `numpy`, `scipy` and `pandas`.

## Quick start

```bash
uv run pymodest template --out study.toml   # a commented starter configuration
uv run pymodest validate study.toml         # check models, data and modules agree
uv run pymodest fit study.toml              # run the estimation
```

Drop the `uv run` prefix if you have activated the environment yourself
(`source .venv/bin/activate`) or installed with pip.

Or from Python:

```python
from pymodest import load_config, fit

config = load_config("study.toml")
result = fit(config)

print(result.parameter_table())
print(result.loop_summary())
result.save(config.output_dir)
```

## The configuration file

One TOML file describes the whole study. Every path in it is resolved relative
to the file itself.

### Models

Several models may share one fitted parameter set. A parameter that exists in
only some of them is written only to those, and is therefore identified only by
their datasets.

```toml
[[models]]
id = "wt"
antimony_file = "models/wt.ant"

[models.observables]          # derived quantities, usable as module variables
Total = "A + B + C"

[[models]]
id = "feedback"
antimony_file = "models/feedback.ant"   # adds Ki; shares everything else
```

`antimony = """..."""` may be given inline instead of `antimony_file`.
`[models.overrides]` pins values for one model only.

### Datasets

Each dataset is measured on one model, under its own conditions.

```toml
[[datasets]]
id = "wt_low"
model = "wt"
file = "data/wt_low.csv"
format = "wide"               # wide: time,A,B,...   long: time,variable,value
weight = 1.0

[datasets.conditions]         # parameters set for this experiment
inducer = 0.0

[datasets.initial_conditions] # species starting values
S = 3.0
```

**Wide** files have one column per variable; a matching `A_sigma` (or `_sd`,
`_std`, `_err`) column supplies measurement errors. **Long** files have
`time, variable, value` and an optional `sigma`. Missing values are dropped per
variable, so variables need not share a measurement schedule. Data may also be
given inline under `[datasets.data]`.

### Modules

A module declares which parameters are fitted together and which measured
variables score them. **The modules must partition the parameter set** — every
fitted parameter belongs to exactly one module — which pyModEst checks at load
time.

```toml
[[modules]]
id = "upstream"
variables = ["A", "B"]        # what this module is scored on
# datasets = ["wt_low"]       # optional: restrict to particular experiments
# [modules.weights]           # optional: per-variable weights
# A = 2.0

[[modules.parameters]]
name = "Vmax1"
lower = 0.05
upper = 50.0
init = 1.0
scale = "log"                 # log | linear; log searches log10(value)
# fixed = true                # hold at init and exclude from the search

[modules.optimizer]           # optional per-module optimizer
name = "scatter_search"
maxiter = 25
```

Use `scale = "log"` for rate constants and affinities spanning orders of
magnitude — it is usually the difference between a fit that converges and one
that does not.

### Fitting

```toml
[fitting]
max_loops = 8
module_order = "as_listed"    # or a list of ids, "random", "round_robin_reversed"
tol = 1e-3                    # relative improvement counting as progress
atol = 1e-12                  # absolute floor, so a cost heading to zero terminates
patience = 2                  # loops without progress before stopping
accept = "module"             # module | total
seed = 7

[fitting.optimizer]
name = "differential_evolution"
maxiter = 40
popsize = 12

[fitting.objective]
scaling = "relative"          # relative | absolute | sigma | max_normalized
aggregation = "mean"          # mean | sum
epsilon = 1e-3

[fitting.simulation]
integrator = "cvode"
relative_tolerance = 1e-8
absolute_tolerance = 1e-10
```

## Residual scaling

Every residual is `simulated - observed`, then scaled:

| `scaling`         | residual                     | use when |
| ----------------- | ---------------------------- | -------- |
| `relative`        | `diff / (abs(obs) + epsilon)` | variables differ in magnitude (**default**) |
| `absolute`        | `diff`                        | all variables share units and scale |
| `sigma`           | `diff / sigma`                | you measured errors — gives chi-square residuals |
| `max_normalized`  | `diff / max(abs(obs))`        | relative scaling is unstable near zero |

With `relative`, raise `epsilon` above the noise level of near-zero
measurements; otherwise an observation of 0.001 dominates the objective.
Residuals are then multiplied by `sqrt(dataset weight x variable weight)` and
aggregated by mean (default) or sum of squares.

## Optimizers

Named in `[fitting.optimizer]` or per module in `[modules.optimizer]`. Any key
other than `name` is passed straight to the backend.

| name | what it is | notable options |
| ---- | ---------- | --------------- |
| `differential_evolution` | bounded global search (SciPy) — robust default | `maxiter`, `popsize`, `polish`, `workers` |
| `scatter_search` | population search with reference-set update and local refinement | `refset_size`, `maxiter`, `max_nfev`, `local_search` |
| `particle_swarm` | swarm with inertia damping and reflecting bounds | `n_particles`, `maxiter`, `inertia`, `cognitive`, `social` |
| `least_squares` | local Trust Region Reflective on the residual vector | `max_nfev`, `loss`, `diff_step` |
| `minimize` | SciPy local minimizers | `method` (`L-BFGS-B`, `Nelder-Mead`, `Powell`, ...) |

Aliases (`de`, `pso`, `ss`, `trf`, `nelder_mead`) work too. Register your own:

```python
from pymodest.optimizers import register, OptimizerResult

@register("my_method")
def my_method(objective, x0, bounds, rng, **options):
    ...                      # objective(x) -> cost; objective.residuals(x) -> vector
    return OptimizerResult(x=best_x, fun=best_cost, nfev=n)
```

The local methods (`least_squares`, `minimize`) only refine where they start.
Use them for a module whose parameters are already close, or after a global
pass; on their own they will sit in whatever basin the initial values fall in.

### Acceptance policy

`accept = "module"` (default) keeps a module fit whenever that module's own cost
improved — the plain divide-and-conquer rule. `accept = "total"` additionally
requires that the total cost across all modules did not rise.

Monotone sounds safer, but it can stall the search: an early step that improves
one module while temporarily worsening another is often exactly the step needed
to escape a poor starting region. In the shipped example, `"total"` gets stuck
at a cost of 0.86 while `"module"` reaches 0.060. Prefer the default unless you
have a specific reason.

## What a run produces

`result.save(directory)` writes:

| file | contents |
| ---- | -------- |
| `best_parameters.toml` | the estimates, ready to feed back to `pymodest simulate` |
| `parameters.csv` | initial value beside final estimate |
| `history.csv` | one row per module fit: costs before/after, evaluations, timing |
| `loop_summary.csv` | one row per loop: total and per-module costs |
| `fit_report.json` | the complete record, including every step |
| `predictions_<id>.csv` | simulated trace beside the observations, per dataset |

In Python, `FitResult` exposes `parameters`, `cost`, `module_costs`, `loops`,
`converged`, `stop_reason`, and the DataFrames `history()`, `loop_summary()`
and `parameter_table()`.

## Worked example

`examples/two_module_pathway/` fits a linear pathway

```
S --J1--> A --J2--> B --J3--> C --J4--> out
```

with **two models sharing one parameter set** — the wild type, and a strain in
which C inhibits its own uptake (adding `Ki`, which only that strain's data can
identify) — against **three datasets** at two substrate concentrations.
Parameters split into an upstream module (`Vmax1, Km1, k2`, scored on A and B)
and a downstream module (`Vmax3, Km3, k4, Ki`, scored on C).

```bash
cd examples/two_module_pathway
uv run python generate_data.py   # regenerate the synthetic measurements
uv run pymodest fit config.toml
```

The data carry 4% proportional noise. Evaluated at the generating parameters the
total cost is **0.0469** — the noise floor, the best any fit could do. Starting
from cost 237, the run converges in about 7 seconds to:

| | cost | mean abs. log10 error |
| --- | --- | --- |
| `differential_evolution` | 0.0604 | 0.066 |
| `scatter_search` | 0.0604 | 0.067 |
| `particle_swarm` | 0.0605 | 0.066 |

All three independent optimizers land on the same answer, which is good evidence
the estimator finds the true optimum of the module-wise objective. The small gap
to the noise floor is inherent to the method, not a failure to converge: each
module minimizes its own objective, so the fixed point differs slightly from the
joint least-squares optimum. `Ki` and `k4` are recovered closely; the `Vmax`/`Km`
pairs remain correlated at this noise level, as they would in any fit.

## Practical notes

- **Choose modules by what the data constrain.** A module's variables should
  respond to its parameters more strongly than to the rest. If two parameters
  are only identifiable together, put them in the same module.
- **Coupled modules need more loops.** The example's feedback strain makes the
  upstream variables depend on a downstream parameter; that coupling is what the
  repeated loops resolve.
- **Watch `loop_summary.csv`.** A total cost that oscillates rather than settles
  means the partition is fighting itself — merge the modules that trade against
  each other, or widen the variables scoring one of them.
- **A failed integration is scored, not raised.** Parameter sets that make the
  ODEs unsolvable get a large finite cost, so a global search can step over them.
- **Set `seed`** in `[fitting]` for reproducible runs; every backend derives its
  randomness from it.

## Development

```bash
uv sync
uv run pytest                 # 118 tests
```

`uv.lock` is committed so results are reproducible. It resolves for every
platform at once; the pinned `antimony` and `libroadrunner` wheels cover macOS
(Intel and Apple silicon), Windows, and Linux on x86-64. Linux on arm64 is not
covered by the current `antimony` release, so on that platform install without
the lock (`uv pip install -e .`), which falls back to the newest version that
does publish an arm64 wheel.

Layout:

```
src/pymodest/
  config.py       TOML schema, dataclasses, validation
  model.py        Antimony -> SBML -> roadrunner, with result caching
  data.py         wide/long experiment tables
  objective.py    residual assembly, scaling, per-module objectives
  estimator.py    the divide-and-conquer loop
  result.py       records, DataFrames, on-disk report
  cli.py          the pymodest command
  optimizers/     registry, SciPy backends, particle swarm, scatter search
```

## License

MIT.
