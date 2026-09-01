# Configuration reference

One TOML file describes a whole study. Every path inside it resolves relative
to the file's own directory, so a study folder can be moved or copied intact.

`pymodest template --out study.toml` writes a commented skeleton, and
`pymodest validate study.toml` checks it — including that the modules partition
the parameters and that every name exists in the models and data — before any
fitting happens.

Unknown keys are rejected with a message naming what is allowed, so a typo
fails loudly instead of being silently ignored. The one exception is optimizer
options, which are passed through to the backend and therefore free-form.

---

## `[study]`

| key | type | default | meaning |
| --- | --- | --- | --- |
| `name` | string | `"pymodest-study"` | label used in logs and the report |
| `output_dir` | path | `"results"` | where `fit` writes; `--out` overrides it |

---

## `[[models]]`

One entry per model. Several models may share one fitted parameter set; give
exactly one of `antimony_file` or `antimony`.

| key | type | default | meaning |
| --- | --- | --- | --- |
| `id` | string | required | referenced by datasets |
| `antimony_file` | path | — | model source, relative to the config file |
| `antimony` | string | — | model source inline |
| `overrides` | table | `{}` | values pinned for this model only |
| `observables` | table | `{}` | derived quantities usable as module variables |
| `description` | string | `""` | free text |

```toml
[[models]]
id = "wt"
antimony_file = "models/wt.ant"

[models.observables]
Total = "A + B + C"          # an expression over the model's own variables

[models.overrides]
compartment_volume = 1.0     # applied to this model on every simulation
```

Observable expressions are evaluated over the simulated traces with numpy, so
`exp`, `log`, `log10`, `sqrt`, `abs`, `maximum`, `minimum` and `where` are
available. A parameter that exists in only some models is written only to those.

---

## `[[datasets]]`

One entry per experiment. Each is attached to exactly one model.

| key | type | default | meaning |
| --- | --- | --- | --- |
| `id` | string | required | referenced by `[[modules]].datasets` |
| `model` | string | required | which model simulates this experiment |
| `file` | path | — | CSV or TSV of measurements |
| `data` | table | — | measurements inline, as columns |
| `format` | string | `"wide"` | `wide` or `long` |
| `weight` | float | `1.0` | multiplies this dataset's squared residuals |
| `conditions` | table | `{}` | parameters set for this experiment |
| `initial_conditions` | table | `{}` | species starting values |
| `time_column` | string | `"time"` | name of the time column |
| `description` | string | `""` | free text |

Give exactly one of `file` or `data`.

```toml
[[datasets]]
id = "wt_low"
model = "wt"
file = "data/wt_low.csv"
weight = 1.0

[datasets.conditions]
inducer = 0.0                # a parameter, for this experiment only

[datasets.initial_conditions]
S = 3.0                      # a species starting value
```

**Wide** files have one column per variable, plus optional error columns named
`<variable>_sigma` (also `_sd`, `_std`, `_err`). **Long** files have
`time, variable, value` and an optional `sigma` column. In both, missing values
are dropped per variable, so variables need not share a measurement schedule.

---

## `[[modules]]`

The partition. Every fitted parameter must appear in exactly one module.

| key | type | default | meaning |
| --- | --- | --- | --- |
| `id` | string | required | module name |
| `variables` | list | required | measured variables scoring this module |
| `parameters` | array of tables | required | the parameters fitted together |
| `weights` | table | `{}` | per-variable weights within this module |
| `datasets` | list | all | restrict scoring to these datasets |
| `optimizer` | table | study default | override the optimizer for this module |
| `description` | string | `""` | free text |

### `[[modules.parameters]]`

| key | type | default | meaning |
| --- | --- | --- | --- |
| `name` | string | required | must be settable in at least one model |
| `lower` | float | required | lower bound |
| `upper` | float | required | upper bound, must exceed `lower` |
| `init` | float | midpoint | starting value; geometric midpoint when `scale = "log"` |
| `scale` | string | `"linear"` | `linear` or `log`; `log` requires `lower > 0` |
| `fixed` | bool | `false` | hold at `init` and exclude from the search |

```toml
[[modules]]
id = "upstream"
variables = ["A", "B"]
datasets = ["wt_low", "wt_high"]     # optional

[modules.weights]
A = 2.0                              # count A twice as much as B

[[modules.parameters]]
name = "Vmax1"
lower = 0.05
upper = 50.0
init = 1.0
scale = "log"

[modules.optimizer]
name = "scatter_search"
maxiter = 25
```

Use `scale = "log"` for rate constants and affinities spanning orders of
magnitude — the search then happens in `log10` space, which is usually the
difference between a fit that converges and one that does not.

---

## `[fitting]`

| key | type | default | meaning |
| --- | --- | --- | --- |
| `max_loops` | int | `10` | maximum sweeps over all modules |
| `module_order` | string or list | `"as_listed"` | `as_listed`, `random`, `round_robin_reversed`, or an explicit list of every module id |
| `tol` | float | `1e-6` | relative improvement counting as progress |
| `atol` | float | `1e-12` | absolute improvement counting as progress |
| `patience` | int | `2` | loops without progress before stopping |
| `accept` | string | `"module"` | `module` or `total` — see below |
| `seed` | int | none | makes a run reproducible |
| `refine_after` | int | `1` | loop after which `[fitting.refine]` takes over |

A loop counts as making no progress when the total cost improved by less than
`tol` *relatively* **and** less than `atol` *absolutely*, measured against the
best cost so far rather than the previous loop. The absolute floor matters: a
cost converging geometrically towards zero keeps producing large relative
gains forever, so a relative test alone never terminates.

### Acceptance policy

`accept = "module"` (default) keeps a module fit whenever that module's own
cost improved. `accept = "total"` additionally requires that the total cost did
not rise, making the total monotone.

Monotone is not the same as better. On the shipped example:

| policy | final cost | loops | steps that raised the total |
| --- | --- | --- | --- |
| `module` | 0.0604 | 5 | 4 |
| `total` | 0.8568 | 3 | 0 |

`total` blocks the step that escapes the poor starting region, because that
step temporarily hurts the other module. Prefer the default; the estimator
already returns the best parameters seen across the run, so tolerating a bad
loop costs nothing.

### `[fitting.optimizer]` and `[fitting.refine]`

`name` picks the backend; every other key is passed straight to it. `refine`
names an optimizer used from loop `refine_after + 1` onward — a global search
first, a cheap local one thereafter.

```toml
[fitting.optimizer]
name = "differential_evolution"
maxiter = 40
popsize = 12

[fitting.refine]
name = "least_squares"
max_nfev = 300
```

See [optimizers.md](optimizers.md) for each backend's options.

### `[fitting.objective]`

| key | type | default | meaning |
| --- | --- | --- | --- |
| `scaling` | string | `"relative"` | `relative`, `absolute`, `sigma`, `max_normalized` |
| `aggregation` | string | `"mean"` | `mean` or `sum` of squared residuals |
| `epsilon` | float | `1e-8` | floor in the `relative` denominator |
| `default_sigma` | float | `1.0` | used by `sigma` scaling where no error was measured |

| `scaling` | residual | use when |
| --- | --- | --- |
| `relative` | `diff / (abs(obs) + epsilon)` | variables differ in magnitude — **default** |
| `absolute` | `diff` | all variables share units and scale |
| `sigma` | `diff / sigma` | you measured errors; gives chi-square residuals |
| `max_normalized` | `diff / max(abs(obs))` | relative scaling is unstable near zero |

With `relative`, raise `epsilon` above the noise level of near-zero
measurements — otherwise an observation of 0.001 dominates the objective.
Residuals are then multiplied by `sqrt(dataset weight × variable weight)`.

### `[fitting.simulation]`

| key | type | default | meaning |
| --- | --- | --- | --- |
| `integrator` | string | `"cvode"` | roadrunner integrator |
| `relative_tolerance` | float | `1e-8` | integrator relative tolerance |
| `absolute_tolerance` | float | `1e-10` | integrator absolute tolerance |
| `stiff` | bool | `true` | use the stiff solver (cvode only) |
| `max_steps` | int | `20000` | internal step cap per interval |
