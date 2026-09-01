# Capabilities

What the package does, each item demonstrated by a runnable example and pinned
by tests. Output below is real — copied from the scripts in
[`examples/`](examples/), which `examples/run_all.py` re-runs.

---

## 1. Module-wise fitting

The core behaviour: a module's parameters are optimized against that module's
variables while every other parameter stays fixed.

Fitting one module moves only its own parameters:

```python
estimator = ModularEstimator(study)
before = estimator.values
estimator.fit_module(study.module("upstream"), loop=1)
after = estimator.values
# after["k2"] == before["k2"]      -- downstream untouched
# after["Vmax1"] != before["Vmax1"]
```

And a module's cost responds only to its own parameters — perturbing a
downstream parameter leaves the upstream cost identical:

```python
problem.set_values(TRUE_PARAMETERS)
before = problem.module_cost(upstream)
problem.set_values({"k3": 4.0})
assert problem.module_cost(upstream) == before          # unchanged
assert problem.module_cost(downstream) > before         # but this moved
```

From a poor starting point, the loop recovers noise-free parameters to a
fraction of a percent ([`01_first_fit.py`](examples/01_first_fit.py)):

```
cost 0.2096 -> 6.799e-08 in 6 loops (7384 evaluations)

parameter     true   estimate    error
Km1          4.000     4.0053    0.1%
Vmax1        2.000     2.0010    0.0%
k2           0.600     0.5999    0.0%
k3           0.300     0.3000    0.0%
```

*Verified by* `test_fitting_one_module_only_moves_that_module_s_parameters`,
`test_a_module_only_sees_its_own_variables`,
`test_the_loop_recovers_the_generating_parameters`,
`test_a_module_step_never_worsens_its_own_cost`.

---

## 2. One parameter set across several models

Wild type and mutant are usually related models sharing most kinetics. A
parameter present in only some models is written only to those, so it is
identified only by their data ([`02_shared_parameters.py`](examples/02_shared_parameters.py)):

```
which model has which parameter
parameter     wt  feedback
Vmax1       True      True
Km1         True      True
k2          True      True
k3          True      True
Ki         False      True

parameters actually written to each model:
  wt       ['Km1', 'Vmax1', 'k2', 'k3']
  feedback ['Ki', 'Km1', 'Vmax1', 'k2', 'k3']

cost 1.36e-08 | 8 loops | 12530 evals | worst error 0.11%
```

`Ki` is recovered from the feedback dataset alone. Nothing special is required:
declare both models, attach each dataset to one of them, and put `Ki` in
whichever module its variable belongs to.

*Verified by* `test_example_has_two_models_sharing_one_parameter_set`,
`test_parameters_are_filtered_to_the_models_that_have_them`.

---

## 3. Several datasets, each with its own conditions

Every dataset carries its own `conditions` (parameters) and
`initial_conditions` (species), and its own `weight`. The same model under two
starting conditions is simply two datasets — often what makes a saturating
parameter identifiable at all.

```toml
[[datasets]]
id = "wt_low"
model = "wt"
file = "data/wt_low.csv"
weight = 2.0
[datasets.initial_conditions]
S = 3.0
```

Conditions never leak between simulations: the model is reset to its SBML
defaults before each run, so alternating two datasets gives the same answers as
running each alone. A module can also be restricted to a subset of datasets
with `datasets = [...]`.

*Verified by* `test_conditions_do_not_leak_between_simulations`,
`test_a_condition_set_once_does_not_persist`,
`test_dataset_weight_scales_the_cost`,
`test_a_module_can_be_restricted_to_particular_datasets`.

---

## 4. Data in the shape you have it

Wide files, long files and inline tables all produce the same measurements
([`03_data_and_objective.py`](examples/03_data_and_objective.py)):

```
  wide file   variables=['A', 'B']  points=6  A=[0.0, 1.0, 1.8]
  long file   variables=['A', 'B']  points=6  A=[0.0, 1.0, 1.8]
  inline      variables=['A', 'B']  points=6  A=[0.0, 1.0, 1.8]
```

A matching error column is recognised and is not mistaken for a variable:

```
  variables:  ['A']   <- A_sigma is not a variable
  sigma:      [0.05, 0.1, 0.2]
  suffixes accepted: _sigma, _sd, _std, _err
```

Variables need not share a measurement schedule — missing values are dropped
per variable, not per row:

```
  A: 3 points at t=[0.0, 2.0, 3.0]
  B: 3 points at t=[0.0, 1.0, 3.0]
```

Rows are sorted by time on load, and a dataset whose measurements start after
`t = 0` is simulated from zero and sampled at the measured times.

*Verified by* `test_wide_csv_is_loaded`, `test_long_format`,
`test_inline_data_is_accepted`,
`test_sigma_columns_are_recognised_and_not_treated_as_variables`,
`test_missing_values_are_dropped_per_variable`, `test_rows_are_sorted_by_time`,
`test_time_grid_not_starting_at_zero`.

---

## 5. Derived observables

Measured quantities are often combinations of model species — a total pool, a
ratio, a labelled fraction. Declare them per model and use them as module
variables like any species:

```toml
[models.observables]
Total = "A + B + C"
Ratio = "B / (A + 1e-9)"
```

Expressions are evaluated over the simulated traces with numpy available
(`exp`, `log`, `log10`, `sqrt`, `abs`, `maximum`, `minimum`, `where`). A bad
expression is reported with the model and observable named.

*Verified by* `test_observables_are_computed`,
`test_bad_observable_expression_is_reported`.

---

## 6. Residual scaling and weighting

Two variables measured on very different scales, each simulated 10% too high:

```
  observed   [1.0, 100.0]   (sigma [0.5, 5.0])
  simulated  [1.1, 110.0]

  scaling                          residuals   what it does
  absolute                       [0.1, 10.0]   plain difference; the large variable dominates
  relative                        [0.1, 0.1]   divided by |observed|; both count equally  <- default
  sigma                           [0.2, 2.0]   divided by measurement error; chi-square
  max_normalized                [0.001, 0.1]   divided by that variable's peak value
```

With `absolute`, the second point contributes 10 000× the first and the fit
effectively ignores the small variable — which is why `relative` is the
default. Residuals are then scaled by `sqrt(dataset weight × variable weight)`,
so `[modules.weights]` and a dataset's `weight` compose.

*Verified by* the five `test_*_scaling_*` tests and
`test_dataset_weight_scales_the_cost`.

---

## 7. Parameter scales and held parameters

`scale = "log"` searches `log10(value)`, which is what makes a range spanning
orders of magnitude tractable; the default `init` becomes the geometric
midpoint. `fixed = true` holds a parameter at `init` — still applied to the
model, just not searched ([`05_loop_control.py`](examples/05_loop_control.py)):

```
  k2 pinned at 0.6  -> estimate 0.6000 (unchanged)
  k3 free           -> estimate 0.3000 (true 0.3)
```

A module whose parameters are all fixed is skipped entirely rather than failing.

*Verified by* `test_log_parameter_uses_geometric_midpoint_and_log_bounds`,
`test_fixed_parameters_are_held_but_still_applied`,
`test_a_module_with_no_free_parameters_is_skipped`.

---

## 8. Five optimizers, and your own

All reached through one registry, so swapping backend is a single string.
On an easy problem the local methods are the best value
([`04_optimizers.py`](examples/04_optimizers.py)):

```
backend                 kind            cost    evals    sec  worst err
-----------------------------------------------------------------------
differential_evolution  global       6.8e-08     7384    0.8     0.13%
scatter_search          global      2.43e-08    10199    1.0     0.05%
particle_swarm          global       1.3e-05     5800    0.5     1.12%
least_squares           local       2.15e-08      279    0.0     0.05%
minimize/L-BFGS-B       local        5.2e-08      684    0.1     0.07%
minimize/Nelder-Mead    local       2.13e-08     1790    0.2     0.05%
```

On the shipped example — 7 parameters, 4% noise, coupled models — they part
company completely:

```
backend                 kind            cost    evals    sec
------------------------------------------------------------
differential_evolution  global       0.06036    17584    6.7
scatter_search          global       0.06037    14515    4.9
least_squares           local          60.18      648    0.3
minimize/Nelder-Mead    local          31.37     2454    1.0
```

The global searches reach the noise floor (0.0469); the local ones sit about a
thousand times above it, because they never leave the region the initial values
put them in. That two independent global methods agree to four significant
figures is the useful evidence that the answer is real.

Registering a backend is a decorator:

```python
from pymodest.optimizers import OptimizerResult, register

@register("random_search", "rs")
def random_search(objective, x0, bounds, rng, n_samples=400, **_):
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    points = lower + rng.random((n_samples, lower.size)) * (upper - lower)
    costs = np.array([objective(p) for p in points])
    best = int(np.argmin(costs))
    return OptimizerResult(x=points[best], fun=float(costs[best]), nfev=n_samples)
```

It is then usable from TOML as `[fitting.optimizer] name = "random_search"`.

*Verified by* `test_backend_finds_the_minimum`,
`test_backend_stays_inside_the_bounds`, `test_aliases_resolve`,
`test_a_custom_backend_can_be_registered`,
`test_different_backends_reach_the_same_answer`.

---

## 9. Loop control

Module order, four ways:

```
  as_listed              loop 1: upstream -> downstream   loop 2: upstream -> downstream
  round_robin_reversed   loop 1: upstream -> downstream   loop 2: downstream -> upstream
  random                 loop 1: downstream -> upstream   loop 2: upstream -> downstream
  explicit list          loop 1: downstream -> upstream   loop 2: downstream -> upstream
```

Stopping is either the loop cap or convergence:

```
  max_loops=2         2 loops -- reached max_loops (2)
  max_loops=50        17 loops -- converged: the total cost improved by less
                                  than 1e-12 for 2 consecutive loop(s)
```

The acceptance policy only matters when modules interact. On a decoupled toy
the two policies are identical; on the coupled example they diverge sharply:

```
  decoupled toy:
    accept=module   cost 6.8e-08   steps that raised the total: 0
    accept=total    cost 6.8e-08   steps that raised the total: 0

  coupled example:
    accept=module   cost 0.06036   5 loops   steps that raised the total: 4
    accept=total    cost 0.8568    3 loops   steps that raised the total: 0
```

*Verified by* `test_round_robin_reversed_alternates`,
`test_explicit_order_is_honoured`, `test_random_order_covers_every_module`,
`test_max_loops_is_respected`, `test_convergence_stops_early`,
`test_accept_total_keeps_the_total_cost_monotone`.

---

## 10. Results you can inspect

`fit()` returns a `FitResult` with DataFrames, not just numbers.

```python
result.history()          # one row per module fit
result.loop_summary()     # one row per loop
result.parameter_table()  # initial beside estimate
```

```
 loop     module              optimizer  cost_before  cost_after  nfev  accepted
    1   upstream differential_evolution     0.143793    0.103182   552      True
    1 downstream differential_evolution     0.283069    0.002714   569      True
    2   upstream differential_evolution     0.052321    0.000910   629      True
    2 downstream differential_evolution     0.000235    0.000027   623      True

 loop  total_cost  relative_improvement  cost:upstream  cost:downstream
    1    0.027518              0.868731       0.052321         0.002714
    2    0.000204              0.992595       0.000380         0.000027
    3    0.000020              0.903923       0.000037         0.000003
```

`result.save(directory)` writes `fit_report.json`, `history.csv`,
`loop_summary.csv`, `parameters.csv` and `best_parameters.toml` — the last
ready to feed back to `pymodest simulate --parameters`:

```toml
# pyModEst estimated parameters
# study = 'toy'
# cost = 1.95783e-05   loops = 3   converged = False

[parameters]
Km1 = 4.055626505106976
Vmax1 = 2.0037312903490614
```

A `callback` fires after every module fit, for live progress:

```python
fit(config, callback=lambda step: print(step.module, step.cost_after))
```

*Verified by* `test_history_records_every_module_fit`,
`test_results_are_written_to_disk`, `test_predictions_are_written_per_dataset`,
`test_progress_callback_sees_each_step`.

---

## 11. Mistakes caught before fitting

Configuration and wiring errors are reported by name, at load time:

```
  overlapping modules  -> parameter 'Vmax1' appears in both module 'upstream' and
                          module 'other'; modules must partition the parameter set
  unknown parameter    -> fitted parameter 'nope' is not a settable parameter in any model
  unmeasured variable  -> module 'upstream': variable 'ghost' is not measured in any
                          of its datasets
  log with zero bound  -> parameter 'k': log scale requires a strictly positive lower bound
```

Also caught: datasets naming an unknown model, a `module_order` that omits a
module, unknown TOML keys, bounds the wrong way round, an `init` outside its
range, weights naming a non-module variable, and missing files.

`pymodest validate study.toml` runs all of this and prints the study, so none
of it costs a fitting run to discover.

*Verified by* the `test_config.py` suite plus
`test_unknown_fitted_parameter_is_caught_at_construction`,
`test_unmeasured_module_variable_is_caught`.

---

## 12. Robustness and reproducibility

A parameter set the integrator cannot handle is **scored, not raised** — it
gets a large finite cost so a global search can step over the region instead of
the run aborting. Simulation results are cached by their inputs, so repeated
evaluations at the same point are free.

Setting `seed` in `[fitting]` makes a run reproducible; every backend derives
its randomness from it.

*Verified by* `test_infeasible_parameters_are_scored_not_raised`,
`test_results_are_cached_by_inputs`, `test_a_seed_makes_the_run_reproducible`,
`test_backends_are_reproducible_given_a_seed`.

---

## 13. Command line

```bash
pymodest template --out study.toml   # commented starter config
pymodest validate study.toml         # check models, data and modules agree
pymodest fit study.toml              # run the estimation
pymodest simulate study.toml --parameters best_parameters.toml
pymodest optimizers                  # list registered backends
```

`fit` accepts `--loops`, `--out`, `--optimizer`, `--seed` and
`--no-predictions`. A broken configuration exits with status 2 and a message on
stderr.

*Verified by* the `test_cli_and_example.py` suite.
