# Troubleshooting

## The fit does not improve much

**Check the starting point matters less than the bounds.** A `log`-scaled
parameter whose true value sits outside `[lower, upper]` can never be found.
`pymodest validate` prints every range.

**Check you are not using a local optimizer cold.** `least_squares` and
`minimize` only refine where they start. On the shipped example they stall
about a thousand times above the noise floor. Use a global backend first.

**Check the objective is not dominated by one variable.** With
`scaling = "absolute"`, a variable measured in thousands drowns one measured in
units. `relative` is the default for this reason.

## The total cost goes up between loops

Expected, and not a bug. Each module minimises its own objective, so a step
that helps one module can hurt another when they are coupled. On the shipped
example four module fits raise the total. The estimator returns the best
parameter set seen across the whole run, so a bad loop cannot be the answer.

It becomes a real problem only when the cost *oscillates without settling* —
visible in `loop_summary.csv`. That means two modules are fighting over the
same information, and the fix is to merge them.

## The fit stops after one or two loops

`patience` (default 2) is the number of loops without progress before stopping,
and progress means clearing `tol` relatively **or** `atol` absolutely, measured
against the best cost so far. If the first loop lands near a local minimum the
run will stop early. Raise `patience`, or lower `tol`.

If you set `accept = "total"`, early stopping is expected: that policy blocks
the step needed to escape a poor region. Use the default `module`.

## A near-zero measurement dominates everything

`relative` scaling divides by `abs(observed) + epsilon`. An observation of
0.001 with `epsilon = 1e-8` produces a residual scaled by 1000. Raise
`epsilon` to around the noise floor of your measurements, or switch that study
to `max_normalized`.

## `parameter 'X' is not a settable parameter in any model`

The name must be a global parameter, compartment or boundary species in at
least one model — check spelling against the Antimony source. Species initial
values are set through a dataset's `initial_conditions`, not by fitting them as
parameters.

## `variable 'X' is not measured in any of its datasets`

A module's `variables` must appear as columns in at least one dataset that
module scores. If the module has a `datasets = [...]` restriction, the variable
must be present in one of *those*.

## `modules must partition the parameter set`

Every fitted parameter belongs to exactly one module. The message names the
parameter and both modules.

## Integration failures

A parameter set the integrator cannot solve is scored as infeasible rather than
raising, so a global search steps over it. If a run seems to spend all its time
there, the bounds are probably admitting unphysical regions — tighten them, or
switch the parameter to `log` scale.

If a specific model fails everywhere, loosen `[fitting.simulation]` tolerances
or raise `max_steps`.

## Fitting is unexpectedly slow

Cost is dominated by the number of ODE integrations: roughly
`max_loops × modules × optimizer evaluations × datasets`. To speed things up,
in order of effect:

- Reduce `maxiter` / `popsize`, or use `[fitting.refine]` so only the first
  loop pays for a global search.
- Set `workers = -1` for `differential_evolution` to use all cores.
- Reduce the number of loops once you know how many it takes to converge.

Note that pyModEst avoids writing species *initial* values on every evaluation,
because in roadrunner 2.10 an `init(...)` write costs ~53 ms against ~0.15 ms
for a whole reset-set-simulate cycle. If you drive `SimulationModel` yourself,
avoid `init(...)` in a hot loop for the same reason.

## Results differ between machines

`antimony` and `libroadrunner` do not publish wheels for every platform, so a
lock cannot pin one version everywhere:

| platform | antimony | libroadrunner |
| --- | --- | --- |
| macOS (Apple silicon and Intel), Linux x86-64, Windows | 3.1.3 | 2.10.0 |
| Linux arm64 | 2.14.0 | 2.7.0 |

The test suite passes on both combinations. If a result looks
platform-dependent, check the versions first. On Linux arm64, `uv sync` cannot
install the lock at all — use `uv pip install -e .`.

## Runs are not reproducible

Set `seed` in `[fitting]`. Every backend derives its randomness from it. Note
that `workers != 1` in `differential_evolution` changes the evaluation order
and so the trajectory, though not the quality of the answer.
