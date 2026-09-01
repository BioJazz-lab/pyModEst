# Optimizers

Every backend is reached through one registry, so changing optimizer is a
single string in the config. Name one in `[fitting.optimizer]`, or per module
in `[modules.optimizer]`. Any key other than `name` is passed straight to the
backend, so options are not validated by pyModEst — a misspelled option is
ignored by the backend rather than rejected.

```bash
pymodest optimizers          # list what is registered
```

## Choosing one

The honest summary is that it depends on how good your starting point is.

| | global search | local search |
| --- | --- | --- |
| backends | `differential_evolution`, `scatter_search`, `particle_swarm` | `least_squares`, `minimize` |
| cost | thousands of evaluations | hundreds |
| finds | the region *and* the point | the nearest minimum only |
| use when | you do not know roughly where the answer is | you are already close |

Measured on an easy, noise-free problem, the local methods win outright —
`least_squares` matched the global searches with 25× fewer evaluations. On the
shipped example, with 4% noise and coupled modules, the same local methods
stall about a thousand times above the noise floor while the global ones reach
it. Both tables are in [capabilities.md](capabilities.md#8-five-optimizers-and-your-own).

The practical rule: search globally when you do not already know the answer's
neighbourhood, then refine locally. `[fitting.refine]` does exactly that.

```toml
[fitting.optimizer]
name = "differential_evolution"
maxiter = 40

[fitting.refine]              # takes over after loop `refine_after` (default 1)
name = "least_squares"
max_nfev = 300
```

When several independent global methods converge on the same answer, that
agreement is the strongest practical evidence the fit is real. It costs one
extra run: `pymodest fit study.toml --optimizer scatter_search`.

---

## `differential_evolution` — aliases `de`, `diffevo`

Bounded global search from SciPy. The robust default for wide biological
ranges.

| option | default | meaning |
| --- | --- | --- |
| `maxiter` | 100 | generations |
| `popsize` | 15 | population multiplier (× number of parameters) |
| `tol` | 1e-6 | convergence tolerance |
| `mutation` | `(0.5, 1.0)` | differential weight, or a range for dithering |
| `recombination` | 0.7 | crossover probability |
| `polish` | true | finish with a local L-BFGS-B step |
| `init` | `latinhypercube` | initial population strategy |
| `strategy` | `best1bin` | mutation strategy |
| `workers` | 1 | parallel processes; `-1` uses all cores |
| `use_x0` | true | seed the population with the current values |

`use_x0` is what lets a later loop start from where the previous one finished
rather than re-searching from scratch.

## `scatter_search` — aliases `ss`, `scatter`, `ess`

Population search with a reference set balancing quality and diversity, plus
periodic local refinement — the scheme that has worked well for kinetic models
(Egea et al.'s eSS). Implemented directly on numpy.

| option | default | meaning |
| --- | --- | --- |
| `refset_size` | 10 | reference set size |
| `pool_multiplier` | 10 | initial pool = `refset_size × this` |
| `maxiter` | 30 | iterations |
| `max_nfev` | 4000 | hard evaluation budget |
| `local_search` | `Nelder-Mead` | improvement method, or falsy to disable |
| `local_maxiter` | 40 | budget per local search |
| `local_frequency` | 2 | run local search every N iterations |
| `patience` | 8 | iterations without improvement before stopping |

Often the best value of the three global methods here: on the shipped example
it matched differential evolution using ~17% fewer evaluations.

## `particle_swarm` — aliases `pso`, `swarm`

Swarm with linearly damped inertia, velocity clamping and reflecting walls at
the bounds.

| option | default | meaning |
| --- | --- | --- |
| `n_particles` | 24 | swarm size |
| `maxiter` | 60 | iterations |
| `inertia` / `inertia_final` | 0.72 / 0.35 | inertia, decayed linearly |
| `cognitive` / `social` | 1.5 / 1.5 | pull toward personal / global best |
| `velocity_fraction` | 0.25 | velocity clamp, as a fraction of the range |
| `patience` | 15 | iterations without improvement before stopping |

## `least_squares` — aliases `trf`, `lsq`

SciPy Trust Region Reflective on the **residual vector** rather than the scalar
cost, which gives it far more information per evaluation. The cheapest way to
polish a nearly-converged fit.

| option | default | meaning |
| --- | --- | --- |
| `method` | `trf` | `trf`, `dogbox` |
| `max_nfev` | 200 | evaluation budget |
| `loss` | `linear` | `soft_l1`, `huber`, `cauchy` for robustness to outliers |
| `f_scale` | 1.0 | soft-margin scale for robust losses |
| `diff_step` | 1e-4 | relative finite-difference step |
| `ftol` / `xtol` / `gtol` | 1e-10 | termination tolerances |

`loss = "soft_l1"` is worth trying when a few measurements are suspect.

## `minimize`

SciPy's general local minimizers on the scalar objective.

| option | default | meaning |
| --- | --- | --- |
| `method` | `L-BFGS-B` | `L-BFGS-B`, `TNC`, `Nelder-Mead`, `Powell`, … |
| `maxiter` | 200 | iterations |
| `eps` | 1e-5 | finite-difference step (gradient methods) |
| `tol` | 1e-10 | termination tolerance |

Any other key is forwarded into SciPy's `options` dict.

---

## Writing a backend

A backend is a callable registered by name:

```python
import numpy as np
from pymodest.optimizers import OptimizerResult, register

@register("my_method", "mm")          # extra names are aliases
def my_method(objective, x0, bounds, rng, **options):
    ...
    return OptimizerResult(x=best_x, fun=best_cost, nfev=n)
```

What you receive:

| argument | is |
| --- | --- |
| `objective` | callable — `objective(x)` returns the scalar cost |
| `objective.residuals(x)` | the residual vector, for least-squares methods |
| `objective.names` | the free parameter names, in order |
| `x0` | current values, in search space |
| `bounds` | `[(lo, hi), …]` in search space |
| `rng` | a seeded `numpy.random.Generator` — use it, so runs stay reproducible |

Everything is in **search space**: a parameter declared `scale = "log"` is
presented as `log10(value)`, and the transform back is handled for you. Return
whatever point you found; the caller clips it to the bounds, and independently
keeps the best point the objective actually saw, so returning a slightly worse
final point cannot lose a good one.

`pymodest.optimizers.latin_hypercube(rng, bounds, n)` and `sample_uniform` are
available for initial designs.

A complete worked backend is in
[`examples/04_optimizers.py`](examples/04_optimizers.py).
