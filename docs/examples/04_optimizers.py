"""The optimizer backends, compared on one study, and how to add your own.

Every backend is called through the same registry, so swapping one for another
is a single string. They are not interchangeable in behaviour, though, and the
table below shows why.

    uv run python docs/examples/04_optimizers.py
"""

from __future__ import annotations

import logging

import numpy as np

logging.disable(logging.INFO)   # quiet the per-module fit log

from dataclasses import replace

from pymodest import fit, optimizers
from pymodest.config import OptimizerSpec
from pymodest.optimizers import OptimizerResult, register

from _toy import TRUE, report, single_model_study

print("registered backends:", ", ".join(optimizers.available()))
print("aliases: de, pso, ss/ess, trf/lsq, nelder_mead ...\n")

base = single_model_study()

CASES = [
    ("differential_evolution", {"maxiter": 30, "popsize": 10}, "global"),
    ("scatter_search",         {"maxiter": 20, "refset_size": 8, "max_nfev": 1500}, "global"),
    ("particle_swarm",         {"maxiter": 40, "n_particles": 20}, "global"),
    ("least_squares",          {"max_nfev": 300}, "local"),
    ("minimize",               {"method": "L-BFGS-B", "maxiter": 200}, "local"),
    ("minimize",               {"method": "Nelder-Mead", "maxiter": 400}, "local"),
]

print(f"{'backend':<24}{'kind':<8}{'cost':>12}{'evals':>9}{'sec':>7}{'worst err':>11}")
print("-" * 71)
for name, options, kind in CASES:
    label = f"{name}/{options['method']}" if name == "minimize" else name
    config = replace(base, fitting=replace(base.fitting,
                                           optimizer=OptimizerSpec(name, options)))
    result = fit(config)
    worst = max(abs(result.parameters[n] - v) / v for n, v in TRUE.items()
                if n in result.parameters)
    print(f"{label:<24}{kind:<8}{result.cost:>12.3g}{result.n_evaluations:>9}"
          f"{result.seconds:>7.1f}{worst:>10.2%}")

print("""
On this problem everything works, and the local methods are the best value:
least_squares matches the global searches using ~25x fewer evaluations. That
is worth knowing -- local methods are not inherently worse, and reaching for a
global search by reflex wastes a lot of time.

What the easy case cannot show is when that breaks down. So:
""")

# --------------------------------------------------------------------------
# where the choice actually matters
# --------------------------------------------------------------------------
print("=" * 71)
print("A HARDER PROBLEM -- the shipped two-module example")
print("=" * 71)
print("""7 parameters, 4% noise, two coupled models, and modules that pull
against each other. The generating parameters score 0.0469; that is the floor.
""")

from pathlib import Path

import pymodest
from pymodest import load_config

repo = Path(pymodest.__file__).resolve().parents[2]
example = repo / "examples" / "two_module_pathway" / "config.toml"

if example.is_file():
    hard = load_config(example)
    print(f"{'backend':<24}{'kind':<8}{'cost':>12}{'evals':>9}{'sec':>7}")
    print("-" * 60)
    for name, options, kind in [
        ("differential_evolution", {"maxiter": 40, "popsize": 12}, "global"),
        ("scatter_search", {"maxiter": 25, "refset_size": 10, "max_nfev": 2500}, "global"),
        ("least_squares", {"max_nfev": 400}, "local"),
        ("minimize", {"method": "Nelder-Mead", "maxiter": 400}, "local"),
    ]:
        label = f"{name}/{options['method']}" if name == "minimize" else name
        cfg = replace(hard, fitting=replace(hard.fitting,
                                            optimizer=OptimizerSpec(name, options)))
        r = fit(cfg)
        print(f"{label:<24}{kind:<8}{r.cost:>12.4g}{r.n_evaluations:>9}{r.seconds:>7.1f}")
    print("""
Here the difference is not subtle: the global searches reach the noise floor,
the local ones sit about a thousand times above it. They never leave the region
the initial values put them in, and no amount of extra budget rescues them --
the problem is the starting point, not the iteration count.

The practical rule: search globally when you do not already know roughly where
the answer is, then refine locally. `[fitting.refine]` does exactly that --
a global optimizer for the first loop, a local one thereafter.""")
else:
    print(f"(example not found at {example}; skipping)")

# --------------------------------------------------------------------------
# adding a backend
# --------------------------------------------------------------------------
print("=" * 71)
print("ADDING YOUR OWN")
print("=" * 71)


@register("random_search", "rs")
def random_search(objective, x0, bounds, rng, n_samples: int = 400, **_):
    """Uniform random sampling -- the simplest thing that could possibly work.

    A backend receives the module objective and returns an OptimizerResult.
    `objective(x)` is the scalar cost over that module's free parameters in
    search space; `objective.residuals(x)` gives the residual vector instead.
    """
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    points = lower + rng.random((n_samples, lower.size)) * (upper - lower)
    costs = np.array([objective(p) for p in points])
    best = int(np.argmin(costs))
    return OptimizerResult(x=points[best], fun=float(costs[best]), nfev=n_samples)


print("registered:", "random_search" in optimizers.available())
config = replace(base, fitting=replace(
    base.fitting, optimizer=OptimizerSpec("random_search", {"n_samples": 600})))
result = fit(config)
print(f"random_search -> {report(result)}")
print("\nCrude, as expected -- but it plugs in exactly like the built-ins, and")
print("can be named from a TOML config as [fitting.optimizer] name = ...")
