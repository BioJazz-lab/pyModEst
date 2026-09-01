"""Controlling the loop: order, acceptance, stopping, and holding parameters.

The knobs in ``[fitting]``, each shown doing something visible.

    uv run python docs/examples/05_loop_control.py
"""

from __future__ import annotations

import logging
from dataclasses import replace

logging.disable(logging.INFO)

from pymodest import fit
from pymodest.config import ModuleSpec, OptimizerSpec, ParameterSpec
from pymodest.estimator import ModularEstimator

from _toy import TRUE, report, single_model_study

base = single_model_study()

# --------------------------------------------------------------------------
print("=" * 70)
print("MODULE ORDER -- which module is fitted first, and does it alternate")
print("=" * 70)
for setting in ("as_listed", "round_robin_reversed", "random", ["downstream", "upstream"]):
    config = replace(base, fitting=replace(base.fitting, module_order=setting))
    estimator = ModularEstimator(config)
    orders = [" -> ".join(m.id for m in estimator.module_order(loop)) for loop in (1, 2)]
    label = setting if isinstance(setting, str) else "explicit list"
    print(f"  {label:<22} loop 1: {orders[0]:<24} loop 2: {orders[1]}")

# --------------------------------------------------------------------------
print()
print("=" * 70)
print("STOPPING -- max_loops, and convergence on tol / atol / patience")
print("=" * 70)

capped = fit(replace(base, fitting=replace(base.fitting, max_loops=2,
                                           tol=0.0, patience=99)))
print(f"  max_loops=2         {capped.n_loops} loops -- {capped.stop_reason}")

converged = fit(replace(base, fitting=replace(base.fitting, max_loops=50)))
print(f"  max_loops=50        {converged.n_loops} loops -- {converged.stop_reason}")
print("""
  Progress is measured against the best cost so far, not the previous loop,
  and needs to clear `tol` relatively OR `atol` absolutely. The absolute floor
  matters: a cost converging towards zero keeps producing large relative gains
  forever, so a relative test alone would never stop.""")

# --------------------------------------------------------------------------
print()
print("=" * 70)
print("ACCEPTANCE -- what to do when helping one module hurts another")
print("=" * 70)
print("  This toy is decoupled -- A does not depend on the downstream")
print("  parameters -- so the two policies cannot differ:\n")
for policy in ("module", "total"):
    result = fit(replace(base, fitting=replace(base.fitting, accept=policy)))
    rises = sum(1 for s in result.steps if s.total_after > s.total_before + 1e-12)
    print(f"    accept={policy:<8} cost {result.cost:.3g}   "
          f"steps that raised the total: {rises}")

# The policy only matters when modules interact. The shipped example has a
# feedback strain, so the downstream parameters change the upstream variables.
from pathlib import Path

import pymodest
from pymodest import load_config

example = (Path(pymodest.__file__).resolve().parents[2]
           / "examples" / "two_module_pathway" / "config.toml")
if example.is_file():
    print("\n  The shipped example IS coupled -- C feeds back on uptake:\n")
    hard = load_config(example)
    for policy in ("module", "total"):
        r = fit(replace(hard, fitting=replace(hard.fitting, accept=policy)))
        rises = sum(1 for s in r.steps if s.total_after > s.total_before + 1e-12)
        print(f"    accept={policy:<8} cost {r.cost:.4g}   {r.n_loops} loops   "
              f"steps that raised the total: {rises}")
    print("""
  'module' (default) keeps any step that improves its own module, and tolerates
  the total rising on the way. 'total' additionally demands the overall cost
  never rise -- which sounds safer and is much worse here, because the step
  that escapes the bad starting region temporarily hurts the other module.

  Monotone is not the same as good. The estimator already protects you by
  returning the best parameters seen across the whole run, not the last ones,
  so 'module' does not risk ending on a bad loop.""")

# --------------------------------------------------------------------------
print()
print("=" * 70)
print("HOLDING PARAMETERS -- fixed values are still applied, just not searched")
print("=" * 70)

pinned = ModuleSpec(
    id="downstream",
    variables=["B"],
    parameters=[
        ParameterSpec("k2", 0.01, 5.0, init=TRUE["k2"], scale="log", fixed=True),
        ParameterSpec("k3", 0.01, 5.0, init=1.0, scale="log"),
    ],
)
result = fit(replace(base, modules=[base.modules[0], pinned]))
print(f"  k2 pinned at {TRUE['k2']}  -> estimate {result.parameters['k2']:.4f} (unchanged)")
print(f"  k3 free               -> estimate {result.parameters['k3']:.4f} "
      f"(true {TRUE['k3']})")
print("  A module whose parameters are all fixed is skipped entirely.")

# --------------------------------------------------------------------------
print()
print("=" * 70)
print("WATCHING IT RUN -- a callback fires after every module fit")
print("=" * 70)

seen = []
fit(replace(base, fitting=replace(base.fitting, max_loops=3, tol=0.0, patience=99)),
    callback=seen.append)
print(f"  {'loop':<6}{'module':<14}{'cost before':>13}{'cost after':>13}{'kept':>7}")
for step in seen[:6]:
    print(f"  {step.loop:<6}{step.module:<14}{step.cost_before:>13.4g}"
          f"{step.cost_after:>13.4g}{str(step.accepted):>7}")
print("\n  The same information lands in result.history() as a DataFrame, and in")
print("  history.csv when you call result.save(directory).")
