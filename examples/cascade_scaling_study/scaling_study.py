"""How does the modular-vs-global comparison change as the model grows?

Sweeps the parametrized cascade family (cascade_family.py) over model size
(number of tiers) and three module schemes:

  global   -- one module, every parameter (the baseline)
  pertier  -- one module per tier: modularization SCALES with model size,
              so per-module dimensionality stays constant
  fixedK   -- always 3 contiguous modules: modularization is FIXED, so
              per-module dimensionality grows with model size

For each (size, scheme, seed) combination, fits and records final cost
relative to the noise floor, function evaluations, wall time, loop count,
number of modules and total parameter count. Data (with a fixed seed) is
generated once per size and shared across every scheme/seed at that size, so
the only things that vary between runs are the module partition and the
optimizer's random seed.

Unlike the other examples in this repo, nothing here is a static
Antimony/TOML file: cascade_family.py builds the model and every StudyConfig
directly as in-memory pymodest objects, which is what makes sweeping 4
sizes x 3 schemes x 3 seeds (36 fits, ~20-25 minutes) tractable -- and is
itself worth reading as an example of constructing a study programmatically
instead of from a config file.

    uv run python scaling_study.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cascade_family import build_config, generate_dataset, true_parameters  # noqa: E402
from pymodest.estimator import ModularEstimator  # noqa: E402
from pymodest.objective import Problem  # noqa: E402

OUT = HERE / "results"

SIZES = [3, 6, 9, 12]
SCHEMES = ["global", "pertier", "fixedK"]
SEEDS = [1, 2, 3]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = []

    for n_tiers in SIZES:
        data = generate_dataset(n_tiers)
        truth = true_parameters(n_tiers)

        for scheme in SCHEMES:
            for seed in SEEDS:
                cfg = build_config(n_tiers, scheme, seed=seed, dataset_dict=data)

                floor_problem = Problem(cfg)
                floor_problem.set_values(truth)
                noise_floor = floor_problem.total_cost()

                estimator = ModularEstimator(cfg)
                started = time.perf_counter()
                result = estimator.run()
                elapsed = time.perf_counter() - started

                n_params = len(cfg.all_parameters())
                n_modules = len(cfg.modules)
                print(
                    f"n_tiers={n_tiers:2d} ({n_params:2d} params) scheme={scheme:8s} "
                    f"seed={seed}  modules={n_modules:2d}  "
                    f"cost {result.initial_cost:.4g}->{result.cost:.4g} "
                    f"(floor {noise_floor:.4g}, ratio {result.cost / noise_floor:.3f})  "
                    f"{result.n_loops} loop(s)  {result.n_evaluations} evals  {elapsed:.1f}s"
                )

                runs.append(dict(
                    n_tiers=n_tiers, n_params=n_params, scheme=scheme, seed=seed,
                    n_modules=n_modules, n_loops=result.n_loops,
                    n_evaluations=result.n_evaluations, seconds=elapsed,
                    initial_cost=result.initial_cost, final_cost=result.cost,
                    noise_floor=noise_floor, cost_ratio=result.cost / noise_floor,
                    converged=result.converged, stop_reason=result.stop_reason,
                ))
                pd.DataFrame(runs).to_csv(OUT / "runs.csv", index=False)  # write incrementally

    print(f"\nwrote {OUT}/runs.csv ({len(runs)} runs)")


if __name__ == "__main__":
    main()
