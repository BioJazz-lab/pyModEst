"""A complete study, start to finish, in one file.

No configuration file and no data files: the model is written inline, the
"measurements" are simulated from known parameters, and the fit has to recover
them. Run it with:

    uv run python docs/examples/01_first_fit.py
"""

from __future__ import annotations

import numpy as np

from pymodest import fit
from pymodest.config import (
    DatasetSpec, FittingSpec, ModelSpec, ModuleSpec, ObjectiveSpec,
    OptimizerSpec, ParameterSpec, SimulationSpec, StudyConfig,
)
from pymodest.model import SimulationModel

# --------------------------------------------------------------------------
# 1. the model
# --------------------------------------------------------------------------
# A three-step pathway. Uptake of S saturates; A and B turn over linearly.
ANTIMONY = """
model toy
  J1: S -> A;  Vmax1 * S / (Km1 + S);
  J2: A -> B;  k2 * A;
  J3: B -> ;   k3 * B;

  S = 10; A = 0; B = 0;
  Vmax1 = 1; Km1 = 1; k2 = 1; k3 = 1;
end
"""

TRUE = {"Vmax1": 2.0, "Km1": 4.0, "k2": 0.6, "k3": 0.3}
TIMES = [0.0, 0.5, 1.0, 2.0, 4.0, 7.0, 10.0, 15.0]

# --------------------------------------------------------------------------
# 2. the measurements
# --------------------------------------------------------------------------
# Normally these come from a CSV. Here we simulate them at the true parameters
# so that the fit has a known right answer to find.
model = ModelSpec(id="toy", antimony=ANTIMONY)
truth = SimulationModel(model, SimulationSpec()).simulate(
    TIMES, parameters=TRUE, initial_conditions={"S": 10.0}
)
measurements = {
    "time": list(TIMES),
    "A": [float(v) for v in truth["A"]],
    "B": [float(v) for v in truth["B"]],
}

# --------------------------------------------------------------------------
# 3. the modules -- the heart of it
# --------------------------------------------------------------------------
# Uptake parameters are fitted against A; turnover parameters against B. Each
# module is optimized while the other module's parameters stay fixed.
upstream = ModuleSpec(
    id="upstream",
    variables=["A"],
    parameters=[
        ParameterSpec("Vmax1", lower=0.1, upper=20.0, init=1.0, scale="log"),
        ParameterSpec("Km1", lower=0.1, upper=40.0, init=1.0, scale="log"),
    ],
)
downstream = ModuleSpec(
    id="downstream",
    variables=["B"],
    parameters=[
        ParameterSpec("k2", lower=0.01, upper=5.0, init=1.0, scale="log"),
        ParameterSpec("k3", lower=0.01, upper=5.0, init=1.0, scale="log"),
    ],
)

# --------------------------------------------------------------------------
# 4. the study
# --------------------------------------------------------------------------
config = StudyConfig(
    name="first-fit",
    models=[model],
    datasets=[
        DatasetSpec(id="exp", model="toy", inline=measurements,
                    initial_conditions={"S": 10.0}),
    ],
    modules=[upstream, downstream],
    fitting=FittingSpec(
        max_loops=6,
        tol=1e-4,
        patience=2,
        seed=3,
        optimizer=OptimizerSpec("differential_evolution", {"maxiter": 30, "popsize": 10}),
        objective=ObjectiveSpec(scaling="relative", epsilon=1e-3),
    ),
)
config.validate()   # catches typos and mis-wired modules before any fitting

# --------------------------------------------------------------------------
# 5. fit, and check
# --------------------------------------------------------------------------
result = fit(config)

print(f"cost {result.initial_cost:.4g} -> {result.cost:.4g} "
      f"in {result.n_loops} loops ({result.n_evaluations} evaluations)")
print(f"stopped because: {result.stop_reason}\n")

print(f"{'parameter':<10}{'true':>8}{'estimate':>11}{'error':>9}")
for name in sorted(TRUE):
    got, want = result.parameters[name], TRUE[name]
    print(f"{name:<10}{want:>8.3f}{got:>11.4f}{abs(got - want) / want:>8.1%}")

worst = max(abs(result.parameters[n] - v) / v for n, v in TRUE.items())
print(f"\nworst relative error: {worst:.2%}")
