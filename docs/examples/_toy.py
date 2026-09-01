"""Shared setup for the examples: one small pathway, two model variants.

Example 01 is deliberately self-contained. The rest import from here so that
each one stays focused on the capability it demonstrates.

    S --J1--> A --J2--> B --J3--> out

``wt`` is the plain pathway. ``feedback`` adds inhibition of uptake by B, and
with it one extra parameter, ``Ki``, that exists in no other model.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pymodest.config import (
    DatasetSpec, FittingSpec, ModelSpec, ModuleSpec, ObjectiveSpec,
    OptimizerSpec, ParameterSpec, SimulationSpec, StudyConfig,
)
from pymodest.model import SimulationModel

WT = """
model wt
  J1: S -> A;  Vmax1 * S / (Km1 + S);
  J2: A -> B;  k2 * A;
  J3: B -> ;   k3 * B;
  S = 10; A = 0; B = 0;
  Vmax1 = 1; Km1 = 1; k2 = 1; k3 = 1;
end
"""

FEEDBACK = """
model feedback
  J1: S -> A;  Vmax1 * S / (Km1 + S) * 1 / (1 + B / Ki);
  J2: A -> B;  k2 * A;
  J3: B -> ;   k3 * B;
  S = 10; A = 0; B = 0;
  Vmax1 = 1; Km1 = 1; k2 = 1; k3 = 1; Ki = 1;
end
"""

TRUE = {"Vmax1": 2.0, "Km1": 4.0, "k2": 0.6, "k3": 0.3, "Ki": 1.5}
TIMES = [0.0, 0.5, 1.0, 2.0, 4.0, 7.0, 10.0, 15.0]


def simulate_truth(antimony: str, s0: float = 10.0,
                   noise: float = 0.0, seed: int = 0) -> Dict[str, List[float]]:
    """Measurements of A and B generated at the true parameters."""
    import numpy as np

    spec = ModelSpec(id="gen", antimony=antimony)
    model = SimulationModel(spec, SimulationSpec())
    params = {k: v for k, v in TRUE.items() if model.has_parameter(k)}
    trace = model.simulate(TIMES, parameters=params,
                           initial_conditions={"S": s0}, use_cache=False)
    rng = np.random.default_rng(seed)
    out = {"time": list(TIMES)}
    for name in ("A", "B"):
        values = np.asarray(trace[name], dtype=float)
        if noise:
            values = np.maximum(values + rng.normal(0.0, noise * np.maximum(values, 1e-3)), 0.0)
        out[name] = [float(v) for v in values]
    return out


def modules(with_ki: bool = False) -> List[ModuleSpec]:
    """Two modules: uptake scored on A, turnover scored on B."""
    downstream = [
        ParameterSpec("k2", lower=0.01, upper=5.0, init=1.0, scale="log"),
        ParameterSpec("k3", lower=0.01, upper=5.0, init=1.0, scale="log"),
    ]
    if with_ki:
        downstream.append(ParameterSpec("Ki", lower=0.05, upper=50.0, init=1.0, scale="log"))
    return [
        ModuleSpec(
            id="upstream",
            variables=["A"],
            parameters=[
                ParameterSpec("Vmax1", lower=0.1, upper=20.0, init=1.0, scale="log"),
                ParameterSpec("Km1", lower=0.1, upper=40.0, init=1.0, scale="log"),
            ],
        ),
        ModuleSpec(id="downstream", variables=["B"], parameters=downstream),
    ]


def fitting(**overrides) -> FittingSpec:
    """Sensible, fast defaults for a demonstration."""
    settings = dict(
        max_loops=6, tol=1e-4, patience=2, seed=3,
        optimizer=OptimizerSpec("differential_evolution", {"maxiter": 30, "popsize": 10}),
        objective=ObjectiveSpec(scaling="relative", epsilon=1e-3),
    )
    settings.update(overrides)
    return FittingSpec(**settings)


def single_model_study(noise: float = 0.0, **fit_overrides) -> StudyConfig:
    """One model, one dataset, two modules -- the baseline study."""
    config = StudyConfig(
        name="toy",
        models=[ModelSpec(id="toy", antimony=WT)],
        datasets=[DatasetSpec(id="exp", model="toy",
                              inline=simulate_truth(WT, noise=noise),
                              initial_conditions={"S": 10.0})],
        modules=modules(),
        fitting=fitting(**fit_overrides),
    )
    config.validate()
    return config


def report(result, truth: Optional[Dict[str, float]] = None) -> str:
    """One line summarising a fit."""
    truth = truth or TRUE
    shared = [n for n in result.parameters if n in truth]
    worst = max(abs(result.parameters[n] - truth[n]) / truth[n] for n in shared)
    return (f"cost {result.cost:.3g} | {result.n_loops} loops | "
            f"{result.n_evaluations} evals | worst error {worst:.2%}")
