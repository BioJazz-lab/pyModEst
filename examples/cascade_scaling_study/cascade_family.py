"""A parametrized family of N-tier feed-forward cascades, for studying how
the modular-vs-global comparison scales with model size and modularization.

Generalizes the fixed 6-tier `models/cascade.ant` design: tier 1 is a
receptor-like fast activation driven by a stimulus L; every tier i >= 2 is a
Hill-activated species driven by tier i-1, with a characteristic timescale
`RATIO`-fold slower than the tier before it. Every tier is normalized to
reach steady state ~1 when its input saturates, so identifiability
difficulty stays uniform across tiers and the only things that change as N
grows are (a) total parameter count and (b) how many well-separated
timescales the system spans -- not per-tier idiosyncrasies.

Everything is built as in-memory pymodest objects (ModelSpec with inline
Antimony text, DatasetSpec with inline data) -- no files touch disk, which is
what makes sweeping many (size, module-scheme, seed) combinations tractable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from pymodest.config import (
    DatasetSpec, FittingSpec, ModelSpec, ModuleSpec, ObjectiveSpec,
    OptimizerSpec, ParameterSpec, SimulationSpec, StudyConfig,
)
from pymodest.model import SimulationModel

TAU1 = 0.3      # tier 1 (receptor) characteristic timescale
RATIO = 2.5     # timescale ratio between successive tiers
DOSE = 1.5      # single stimulus dose L
NOISE = 0.04    # proportional noise
DATA_SEED = 20260903  # fixed, so every (scheme, optimizer-seed) run at a
                       # given size sees exactly the same synthetic data


def tier_names(n_tiers: int) -> List[str]:
    return [f"X{i}" for i in range(1, n_tiers + 1)]


def true_parameters(n_tiers: int) -> Dict[str, float]:
    params = {"k1on": 3.0, "k1off": 1.5}
    for i in range(2, n_tiers + 1):
        tau_i = TAU1 * RATIO ** (i - 1)
        d_i = 1.0 / tau_i
        params[f"d_{i}"] = d_i
        params[f"aM_{i}"] = d_i * 1.0
        params[f"a0_{i}"] = d_i * 0.02
        params[f"KM_{i}"] = 0.3
        params[f"n_{i}"] = 2.0
    return params


def build_antimony(n_tiers: int) -> str:
    lines = [f"model cascade_n{n_tiers}", "  L = 1.0;"]
    lines.append("  JX1on:  -> X1;  k1on * L * (1 - X1);")
    lines.append("  JX1off: X1 -> ; k1off * X1;")
    for i in range(2, n_tiers + 1):
        prev = f"X{i - 1}"
        lines.append(
            f"  JX{i}prod: -> X{i};  a0_{i} + aM_{i} * {prev}^n_{i} / "
            f"(KM_{i}^n_{i} + {prev}^n_{i});"
        )
        lines.append(f"  JX{i}deg:  X{i} -> ; d_{i} * X{i};")
    lines.append("  " + "  ".join(f"X{i} = 0;" for i in range(1, n_tiers + 1)))
    true = true_parameters(n_tiers)
    for name, value in true.items():
        lines.append(f"  {name} = {value};")
    lines.append("end")
    return "\n".join(lines)


def time_grid(n_tiers: int, n_points: int = 24) -> np.ndarray:
    tau_last = TAU1 * RATIO ** (n_tiers - 1)
    t_end = tau_last * 8.0
    t_min = TAU1 * 0.05
    return np.concatenate(([0.0], np.geomspace(t_min, t_end, n_points)))


def generate_dataset(n_tiers: int, data_seed: int = DATA_SEED) -> Dict[str, list]:
    """One inline dataset (dict of column -> list), ready for DatasetSpec(inline=...)."""
    antimony = build_antimony(n_tiers)
    model = SimulationModel(ModelSpec(id="cascade", antimony=antimony), SimulationSpec())
    true = true_parameters(n_tiers)
    times = time_grid(n_tiers)
    trace = model.simulate(times, parameters=true, conditions={"L": DOSE}, use_cache=False)

    rng = np.random.default_rng(data_seed)
    inline: Dict[str, list] = {"time": times.tolist()}
    for name in tier_names(n_tiers):
        clean = trace[name]
        peak = max(float(np.max(np.abs(clean))), 1e-6)
        sigma = NOISE * np.maximum(np.abs(clean), 0.05 * peak)
        observed = np.maximum(clean + rng.normal(0.0, sigma), 0.0)
        inline[name] = observed.tolist()
        inline[f"{name}_sigma"] = sigma.tolist()
    return inline


def parameter_specs_for_tier(i: int, true: Dict[str, float]) -> List[ParameterSpec]:
    names = ["k1on", "k1off"] if i == 1 else [f"a0_{i}", f"aM_{i}", f"KM_{i}", f"n_{i}", f"d_{i}"]
    specs = []
    for j, name in enumerate(names):
        value = true[name]
        if name.startswith("n_"):
            lower, upper = 1.1, 5.0
        else:
            lower, upper = value / 12.0, value * 12.0
        factor = 4.0 if (i + j) % 2 == 0 else 1.0 / 4.0
        init = max(lower * 1.05, min(upper * 0.95, value * factor))
        specs.append(ParameterSpec(name=name, lower=lower, upper=upper, init=init, scale="log"))
    return specs


def module_groups(n_tiers: int, scheme: str, fixed_k: int = 3) -> List[List[int]]:
    """Tiers (1-indexed) grouped into modules under one of three schemes.

    "global"  -- one module, every tier
    "pertier" -- one module per tier (modularization scales with size)
    "fixedK"  -- always `fixed_k` contiguous modules (modularization fixed)
    """
    tiers = list(range(1, n_tiers + 1))
    if scheme == "global":
        return [tiers]
    if scheme == "pertier":
        return [[t] for t in tiers]
    if scheme == "fixedK":
        if n_tiers % fixed_k != 0:
            raise ValueError(f"n_tiers={n_tiers} must be divisible by fixed_k={fixed_k}")
        size = n_tiers // fixed_k
        return [tiers[i * size:(i + 1) * size] for i in range(fixed_k)]
    raise ValueError(f"unknown scheme {scheme!r}")


def build_config(
    n_tiers: int,
    scheme: str,
    seed: int,
    dataset_dict: Dict[str, list],
    *,
    fixed_k: int = 3,
    popsize: int = 10,
    maxiter: int = 40,
    max_loops: int = 6,
    patience: int = 2,
) -> StudyConfig:
    antimony = build_antimony(n_tiers)
    true = true_parameters(n_tiers)
    model = ModelSpec(id="cascade", antimony=antimony, description=f"{n_tiers}-tier cascade")
    dataset = DatasetSpec(
        id="dose", model="cascade", inline=dataset_dict, format="wide",
        weight=1.0, conditions={"L": DOSE},
    )
    groups = module_groups(n_tiers, scheme, fixed_k=fixed_k)
    modules = []
    for gi, group in enumerate(groups):
        params: List[ParameterSpec] = []
        variables: List[str] = []
        for tier in group:
            params.extend(parameter_specs_for_tier(tier, true))
            variables.append(f"X{tier}")
        modules.append(ModuleSpec(id=f"m{gi}", parameters=params, variables=variables))

    fitting = FittingSpec(
        max_loops=max_loops, module_order="as_listed", tol=1e-3, patience=patience, seed=seed,
        optimizer=OptimizerSpec(
            name="differential_evolution",
            options=dict(maxiter=maxiter, popsize=popsize, tol=1e-8, polish=True),
        ),
        objective=ObjectiveSpec(scaling="max_normalized", aggregation="mean"),
        simulation=SimulationSpec(),
    )
    config = StudyConfig(
        models=[model], datasets=[dataset], modules=modules, fitting=fitting,
        name=f"cascade-n{n_tiers}-{scheme}-seed{seed}",
        output_dir=Path("results") / "scaling_study",
    )
    config.validate()
    return config
