"""Shared fixtures: a tiny two-module study built entirely in memory."""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest

from pymodest.config import (
    DatasetSpec, FittingSpec, ModelSpec, ModuleSpec, ObjectiveSpec,
    OptimizerSpec, ParameterSpec, SimulationSpec, StudyConfig,
)
from pymodest.model import SimulationModel

TOY_ANTIMONY = textwrap.dedent(
    """
    model toy
      J1: S -> A;  Vmax1 * S / (Km1 + S);
      J2: A -> B;  k2 * A;
      J3: B -> ;   k3 * B;
      S = 10; A = 0; B = 0;
      Vmax1 = 1; Km1 = 1; k2 = 1; k3 = 1;
    end
    """
).strip()

TRUE_PARAMETERS = {"Vmax1": 2.0, "Km1": 4.0, "k2": 0.6, "k3": 0.3}
TIMES = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 7.0, 10.0, 15.0])


@pytest.fixture(scope="session")
def toy_model_spec() -> ModelSpec:
    return ModelSpec(id="toy", antimony=TOY_ANTIMONY, observables={"Total": "A + B"})


@pytest.fixture(scope="session")
def toy_model(toy_model_spec) -> SimulationModel:
    return SimulationModel(toy_model_spec, SimulationSpec())


@pytest.fixture(scope="session")
def synthetic_csv(tmp_path_factory, toy_model) -> Path:
    """Noise-free measurements of A and B generated at the true parameters."""
    import pandas as pd

    trace = toy_model.simulate(
        TIMES, parameters=TRUE_PARAMETERS, initial_conditions={"S": 10.0}, use_cache=False
    )
    frame = pd.DataFrame({"time": TIMES, "A": trace["A"], "B": trace["B"]})
    path = tmp_path_factory.mktemp("data") / "exp.csv"
    frame.to_csv(path, index=False)
    return path


@pytest.fixture
def study(toy_model_spec, synthetic_csv, tmp_path) -> StudyConfig:
    """A two-module study over the toy model, ready to fit."""
    upstream = ModuleSpec(
        id="upstream",
        parameters=[
            ParameterSpec("Vmax1", 0.1, 20.0, init=1.0, scale="log"),
            ParameterSpec("Km1", 0.1, 40.0, init=1.0, scale="log"),
        ],
        variables=["A"],
    )
    downstream = ModuleSpec(
        id="downstream",
        parameters=[
            ParameterSpec("k2", 0.01, 5.0, init=1.0, scale="log"),
            ParameterSpec("k3", 0.01, 5.0, init=1.0, scale="log"),
        ],
        variables=["B"],
    )
    dataset = DatasetSpec(
        id="exp", model="toy", file=synthetic_csv, initial_conditions={"S": 10.0}
    )
    fitting = FittingSpec(
        max_loops=6,
        tol=1e-4,
        patience=2,
        seed=3,
        optimizer=OptimizerSpec("differential_evolution", {"maxiter": 30, "popsize": 10}),
        objective=ObjectiveSpec(scaling="relative", epsilon=1e-3),
    )
    config = StudyConfig(
        models=[toy_model_spec],
        datasets=[dataset],
        modules=[upstream, downstream],
        fitting=fitting,
        name="toy-study",
        output_dir=tmp_path / "results",
        base_dir=tmp_path,
    )
    config.validate()
    return config
