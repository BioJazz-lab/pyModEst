"""Simulation wrapper and experiment-data loading."""

from __future__ import annotations

import numpy as np
import pytest

from pymodest.config import DatasetSpec, ModelSpec
from pymodest.data import DataError, Measurement, load_dataset
from pymodest.model import ModelError, SimulationModel

from conftest import TIMES, TRUE_PARAMETERS


# -- model ------------------------------------------------------------------

def test_model_exposes_species_and_parameters(toy_model):
    assert set(toy_model.floating_species) == {"S", "A", "B"}
    assert set(toy_model.global_parameters) >= {"Vmax1", "Km1", "k2", "k3"}
    assert toy_model.has_variable("A")
    assert toy_model.has_variable("Total")  # a declared observable
    assert not toy_model.has_variable("nonexistent")


def test_bad_antimony_raises_a_readable_error():
    with pytest.raises(ModelError, match="could not parse"):
        SimulationModel(ModelSpec(id="broken", antimony="model x\n this is not antimony %%%"))


def test_parameters_change_the_trajectory(toy_model):
    slow = toy_model.simulate(TIMES, parameters={"Vmax1": 0.5}, initial_conditions={"S": 10.0})
    fast = toy_model.simulate(TIMES, parameters={"Vmax1": 5.0}, initial_conditions={"S": 10.0})
    assert fast["A"][1] > slow["A"][1]


def test_initial_conditions_are_applied_and_survive_parameter_writes(toy_model):
    """Regression: parameters and starting values must both take effect.

    Writing an ``init(...)`` value makes roadrunner re-initialise and drop the
    global parameters, which silently reverted every parameter write. Species
    are now set by current value after a reset instead; this guards the
    behaviour either way."""
    out = toy_model.simulate(
        TIMES, parameters={"Vmax1": 3.0}, initial_conditions={"S": 10.0, "A": 0.4}
    )
    assert out["A"][0] == pytest.approx(0.4)
    baseline = toy_model.simulate(
        TIMES, parameters={"Vmax1": 0.2}, initial_conditions={"S": 10.0, "A": 0.4}
    )
    assert out["A"][3] > baseline["A"][3]


def test_observables_are_computed(toy_model):
    out = toy_model.simulate(TIMES, parameters=TRUE_PARAMETERS, initial_conditions={"S": 10.0})
    assert np.allclose(out["Total"], out["A"] + out["B"])


def test_bad_observable_expression_is_reported(toy_model_spec):
    model = SimulationModel(
        ModelSpec(id="x", antimony=toy_model_spec.antimony, observables={"Bad": "A + Nope"})
    )
    with pytest.raises(ModelError, match="observable 'Bad'"):
        model.simulate(TIMES, initial_conditions={"S": 10.0})


def test_results_are_cached_by_inputs(toy_model_spec):
    model = SimulationModel(toy_model_spec)
    model.simulate(TIMES, parameters={"k2": 0.5})
    model.simulate(TIMES, parameters={"k2": 0.5})
    assert model.n_simulations == 1
    model.simulate(TIMES, parameters={"k2": 0.9})
    assert model.n_simulations == 2


def test_time_grid_not_starting_at_zero(toy_model):
    """The model always starts at t=0; the prefix point is trimmed off."""
    later = np.array([3.0, 6.0, 9.0])
    out = toy_model.simulate(later, parameters=TRUE_PARAMETERS, initial_conditions={"S": 10.0})
    assert out["time"].shape == later.shape
    assert np.allclose(out["time"], later)


def test_unknown_variable_request_is_rejected(toy_model):
    with pytest.raises(KeyError, match="no such variable"):
        toy_model.simulate(TIMES, variables=["ghost"])


def test_setting_an_unknown_name_is_rejected(toy_model):
    with pytest.raises(ModelError, match="cannot set"):
        toy_model.simulate(TIMES, parameters={"not_a_parameter": 1.0})


# -- data -------------------------------------------------------------------

def test_wide_csv_is_loaded(synthetic_csv):
    data = load_dataset(DatasetSpec(id="exp", model="toy", file=synthetic_csv))
    assert set(data.variables) == {"A", "B"}
    assert data.measurements["A"].n == len(TIMES)
    assert data.n_points == 2 * len(TIMES)


def test_sigma_columns_are_recognised_and_not_treated_as_variables(tmp_path):
    import pandas as pd

    path = tmp_path / "d.csv"
    pd.DataFrame(
        {"time": [0.0, 1.0], "A": [1.0, 2.0], "A_sigma": [0.1, 0.2]}
    ).to_csv(path, index=False)
    data = load_dataset(DatasetSpec(id="d", model="m", file=path))
    assert data.variables == ["A"]
    assert np.allclose(data.measurements["A"].sigma, [0.1, 0.2])


def test_long_format(tmp_path):
    import pandas as pd

    path = tmp_path / "d.csv"
    pd.DataFrame(
        {
            "time": [0.0, 1.0, 0.0, 1.0],
            "variable": ["A", "A", "B", "B"],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    ).to_csv(path, index=False)
    data = load_dataset(DatasetSpec(id="d", model="m", file=path, format="long"))
    assert set(data.variables) == {"A", "B"}
    assert np.allclose(data.measurements["B"].values, [3.0, 4.0])


def test_missing_values_are_dropped_per_variable(tmp_path):
    """Variables need not share a measurement schedule."""
    import pandas as pd

    path = tmp_path / "d.csv"
    pd.DataFrame(
        {"time": [0.0, 1.0, 2.0], "A": [1.0, np.nan, 3.0], "B": [1.0, 2.0, 3.0]}
    ).to_csv(path, index=False)
    data = load_dataset(DatasetSpec(id="d", model="m", file=path))
    assert data.measurements["A"].n == 2
    assert data.measurements["B"].n == 3
    assert np.allclose(data.time_grid(["A"]), [0.0, 2.0])


def test_rows_are_sorted_by_time(tmp_path):
    import pandas as pd

    path = tmp_path / "d.csv"
    pd.DataFrame({"time": [2.0, 0.0, 1.0], "A": [3.0, 1.0, 2.0]}).to_csv(path, index=False)
    data = load_dataset(DatasetSpec(id="d", model="m", file=path))
    assert np.allclose(data.measurements["A"].times, [0.0, 1.0, 2.0])
    assert np.allclose(data.measurements["A"].values, [1.0, 2.0, 3.0])


def test_missing_time_column_is_reported(tmp_path):
    import pandas as pd

    path = tmp_path / "d.csv"
    pd.DataFrame({"t": [0.0], "A": [1.0]}).to_csv(path, index=False)
    with pytest.raises(DataError, match="no time column"):
        load_dataset(DatasetSpec(id="d", model="m", file=path))


def test_inline_data_is_accepted():
    spec = DatasetSpec(
        id="d", model="m", inline={"time": [0.0, 1.0], "A": [1.0, 2.0]}
    )
    data = load_dataset(spec)
    assert data.variables == ["A"]


def test_negative_sigma_is_rejected():
    with pytest.raises(DataError, match="sigma values must be positive"):
        Measurement("A", np.array([0.0]), np.array([1.0]), np.array([-1.0]))


def test_select_skips_variables_the_dataset_does_not_have(synthetic_csv):
    data = load_dataset(DatasetSpec(id="exp", model="toy", file=synthetic_csv))
    assert [m.variable for m in data.select(["A", "ghost"])] == ["A"]


def test_conditions_do_not_leak_between_simulations(toy_model):
    """Each call starts from the SBML defaults, so nothing carries over.

    Datasets in one study routinely differ in their starting conditions, and
    the model instance is shared between them. Alternating two conditions must
    give the same answers as running each on its own.
    """
    hi = dict(parameters=TRUE_PARAMETERS, initial_conditions={"S": 10.0})
    lo = dict(parameters=TRUE_PARAMETERS, initial_conditions={"S": 2.0})
    alone_hi = toy_model.simulate(TIMES, use_cache=False, **hi)["A"]
    alone_lo = toy_model.simulate(TIMES, use_cache=False, **lo)["A"]
    assert not np.allclose(alone_hi, alone_lo)
    for _ in range(3):
        assert np.allclose(toy_model.simulate(TIMES, use_cache=False, **hi)["A"], alone_hi)
        assert np.allclose(toy_model.simulate(TIMES, use_cache=False, **lo)["A"], alone_lo)


def test_a_condition_set_once_does_not_persist(toy_model):
    """A parameter set for one dataset must not survive into the next."""
    with_change = toy_model.simulate(
        TIMES, parameters={**TRUE_PARAMETERS, "k3": 2.5},
        initial_conditions={"S": 10.0}, use_cache=False,
    )["B"]
    without = toy_model.simulate(
        TIMES, parameters=TRUE_PARAMETERS, initial_conditions={"S": 10.0}, use_cache=False,
    )["B"]
    baseline = toy_model.simulate(
        TIMES, parameters=TRUE_PARAMETERS, initial_conditions={"S": 10.0}, use_cache=False,
    )["B"]
    assert not np.allclose(with_change, without)
    assert np.allclose(without, baseline)
