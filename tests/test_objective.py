"""Residual assembly, scaling, and the module objective's parameter transforms."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pymodest.config import ObjectiveSpec, ParameterSpec
from pymodest.data import Measurement
from pymodest.objective import Problem, ProblemError, _scale_residuals

from conftest import TRUE_PARAMETERS


# -- residual scaling -------------------------------------------------------

MEASURED = Measurement(
    "A", np.array([0.0, 1.0]), np.array([1.0, 100.0]), np.array([0.5, 5.0])
)
SIMULATED = np.array([2.0, 110.0])


def test_absolute_scaling_is_the_plain_difference():
    r = _scale_residuals(SIMULATED, MEASURED, ObjectiveSpec(scaling="absolute"))
    assert np.allclose(r, [1.0, 10.0])


def test_relative_scaling_equalises_variables_of_different_magnitude():
    r = _scale_residuals(SIMULATED, MEASURED, ObjectiveSpec(scaling="relative", epsilon=1e-9))
    assert np.allclose(r, [1.0, 0.1])


def test_sigma_scaling_uses_the_measurement_error():
    r = _scale_residuals(SIMULATED, MEASURED, ObjectiveSpec(scaling="sigma"))
    assert np.allclose(r, [2.0, 2.0])


def test_sigma_scaling_falls_back_when_no_error_was_measured():
    bare = Measurement("A", MEASURED.times, MEASURED.values)
    r = _scale_residuals(SIMULATED, bare, ObjectiveSpec(scaling="sigma", default_sigma=2.0))
    assert np.allclose(r, [0.5, 5.0])


def test_max_normalized_scaling_divides_by_the_peak():
    r = _scale_residuals(SIMULATED, MEASURED, ObjectiveSpec(scaling="max_normalized"))
    assert np.allclose(r, [0.01, 0.1])


# -- problem assembly -------------------------------------------------------

def test_cost_is_zero_at_the_generating_parameters(study):
    problem = Problem(study)
    problem.set_values(TRUE_PARAMETERS)
    assert problem.total_cost() == pytest.approx(0.0, abs=1e-10)


def test_cost_is_positive_away_from_the_truth(study):
    problem = Problem(study)
    problem.set_values({**TRUE_PARAMETERS, "k2": 3.0})
    assert problem.total_cost() > 1e-6


def test_a_module_only_sees_its_own_variables(study):
    """Perturbing a downstream parameter must not move the upstream cost."""
    problem = Problem(study)
    problem.set_values(TRUE_PARAMETERS)
    upstream = study.module("upstream")
    before = problem.module_cost(upstream)
    problem.set_values({"k3": 4.0})
    assert problem.module_cost(upstream) == pytest.approx(before)
    assert problem.module_cost(study.module("downstream")) > before


def test_residual_blocks_are_labelled(study):
    problem = Problem(study)
    blocks = problem.residual_blocks(study.module("upstream"))
    assert [b.variable for b in blocks] == ["A"]
    assert blocks[0].dataset == "exp"
    assert blocks[0].residuals.shape == blocks[0].observed.shape


def test_dataset_weight_scales_the_cost(study):
    problem = Problem(study)
    problem.set_values({**TRUE_PARAMETERS, "Vmax1": 1.0})
    plain = problem.module_cost(study.module("upstream"))

    heavier = replace(study, datasets=[replace(study.datasets[0], weight=4.0)])
    weighted = Problem(heavier)
    weighted.set_values({**TRUE_PARAMETERS, "Vmax1": 1.0})
    assert weighted.module_cost(heavier.module("upstream")) == pytest.approx(4.0 * plain)


def test_unknown_fitted_parameter_is_caught_at_construction(study):
    from pymodest.config import ModuleSpec

    bad = ModuleSpec(
        id="upstream",
        parameters=[ParameterSpec("not_in_model", 0.1, 1.0)],
        variables=["A"],
    )
    broken = replace(study, modules=[bad, study.modules[1]])
    with pytest.raises(ProblemError, match="not a settable parameter"):
        Problem(broken)


def test_unmeasured_module_variable_is_caught(study):
    from pymodest.config import ModuleSpec

    bad = ModuleSpec(
        id="upstream",
        parameters=[ParameterSpec("Vmax1", 0.1, 10.0)],
        variables=["never_measured"],
    )
    broken = replace(study, modules=[bad, study.modules[1]])
    with pytest.raises(ProblemError, match="not measured"):
        Problem(broken)


def test_parameters_are_filtered_to_the_models_that_have_them(study):
    """A parameter absent from a model is simply not written to it."""
    problem = Problem(study)
    model = problem.models["toy"]
    subset = problem.parameters_for(model, {**TRUE_PARAMETERS, "Ki": 2.0})
    assert "Ki" not in subset
    assert subset["Vmax1"] == TRUE_PARAMETERS["Vmax1"]


# -- module objective -------------------------------------------------------

def test_objective_round_trips_through_search_space(study):
    problem = Problem(study)
    objective = problem.objective_for(study.module("upstream"))
    assert objective.names == ["Vmax1", "Km1"]
    x = objective.to_search({"Vmax1": 2.0, "Km1": 4.0})
    assert objective.to_model(x)["Vmax1"] == pytest.approx(2.0)
    assert objective.to_model(x)["Km1"] == pytest.approx(4.0)


def test_objective_leaves_other_modules_untouched(study):
    problem = Problem(study)
    problem.set_values(TRUE_PARAMETERS)
    objective = problem.objective_for(study.module("upstream"))
    objective(objective.to_search({"Vmax1": 9.0, "Km1": 9.0}))
    # evaluating a trial point must not commit it
    assert problem.values["Vmax1"] == pytest.approx(TRUE_PARAMETERS["Vmax1"])
    assert problem.values["k3"] == pytest.approx(TRUE_PARAMETERS["k3"])


def test_commit_writes_back(study):
    problem = Problem(study)
    objective = problem.objective_for(study.module("upstream"))
    objective.commit(objective.to_search({"Vmax1": 3.0, "Km1": 5.0}))
    assert problem.values["Vmax1"] == pytest.approx(3.0)


def test_objective_tracks_the_best_point_it_saw(study):
    problem = Problem(study)
    objective = problem.objective_for(study.module("upstream"))
    good = objective.to_search({"Vmax1": TRUE_PARAMETERS["Vmax1"], "Km1": TRUE_PARAMETERS["Km1"]})
    bad = objective.to_search({"Vmax1": 15.0, "Km1": 0.2})
    objective(bad)
    objective(good)
    objective(bad)
    assert np.allclose(objective.best_x, good)


def test_residual_vector_matches_the_scalar_cost(study):
    problem = Problem(study)
    objective = problem.objective_for(study.module("upstream"))
    x = objective.to_search({"Vmax1": 1.5, "Km1": 2.0})
    residuals = objective.residuals(x)
    assert np.sum(residuals**2) == pytest.approx(objective(x), rel=1e-9)


def test_infeasible_parameters_are_scored_not_raised(study):
    """A parameter set the integrator cannot handle must not abort the fit."""
    problem = Problem(study)
    cost = problem.module_cost(study.module("upstream"), {"Vmax1": 1e9, "Km1": 1e-12})
    assert np.isfinite(cost)
