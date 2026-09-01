"""The divide-and-conquer loop: ordering, acceptance, convergence, recovery."""

from __future__ import annotations

from dataclasses import replace

import pytest

from pymodest.config import OptimizerSpec
from pymodest.estimator import ModularEstimator, fit
from pymodest.objective import Problem

from conftest import TRUE_PARAMETERS


# -- module ordering --------------------------------------------------------

def test_default_order_follows_the_config(study):
    estimator = ModularEstimator(study)
    assert [m.id for m in estimator.module_order(1)] == ["upstream", "downstream"]


def test_explicit_order_is_honoured(study):
    cfg = replace(study, fitting=replace(study.fitting, module_order=["downstream", "upstream"]))
    assert [m.id for m in ModularEstimator(cfg).module_order(1)] == ["downstream", "upstream"]


def test_round_robin_reversed_alternates(study):
    cfg = replace(study, fitting=replace(study.fitting, module_order="round_robin_reversed"))
    estimator = ModularEstimator(cfg)
    assert [m.id for m in estimator.module_order(1)] == ["upstream", "downstream"]
    assert [m.id for m in estimator.module_order(2)] == ["downstream", "upstream"]


def test_random_order_covers_every_module(study):
    cfg = replace(study, fitting=replace(study.fitting, module_order="random"))
    estimator = ModularEstimator(cfg)
    for loop in range(1, 6):
        assert {m.id for m in estimator.module_order(loop)} == {"upstream", "downstream"}


# -- a single module fit ----------------------------------------------------

def test_fitting_one_module_only_moves_that_module_s_parameters(study):
    estimator = ModularEstimator(study)
    before = estimator.values
    step = estimator.fit_module(study.module("upstream"), loop=1)
    after = estimator.values
    assert step.module == "upstream"
    assert after["k2"] == pytest.approx(before["k2"])
    assert after["k3"] == pytest.approx(before["k3"])
    assert after["Vmax1"] != pytest.approx(before["Vmax1"])


def test_a_module_step_never_worsens_its_own_cost(study):
    estimator = ModularEstimator(study)
    for module in study.modules:
        step = estimator.fit_module(module, loop=1)
        assert step.cost_after <= step.cost_before + 1e-12


def test_a_step_that_cannot_improve_is_rejected(study):
    """Starting at the optimum, the fit must leave the parameters alone."""
    estimator = ModularEstimator(study)
    estimator.problem.set_values(TRUE_PARAMETERS)
    step = estimator.fit_module(study.module("upstream"), loop=1)
    assert step.cost_after == pytest.approx(step.cost_before, abs=1e-9)
    assert estimator.values["Vmax1"] == pytest.approx(TRUE_PARAMETERS["Vmax1"], rel=1e-3)


# -- the outer loop ---------------------------------------------------------

def test_the_loop_recovers_the_generating_parameters(study):
    """The headline behaviour: noise-free data, parameters recovered."""
    result = fit(study)
    for name, truth in TRUE_PARAMETERS.items():
        assert result.parameters[name] == pytest.approx(truth, rel=0.05), name
    assert result.cost < 1e-4
    assert result.cost < result.initial_cost


def test_the_result_holds_the_best_parameters_not_the_last(study):
    result = fit(study)
    problem = Problem(study)
    assert problem.total_cost(result.parameters) == pytest.approx(result.cost, rel=1e-6)
    for loop in result.loops:
        assert result.cost <= loop.total_cost + 1e-12


def test_max_loops_is_respected(study):
    cfg = replace(study, fitting=replace(study.fitting, max_loops=2, tol=0.0, patience=99))
    result = fit(cfg)
    assert result.n_loops == 2
    assert "max_loops" in result.stop_reason


def test_convergence_stops_early(study):
    result = fit(replace(study, fitting=replace(study.fitting, max_loops=20)))
    assert result.n_loops < 20
    assert result.converged


def test_history_records_every_module_fit(study):
    result = fit(study)
    history = result.history()
    assert len(history) == sum(len(loop.steps) for loop in result.loops)
    assert set(history["module"]) == {"upstream", "downstream"}
    assert "p:Vmax1" in history.columns
    summary = result.loop_summary()
    assert len(summary) == result.n_loops
    assert "cost:upstream" in summary.columns


def test_progress_callback_sees_each_step(study):
    seen = []
    fit(replace(study, fitting=replace(study.fitting, max_loops=2, tol=0.0, patience=99)),
        callback=seen.append)
    assert [s.module for s in seen[:2]] == ["upstream", "downstream"]
    assert all(s.loop in (1, 2) for s in seen)


@pytest.mark.parametrize("backend", ["differential_evolution", "particle_swarm", "scatter_search"])
def test_different_backends_reach_the_same_answer(study, backend):
    options = {
        "differential_evolution": {"maxiter": 40, "popsize": 12},
        "particle_swarm": {"maxiter": 40, "n_particles": 20},
        "scatter_search": {"maxiter": 20, "refset_size": 8, "max_nfev": 1500},
    }[backend]
    cfg = replace(
        study,
        fitting=replace(study.fitting, optimizer=OptimizerSpec(backend, options), max_loops=8),
    )
    result = fit(cfg)
    for name, truth in TRUE_PARAMETERS.items():
        assert result.parameters[name] == pytest.approx(truth, rel=0.1), f"{backend}/{name}"


def test_accept_total_keeps_the_total_cost_monotone(study):
    cfg = replace(study, fitting=replace(study.fitting, accept="total", max_loops=5))
    result = fit(cfg)
    for step in result.steps:
        assert step.total_after <= step.total_before + 1e-12


def test_a_seed_makes_the_run_reproducible(study):
    a = fit(study).parameters
    b = fit(study).parameters
    assert a == pytest.approx(b)


def test_results_are_written_to_disk(study, tmp_path):
    result = fit(study)
    written = result.save(tmp_path / "out")
    for key in ("report", "history", "loops", "parameters_csv", "parameters_toml"):
        assert written[key].is_file()
    text = written["parameters_toml"].read_text()
    assert "[parameters]" in text and "Vmax1" in text

    from pymodest.config import _toml

    reloaded = _toml.loads(text)["parameters"]
    assert reloaded["Vmax1"] == pytest.approx(result.parameters["Vmax1"])


def test_predictions_are_written_per_dataset(study, tmp_path):
    estimator = ModularEstimator(study)
    result = estimator.run()
    paths = estimator.write_predictions(tmp_path / "pred", result.parameters)
    assert [p.name for p in paths] == ["predictions_exp.csv"]

    import pandas as pd

    frame = pd.read_csv(paths[0])
    assert {"time", "A_fit", "B_fit", "A_obs", "B_obs"} <= set(frame.columns)


# -- edge cases -------------------------------------------------------------

def test_fixed_parameters_are_held_but_still_applied(study):
    """A fixed parameter keeps its init value and is written to the model."""
    from pymodest.config import ModuleSpec, ParameterSpec

    pinned = ModuleSpec(
        id="downstream",
        parameters=[
            ParameterSpec("k2", 0.01, 5.0, init=TRUE_PARAMETERS["k2"], scale="log", fixed=True),
            ParameterSpec("k3", 0.01, 5.0, init=1.0, scale="log"),
        ],
        variables=["B"],
    )
    cfg = replace(study, modules=[study.modules[0], pinned])
    result = fit(cfg)
    assert result.parameters["k2"] == pytest.approx(TRUE_PARAMETERS["k2"])
    assert result.parameters["k3"] == pytest.approx(TRUE_PARAMETERS["k3"], rel=0.05)


def test_a_module_with_no_free_parameters_is_skipped(study):
    from pymodest.config import ModuleSpec, ParameterSpec

    frozen = ModuleSpec(
        id="downstream",
        parameters=[
            ParameterSpec("k2", 0.01, 5.0, init=0.6, scale="log", fixed=True),
            ParameterSpec("k3", 0.01, 5.0, init=0.3, scale="log", fixed=True),
        ],
        variables=["B"],
    )
    cfg = replace(
        study, modules=[study.modules[0], frozen],
        fitting=replace(study.fitting, max_loops=2, tol=0.0, patience=99),
    )
    result = fit(cfg)
    assert {s.module for s in result.steps} == {"upstream"}
    assert result.parameters["k2"] == pytest.approx(0.6)


def test_a_module_can_be_restricted_to_particular_datasets(study, synthetic_csv):
    """A module scored on a subset of the experiments ignores the others."""
    from pymodest.config import DatasetSpec

    second = DatasetSpec(
        id="exp2", model="toy", file=synthetic_csv, initial_conditions={"S": 4.0}
    )
    upstream = replace(study.modules[0], datasets=["exp"])
    cfg = replace(
        study,
        datasets=[study.datasets[0], second],
        modules=[upstream, study.modules[1]],
    )
    cfg.validate()
    problem = Problem(cfg)
    assert [d.id for d in problem.datasets_for(cfg.module("upstream"))] == ["exp"]
    assert [d.id for d in problem.datasets_for(cfg.module("downstream"))] == ["exp", "exp2"]


def test_a_per_module_optimizer_overrides_the_study_default(study):
    cfg = replace(
        study,
        modules=[
            replace(study.modules[0], optimizer=OptimizerSpec("particle_swarm", {"maxiter": 15})),
            study.modules[1],
        ],
        fitting=replace(study.fitting, max_loops=1, tol=0.0, patience=99),
    )
    result = fit(cfg)
    used = {s.module: s.optimizer for s in result.steps}
    assert used["upstream"] == "particle_swarm"
    assert used["downstream"] == "differential_evolution"


def test_the_refine_optimizer_takes_over_after_the_first_loop(study):
    cfg = replace(
        study,
        fitting=replace(
            study.fitting,
            refine=OptimizerSpec("least_squares", {"max_nfev": 100}),
            refine_after=1,
            max_loops=3, tol=0.0, patience=99,
        ),
    )
    result = fit(cfg)
    by_loop = {(s.loop, s.module): s.optimizer for s in result.steps}
    assert by_loop[(1, "upstream")] == "differential_evolution"
    assert by_loop[(2, "upstream")] == "least_squares"
