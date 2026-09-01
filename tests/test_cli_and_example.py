"""The command line interface, and the shipped example end to end."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pymodest.cli import main

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "two_module_pathway"


# -- CLI --------------------------------------------------------------------

def test_optimizers_command_lists_backends(capsys):
    assert main(["optimizers"]) == 0
    out = capsys.readouterr().out
    for name in ("differential_evolution", "particle_swarm", "scatter_search"):
        assert name in out


def test_template_command_writes_a_loadable_skeleton(tmp_path, capsys):
    path = tmp_path / "study.toml"
    assert main(["template", "--out", str(path)]) == 0
    from pymodest.config import _toml

    with open(path, "rb") as handle:
        table = _toml.load(handle)
    assert "models" in table and "modules" in table and "fitting" in table


def test_validate_reports_the_study(capsys):
    assert main(["validate", str(EXAMPLE / "config.toml")]) == 0
    out = capsys.readouterr().out
    assert "two-module-pathway" in out
    assert "upstream" in out and "downstream" in out
    assert "cost at initial parameter values" in out


def test_a_broken_config_exits_with_an_error_code(tmp_path, capsys):
    bad = tmp_path / "bad.toml"
    bad.write_text('[[modules]]\nid = "m"\nvariables = ["A"]\n')
    assert main(["validate", str(bad)]) == 2
    assert "error:" in capsys.readouterr().err


def test_fit_command_writes_results(tmp_path, capsys):
    out = tmp_path / "cli-results"
    code = main([
        "-q", "fit", str(EXAMPLE / "config.toml"),
        "--loops", "2", "--out", str(out), "--seed", "1",
    ])
    assert code == 0
    for name in ("best_parameters.toml", "history.csv", "loop_summary.csv", "fit_report.json"):
        assert (out / name).is_file()
    assert (out / "predictions_wt_high.csv").is_file()
    assert (out / "predictions_fb_high.csv").is_file()


def test_simulate_command_uses_supplied_parameters(tmp_path, capsys):
    out = tmp_path / "sim"
    code = main([
        "-q", "simulate", str(EXAMPLE / "config.toml"),
        "--parameters", str(EXAMPLE / "true_parameters.toml"), "--out", str(out),
    ])
    assert code == 0
    assert (out / "predictions_wt_low.csv").is_file()
    assert "cost at these parameters" in capsys.readouterr().out


# -- the example itself -----------------------------------------------------

@pytest.fixture(scope="module")
def example_fit():
    from pymodest import load_config
    from pymodest.estimator import ModularEstimator

    config = load_config(EXAMPLE / "config.toml")
    estimator = ModularEstimator(config)
    return config, estimator, estimator.run()


def test_example_has_two_models_sharing_one_parameter_set(example_fit):
    config, estimator, _ = example_fit
    assert set(config.model_ids) == {"wt", "feedback"}
    # Ki exists only in the feedback model; the shared parameters exist in both
    assert not estimator.models["wt"].has_parameter("Ki")
    assert estimator.models["feedback"].has_parameter("Ki")
    for shared in ("Vmax1", "Km1", "k2", "Vmax3", "Km3", "k4"):
        assert estimator.models["wt"].has_parameter(shared)
        assert estimator.models["feedback"].has_parameter(shared)


def test_example_fit_beats_the_starting_point_and_nears_the_noise_floor(example_fit):
    config, estimator, result = example_fit
    from pymodest.config import _toml
    from pymodest.objective import Problem

    with open(EXAMPLE / "true_parameters.toml", "rb") as handle:
        truth = _toml.load(handle)["parameters"]

    floor = Problem(config)
    floor.set_values(truth)
    noise_floor = floor.total_cost()

    assert result.cost < result.initial_cost / 100
    # the data carry ~4% noise, so the fit cannot beat the truth by much
    assert result.cost < 2.0 * noise_floor


def test_example_recovers_the_generating_parameters(example_fit):
    _, _, result = example_fit
    from pymodest.config import _toml

    with open(EXAMPLE / "true_parameters.toml", "rb") as handle:
        truth = _toml.load(handle)["parameters"]

    # noisy data and correlated Vmax/Km pairs: judge on the order of magnitude
    errors = {k: abs(np.log10(result.parameters[k] / truth[k])) for k in truth}
    assert max(errors.values()) < 0.35, errors      # within ~2.2x, worst case
    assert float(np.mean(list(errors.values()))) < 0.15, errors
    # the parameters that the data pin down tightly should be close
    assert result.parameters["k4"] == pytest.approx(truth["k4"], rel=0.15)


def test_example_reduces_every_module_cost(example_fit):
    config, estimator, result = example_fit
    from pymodest.objective import Problem

    start = Problem(config)
    for module in config.modules:
        assert result.module_costs[module.id] < start.module_cost(module)
