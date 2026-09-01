"""Configuration parsing and validation."""

from __future__ import annotations

import textwrap

import pytest

from pymodest.config import (
    ConfigError, ModuleSpec, OptimizerSpec, ParameterSpec, load_config,
)


# -- parameter specs --------------------------------------------------------

def test_linear_parameter_defaults_to_midpoint():
    p = ParameterSpec("k", 0.0, 10.0)
    assert p.initial_value == 5.0
    assert p.search_bounds == (0.0, 10.0)
    assert p.to_model(p.to_search(3.0)) == pytest.approx(3.0)


def test_log_parameter_uses_geometric_midpoint_and_log_bounds():
    p = ParameterSpec("k", 0.01, 100.0, scale="log")
    assert p.initial_value == pytest.approx(1.0)
    assert p.search_bounds == (pytest.approx(-2.0), pytest.approx(2.0))
    assert p.to_search(10.0) == pytest.approx(1.0)
    assert p.to_model(1.0) == pytest.approx(10.0)


def test_explicit_init_is_respected():
    assert ParameterSpec("k", 0.1, 10.0, init=7.0).initial_value == 7.0


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(lower=5.0, upper=1.0), "must exceed"),
        (dict(lower=0.0, upper=10.0, scale="log"), "positive lower bound"),
        (dict(lower=1.0, upper=10.0, init=99.0), "outside"),
        (dict(lower=1.0, upper=10.0, scale="quadratic"), "scale must be"),
    ],
)
def test_invalid_parameter_specs_are_rejected(kwargs, message):
    with pytest.raises(ConfigError, match=message):
        ParameterSpec("k", **kwargs)


# -- modules ----------------------------------------------------------------

def test_module_requires_parameters_and_variables():
    with pytest.raises(ConfigError, match="at least one parameter"):
        ModuleSpec(id="m", parameters=[], variables=["A"])
    with pytest.raises(ConfigError, match="at least one variable"):
        ModuleSpec(id="m", parameters=[ParameterSpec("k", 0.0, 1.0)], variables=[])


def test_module_rejects_duplicate_parameters():
    with pytest.raises(ConfigError, match="duplicate parameter"):
        ModuleSpec(
            id="m",
            parameters=[ParameterSpec("k", 0.0, 1.0), ParameterSpec("k", 0.0, 2.0)],
            variables=["A"],
        )


def test_module_weights_must_name_module_variables():
    with pytest.raises(ConfigError, match="non-module variable"):
        ModuleSpec(
            id="m",
            parameters=[ParameterSpec("k", 0.0, 1.0)],
            variables=["A"],
            weights={"B": 2.0},
        )


def test_module_optimizer_options_merge_with_the_study_default():
    default = OptimizerSpec("differential_evolution", {"maxiter": 50, "popsize": 10})
    same = OptimizerSpec("differential_evolution", {"maxiter": 5}).merged_with(default)
    assert same.options == {"maxiter": 5, "popsize": 10}
    # a different backend does not inherit the other one's options
    other = OptimizerSpec("particle_swarm", {"maxiter": 5}).merged_with(default)
    assert other.options == {"maxiter": 5}


# -- whole studies ----------------------------------------------------------

def test_modules_must_partition_the_parameters(study):
    from dataclasses import replace

    clash = ModuleSpec(
        id="other", parameters=[ParameterSpec("Vmax1", 0.1, 5.0)], variables=["B"]
    )
    bad = replace(study, modules=list(study.modules) + [clash])
    with pytest.raises(ConfigError, match="must partition"):
        bad.validate()


def test_dataset_must_reference_a_known_model(study):
    from dataclasses import replace

    bad = replace(study, datasets=[replace(study.datasets[0], model="ghost")])
    with pytest.raises(ConfigError, match="unknown model"):
        bad.validate()


def test_module_order_must_cover_every_module(study):
    from dataclasses import replace

    bad = replace(study, fitting=replace(study.fitting, module_order=["upstream"]))
    with pytest.raises(ConfigError, match="omits module"):
        bad.validate()


def test_lookups_and_initial_values(study):
    assert study.module_ids == ["upstream", "downstream"]
    assert study.parameter_module("k3") == "downstream"
    assert study.initial_parameter_values()["Vmax1"] == pytest.approx(1.0)
    assert study.resolved_module_order() == ["upstream", "downstream"]


# -- TOML round trip --------------------------------------------------------

TOML = """
[study]
name = "written-study"

[[models]]
id = "m1"
antimony = "model m1\\n J1: -> A; k1;\\n A = 0; k1 = 1;\\nend"

[[datasets]]
id = "d1"
model = "m1"
[datasets.data]
time = [0.0, 1.0, 2.0]
A = [0.0, 1.0, 2.0]

[[modules]]
id = "only"
variables = ["A"]
[[modules.parameters]]
name = "k1"
lower = 0.1
upper = 10.0
scale = "log"

[fitting]
max_loops = 3
accept = "total"
[fitting.optimizer]
name = "particle_swarm"
maxiter = 5
"""


def test_load_config_reads_inline_models_and_data(tmp_path):
    path = tmp_path / "study.toml"
    path.write_text(TOML)
    config = load_config(path)
    assert config.name == "written-study"
    assert config.models[0].antimony.startswith("model m1")
    assert config.datasets[0].inline["A"] == [0.0, 1.0, 2.0]
    assert config.fitting.accept == "total"
    assert config.optimizer_for(config.modules[0]).name == "particle_swarm"


def test_unknown_top_level_keys_are_reported(tmp_path):
    path = tmp_path / "study.toml"
    path.write_text("nonsense = 1\n" + TOML)
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(path)


def test_unknown_model_keys_are_reported(tmp_path):
    path = tmp_path / "study.toml"
    path.write_text(TOML.replace('id = "m1"', 'id = "m1"\nmispelled = 1'))
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(path)


def test_optimizer_options_stay_free_form(tmp_path):
    """Backend options are passed straight through, so they are not validated."""
    path = tmp_path / "study.toml"
    path.write_text(TOML + "\ncognitive = 1.4\n")
    config = load_config(path)
    assert config.fitting.optimizer.options["cognitive"] == 1.4


def test_relative_paths_resolve_against_the_config_file(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "m.ant").write_text("model m\n J: -> A; k;\n A=0; k=1;\nend")
    path = tmp_path / "study.toml"
    path.write_text(
        textwrap.dedent(
            """
            [[models]]
            id = "m"
            antimony_file = "models/m.ant"

            [[datasets]]
            id = "d"
            model = "m"
            [datasets.data]
            time = [0.0, 1.0]
            A = [0.0, 1.0]

            [[modules]]
            id = "only"
            variables = ["A"]
            [[modules.parameters]]
            name = "k"
            lower = 0.1
            upper = 10.0
            """
        )
    )
    config = load_config(path)
    assert config.models[0].source == (tmp_path / "models" / "m.ant").resolve()


def test_missing_file_is_reported_clearly(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "absent.toml")
