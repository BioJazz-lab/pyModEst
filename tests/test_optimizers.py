"""Every registered backend must minimise a bounded problem and respect bounds."""

from __future__ import annotations

import numpy as np
import pytest

from pymodest import optimizers
from pymodest.optimizers import OptimizerError, OptimizerResult, latin_hypercube


class Quadratic:
    """A ModuleObjective-shaped stand-in with a known minimum at (1, -2)."""

    target = np.array([1.0, -2.0])

    def __init__(self, bounds=((-5.0, 5.0), (-5.0, 5.0)), x0=(4.0, 4.0)):
        self.bounds = [tuple(b) for b in bounds]
        self.x0 = np.array(x0, dtype=float)
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return float(np.sum((np.asarray(x) - self.target) ** 2))

    def residuals(self, x):
        self.calls += 1
        return np.asarray(x, dtype=float) - self.target


BACKENDS = ["differential_evolution", "least_squares", "minimize", "particle_swarm", "scatter_search"]


def test_all_expected_backends_are_registered():
    assert set(BACKENDS) <= set(optimizers.available())


@pytest.mark.parametrize("name", BACKENDS)
def test_backend_finds_the_minimum(name):
    objective = Quadratic()
    result = optimizers.run(name, objective, rng=np.random.default_rng(0))
    assert isinstance(result, OptimizerResult)
    assert result.fun == pytest.approx(0.0, abs=1e-4)
    assert np.allclose(result.x, Quadratic.target, atol=1e-2)
    assert result.backend == name
    assert result.nfev > 0


@pytest.mark.parametrize("name", BACKENDS)
def test_backend_stays_inside_the_bounds(name):
    """The optimum lies outside the box, so the answer must sit on the wall."""
    objective = Quadratic(bounds=((2.0, 5.0), (2.0, 5.0)), x0=(3.0, 3.0))
    result = optimizers.run(name, objective, rng=np.random.default_rng(1))
    assert result.x[0] >= 2.0 - 1e-9 and result.x[0] <= 5.0 + 1e-9
    assert result.x[1] >= 2.0 - 1e-9 and result.x[1] <= 5.0 + 1e-9
    assert result.x[0] == pytest.approx(2.0, abs=1e-2)


@pytest.mark.parametrize("name", BACKENDS)
def test_backends_are_reproducible_given_a_seed(name):
    a = optimizers.run(name, Quadratic(), rng=np.random.default_rng(42))
    b = optimizers.run(name, Quadratic(), rng=np.random.default_rng(42))
    assert np.allclose(a.x, b.x)


def test_aliases_resolve():
    for alias, canonical in [
        ("de", "differential_evolution"),
        ("pso", "particle_swarm"),
        ("ss", "scatter_search"),
        ("trf", "least_squares"),
        ("Nelder-Mead", "minimize"),
    ]:
        result = optimizers.run(alias, Quadratic(), rng=np.random.default_rng(0))
        assert result.backend == canonical


def test_unknown_backend_is_reported():
    with pytest.raises(OptimizerError, match="unknown optimizer"):
        optimizers.run("no_such_method", Quadratic())


def test_a_custom_backend_can_be_registered():
    @optimizers.register("always_target")
    def always_target(objective, x0, bounds, rng, **options):
        return OptimizerResult(x=Quadratic.target, fun=objective(Quadratic.target))

    result = optimizers.run("always_target", Quadratic())
    assert result.fun == pytest.approx(0.0)
    assert "always_target" in optimizers.available()


def test_latin_hypercube_covers_the_box():
    rng = np.random.default_rng(0)
    points = latin_hypercube(rng, [(0.0, 1.0), (10.0, 20.0)], 20)
    assert points.shape == (20, 2)
    assert points[:, 0].min() >= 0.0 and points[:, 0].max() <= 1.0
    assert points[:, 1].min() >= 10.0 and points[:, 1].max() <= 20.0
    # one point per stratum in each dimension
    assert len(np.unique(np.floor(points[:, 0] * 20))) == 20


def test_swarm_handles_a_single_parameter():
    objective = Quadratic(bounds=((-5.0, 5.0),), x0=(3.0,))
    objective.target = np.array([1.0])
    result = optimizers.run("particle_swarm", objective, rng=np.random.default_rng(0))
    assert result.x[0] == pytest.approx(1.0, abs=1e-2)


def test_scatter_search_respects_its_evaluation_budget():
    objective = Quadratic()
    optimizers.run(
        "scatter_search", objective, rng=np.random.default_rng(0),
        max_nfev=200, maxiter=100, refset_size=6,
    )
    assert objective.calls <= 400  # budget plus one final iteration's trials
