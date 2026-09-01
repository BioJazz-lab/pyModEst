# pyModEst documentation

Divide-and-conquer parameter estimation for biological models written in
Antimony.

| | |
| --- | --- |
| [Concepts](concepts.md) | what module-wise fitting is, when it helps, and how it fails |
| [Configuration](configuration.md) | the complete TOML reference |
| [Capabilities](capabilities.md) | what the package does, each demonstrated by a test |
| [Optimizers](optimizers.md) | the five backends, how to choose, how to add one |
| [Python API](api.md) | using pyModEst as a library |
| [Troubleshooting](troubleshooting.md) | what goes wrong and what it means |
| [Examples](examples/) | runnable scripts — the source of the output quoted here |

The repository [README](../README.md) is the short overview; these pages are
the reference.

## Install and run

```bash
uv sync                                       # from a clone
uv run pymodest template --out study.toml     # a commented starter config
uv run pymodest validate study.toml           # check it before fitting anything
uv run pymodest fit study.toml
```

## The shape of a study

Four things go into a study, and they are declared in one TOML file:

- **Models** — one or more Antimony models that share a single fitted
  parameter set.
- **Datasets** — measurements, each attached to one model, each under its own
  starting conditions.
- **Modules** — the partition. Each module names the parameters fitted
  together and the measured variables that score them.
- **Fitting** — how the loop runs: optimizer, stopping rules, objective.

Fitting then proceeds module by module:

```
theta <- initial values
repeat up to max_loops times:
    for each module m:
        theta[m] <- argmin cost_m(theta[m] ; theta[not m] held fixed)
    stop when the total cost stops improving
```

## The shortest complete example

No config file, no data files — see
[`examples/01_first_fit.py`](examples/01_first_fit.py) for the full version:

```python
from pymodest import fit
from pymodest.config import (
    DatasetSpec, FittingSpec, ModelSpec, ModuleSpec,
    OptimizerSpec, ParameterSpec, StudyConfig,
)

config = StudyConfig(
    name="first-fit",
    models=[ModelSpec(id="toy", antimony=ANTIMONY)],
    datasets=[DatasetSpec(id="exp", model="toy", inline=measurements,
                          initial_conditions={"S": 10.0})],
    modules=[
        ModuleSpec(id="upstream", variables=["A"], parameters=[
            ParameterSpec("Vmax1", lower=0.1, upper=20.0, scale="log"),
            ParameterSpec("Km1", lower=0.1, upper=40.0, scale="log"),
        ]),
        ModuleSpec(id="downstream", variables=["B"], parameters=[
            ParameterSpec("k2", lower=0.01, upper=5.0, scale="log"),
            ParameterSpec("k3", lower=0.01, upper=5.0, scale="log"),
        ]),
    ],
    fitting=FittingSpec(max_loops=6, seed=3),
)
config.validate()
result = fit(config)
print(result.parameter_table())
```

Running it recovers all four parameters from noise-free data:

```
cost 0.2096 -> 6.799e-08 in 6 loops (7384 evaluations)

parameter     true   estimate    error
Km1          4.000     4.0053    0.1%
Vmax1        2.000     2.0010    0.0%
k2           0.600     0.5999    0.0%
k3           0.300     0.3000    0.0%
```
