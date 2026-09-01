"""pyModEst -- divide-and-conquer parameter estimation for biological models.

Models are written in Antimony, simulated with roadrunner, and fitted against
one or more experimental datasets. Parameters are partitioned into *modules*;
each module is fitted against its own measured variables while the rest of the
parameter set is held fixed, and the procedure cycles through the modules for a
finite number of loops.

Typical use::

    from pymodest import load_config, fit

    config = load_config("study.toml")
    result = fit(config)
    print(result.parameters)
    result.save(config.output_dir)
"""

from .config import (  # noqa: F401
    ConfigError,
    DatasetSpec,
    FittingSpec,
    ModelSpec,
    ModuleSpec,
    ObjectiveSpec,
    OptimizerSpec,
    ParameterSpec,
    SimulationSpec,
    StudyConfig,
    load_config,
)
from .data import DataError, ExperimentData, Measurement, load_dataset, load_datasets  # noqa: F401
from .estimator import ModularEstimator, fit, fit_from_file  # noqa: F401
from .model import ModelError, SimulationFailure, SimulationModel, build_models  # noqa: F401
from .objective import ModuleObjective, Problem, ProblemError  # noqa: F401
from .optimizers import OptimizerResult, available as available_optimizers, register  # noqa: F401
from .result import FitResult, LoopRecord, ModuleStep  # noqa: F401

__version__ = "0.1.0"

__all__ = [
    "ConfigError", "DataError", "DatasetSpec", "ExperimentData", "FitResult",
    "FittingSpec", "LoopRecord", "Measurement", "ModelError", "ModelSpec",
    "ModularEstimator", "ModuleObjective", "ModuleSpec", "ModuleStep",
    "ObjectiveSpec", "OptimizerResult", "OptimizerSpec", "ParameterSpec",
    "Problem", "ProblemError", "SimulationFailure", "SimulationModel",
    "SimulationSpec", "StudyConfig", "available_optimizers", "build_models",
    "fit", "fit_from_file", "load_config", "load_dataset", "load_datasets",
    "register", "__version__",
]
