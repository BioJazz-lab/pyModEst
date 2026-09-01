"""Command line interface: ``pymodest <command> ...``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__, optimizers
from .config import ConfigError, StudyConfig, load_config
from .data import DataError
from .estimator import ModularEstimator
from .objective import Problem, ProblemError

LOG_FORMAT = "%(message)s"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity < 0 else (
        logging.DEBUG if verbosity > 0 else logging.INFO
    )
    logging.basicConfig(level=level, format=LOG_FORMAT, stream=sys.stdout, force=True)


def _load(path: str) -> StudyConfig:
    return load_config(path)


def _describe(config: StudyConfig) -> str:
    lines = [f"study: {config.name}", ""]
    lines.append(f"models ({len(config.models)}):")
    for m in config.models:
        source = m.source.name if m.source else "inline"
        lines.append(f"  - {m.id}  [{source}]" + (f"  {m.description}" if m.description else ""))
    lines.append("")
    lines.append(f"datasets ({len(config.datasets)}):")
    for d in config.datasets:
        source = d.file.name if d.file else "inline"
        lines.append(f"  - {d.id}  model={d.model}  [{source}]  weight={d.weight:g}")
    lines.append("")
    lines.append(f"modules ({len(config.modules)}):")
    for mod in config.modules:
        opt = config.optimizer_for(mod)
        lines.append(f"  - {mod.id}  optimizer={opt.name}")
        lines.append(f"      variables:  {', '.join(mod.variables)}")
        for p in mod.parameters:
            flag = "  (fixed)" if p.fixed else ""
            lines.append(
                f"      parameter:  {p.name:<12} [{p.lower:g}, {p.upper:g}] "
                f"{p.scale}  init={p.initial_value:g}{flag}"
            )
    lines.append("")
    fitting = config.fitting
    lines.append(
        f"fitting: max_loops={fitting.max_loops}  order={fitting.module_order}  "
        f"tol={fitting.tol:g}  patience={fitting.patience}"
    )
    lines.append(
        f"objective: scaling={fitting.objective.scaling}  "
        f"aggregation={fitting.objective.aggregation}"
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_validate(args: argparse.Namespace) -> int:
    config = _load(args.config)
    print(_describe(config))
    problem = Problem(config)
    print("")
    print("consistency check: models, datasets and modules agree")
    total = problem.total_cost()
    print(f"cost at initial parameter values: {total:.6g}")
    for module in config.modules:
        print(f"  {module.id:<20} {problem.module_cost(module):.6g}")
    return 0


def cmd_fit(args: argparse.Namespace) -> int:
    config = _load(args.config)
    if args.out:
        config = config.with_output_dir(Path(args.out).resolve())
    if args.optimizer:
        from dataclasses import replace
        from .config import OptimizerSpec

        fitting = replace(config.fitting, optimizer=OptimizerSpec(name=args.optimizer))
        config = replace(config, fitting=fitting)
    if args.seed is not None:
        from dataclasses import replace

        config = replace(config, fitting=replace(config.fitting, seed=args.seed))

    estimator = ModularEstimator(config)
    result = estimator.run(max_loops=args.loops)

    print("")
    print(result.parameter_table().to_string(index=False))
    print("")
    print(f"total cost: {result.initial_cost:.6g} -> {result.cost:.6g}")
    print(f"stopped because: {result.stop_reason}")

    written = result.save(config.output_dir)
    if not args.no_predictions:
        written_predictions = estimator.write_predictions(
            config.output_dir, result.parameters
        )
    else:
        written_predictions = []
    print("")
    print(f"results written to {config.output_dir}")
    for path in list(written.values()) + written_predictions:
        print(f"  {path.name}")
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    config = _load(args.config)
    if args.out:
        config = config.with_output_dir(Path(args.out).resolve())
    estimator = ModularEstimator(config)

    values = None
    if args.parameters:
        try:
            import tomllib as _toml
        except ModuleNotFoundError:  # pragma: no cover
            import tomli as _toml  # type: ignore
        with open(args.parameters, "rb") as handle:
            table = _toml.load(handle)
        values = {k: float(v) for k, v in table.get("parameters", table).items()}
        estimator.problem.set_values(
            {k: v for k, v in values.items() if k in estimator.problem.parameter_specs}
        )

    paths = estimator.write_predictions(config.output_dir, estimator.problem.snapshot())
    print(f"cost at these parameters: {estimator.problem.total_cost():.6g}")
    print(f"predictions written to {config.output_dir}")
    for path in paths:
        print(f"  {path.name}")
    return 0


def cmd_optimizers(_: argparse.Namespace) -> int:
    print("available optimizer backends:")
    for name in optimizers.available():
        print(f"  {name}")
    return 0


TEMPLATE = '''# pyModEst study configuration
#
# Fitting proceeds module by module: each module's parameters are optimized
# against that module's variables while all other parameters stay fixed, and
# the loop repeats until the total cost stops improving.

[study]
name = "my-study"
output_dir = "results"

# ---------------------------------------------------------------- models ---
# Several models may share one fitted parameter set.
[[models]]
id = "wt"
antimony_file = "models/wt.ant"
# [models.overrides]        # values pinned for this model only
# [models.observables]      # derived quantities, e.g. Total = "A + B"

# -------------------------------------------------------------- datasets ---
# Each dataset is measured on one model, under its own conditions.
[[datasets]]
id = "exp1"
model = "wt"
file = "data/exp1.csv"
format = "wide"              # wide: time,A,B,...   long: time,variable,value
weight = 1.0
# [datasets.conditions]         # parameters set for this experiment
# [datasets.initial_conditions] # species starting values

# --------------------------------------------------------------- modules ---
[[modules]]
id = "module_one"
variables = ["A", "B"]       # measured variables scoring this module

[[modules.parameters]]
name = "k1"
lower = 1e-3
upper = 1e2
scale = "log"

[[modules.parameters]]
name = "Km1"
lower = 1e-2
upper = 1e3
scale = "log"

# --------------------------------------------------------------- fitting ---
[fitting]
max_loops = 12
module_order = "as_listed"   # or a list of module ids, "random", "round_robin_reversed"
tol = 1e-4                   # relative improvement that counts as progress
atol = 1e-12                 # absolute floor, so a cost heading to zero terminates
patience = 2                 # loops without progress before stopping
accept = "module"            # "module" keeps a step that helps its own module;
                             # "total" also requires the overall cost not to rise
seed = 0

[fitting.optimizer]
name = "differential_evolution"
maxiter = 60
popsize = 15

[fitting.objective]
scaling = "relative"         # relative | absolute | sigma | max_normalized
aggregation = "mean"

[fitting.simulation]
integrator = "cvode"
relative_tolerance = 1e-8
absolute_tolerance = 1e-10
'''


def cmd_template(args: argparse.Namespace) -> int:
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(TEMPLATE)
        print(f"wrote {path}")
    else:
        print(TEMPLATE)
    return 0


# --------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pymodest",
        description="Divide-and-conquer parameter estimation for biological models.",
    )
    parser.add_argument("--version", action="version", version=f"pymodest {__version__}")
    parser.add_argument("-q", "--quiet", action="store_true", help="only warnings and errors")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", help="load a study and report what it contains")
    p.add_argument("config", help="path to the study TOML file")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("fit", help="run the module-wise parameter estimation")
    p.add_argument("config", help="path to the study TOML file")
    p.add_argument("--loops", type=int, default=None, help="override max_loops")
    p.add_argument("--out", default=None, help="override the output directory")
    p.add_argument("--optimizer", default=None, help="override the default optimizer")
    p.add_argument("--seed", type=int, default=None, help="override the random seed")
    p.add_argument(
        "--no-predictions", action="store_true", help="skip writing simulated traces"
    )
    p.set_defaults(func=cmd_fit)

    p = sub.add_parser("simulate", help="simulate the study at given parameter values")
    p.add_argument("config", help="path to the study TOML file")
    p.add_argument(
        "--parameters", default=None, help="TOML file with a [parameters] table"
    )
    p.add_argument("--out", default=None, help="override the output directory")
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("optimizers", help="list the registered optimizer backends")
    p.set_defaults(func=cmd_optimizers)

    p = sub.add_parser("template", help="print a commented starter configuration")
    p.add_argument("--out", default=None, help="write to this file instead of stdout")
    p.set_defaults(func=cmd_template)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(-1 if args.quiet else (1 if args.verbose else 0))
    try:
        return int(args.func(args))
    except (ConfigError, DataError, ProblemError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
