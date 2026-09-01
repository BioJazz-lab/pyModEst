"""Records produced by a fit: per-module steps, per-loop summaries, final result."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

__all__ = ["ModuleStep", "LoopRecord", "FitResult"]


def _clean(value: Any) -> Any:
    """Make numpy types JSON- and TOML-serialisable."""
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_clean(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass
class ModuleStep:
    """One module fitted once, inside one loop."""

    loop: int
    module: str
    optimizer: str
    cost_before: float
    cost_after: float
    total_before: float
    total_after: float
    parameters: Dict[str, float]
    nfev: int = 0
    seconds: float = 0.0
    accepted: bool = True
    message: str = ""

    @property
    def improvement(self) -> float:
        return self.cost_before - self.cost_after


@dataclass
class LoopRecord:
    """One full sweep over every module."""

    loop: int
    steps: List[ModuleStep] = field(default_factory=list)
    total_cost: float = float("nan")
    module_costs: Dict[str, float] = field(default_factory=dict)
    parameters: Dict[str, float] = field(default_factory=dict)
    seconds: float = 0.0
    relative_improvement: float = float("nan")


@dataclass
class FitResult:
    """Everything a completed run produced."""

    parameters: Dict[str, float]
    cost: float
    module_costs: Dict[str, float]
    loops: List[LoopRecord] = field(default_factory=list)
    converged: bool = False
    stop_reason: str = ""
    n_evaluations: int = 0
    seconds: float = 0.0
    initial_parameters: Dict[str, float] = field(default_factory=dict)
    initial_cost: float = float("nan")
    study: str = ""

    # -- views -------------------------------------------------------------
    @property
    def n_loops(self) -> int:
        return len(self.loops)

    @property
    def steps(self) -> List[ModuleStep]:
        return [s for loop in self.loops for s in loop.steps]

    def history(self) -> pd.DataFrame:
        """One row per module fit, in the order they were performed."""
        rows = []
        for step in self.steps:
            row = {
                "loop": step.loop,
                "module": step.module,
                "optimizer": step.optimizer,
                "cost_before": step.cost_before,
                "cost_after": step.cost_after,
                "improvement": step.improvement,
                "total_before": step.total_before,
                "total_after": step.total_after,
                "nfev": step.nfev,
                "seconds": step.seconds,
                "accepted": step.accepted,
            }
            row.update({f"p:{k}": v for k, v in step.parameters.items()})
            rows.append(row)
        return pd.DataFrame(rows)

    def loop_summary(self) -> pd.DataFrame:
        """One row per loop: the total cost after that sweep."""
        rows = []
        for loop in self.loops:
            row = {
                "loop": loop.loop,
                "total_cost": loop.total_cost,
                "relative_improvement": loop.relative_improvement,
                "seconds": loop.seconds,
            }
            row.update({f"cost:{k}": v for k, v in loop.module_costs.items()})
            row.update({f"p:{k}": v for k, v in loop.parameters.items()})
            rows.append(row)
        return pd.DataFrame(rows)

    def parameter_table(self) -> pd.DataFrame:
        """Final values beside their starting values, per parameter."""
        names = sorted(self.parameters)
        return pd.DataFrame(
            {
                "parameter": names,
                "initial": [self.initial_parameters.get(n, float("nan")) for n in names],
                "estimate": [self.parameters[n] for n in names],
            }
        )

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return _clean(
            {
                "study": self.study,
                "converged": self.converged,
                "stop_reason": self.stop_reason,
                "cost": self.cost,
                "initial_cost": self.initial_cost,
                "module_costs": self.module_costs,
                "parameters": self.parameters,
                "initial_parameters": self.initial_parameters,
                "n_loops": self.n_loops,
                "n_evaluations": self.n_evaluations,
                "seconds": self.seconds,
                "loops": [
                    {
                        "loop": loop.loop,
                        "total_cost": loop.total_cost,
                        "relative_improvement": loop.relative_improvement,
                        "module_costs": loop.module_costs,
                        "parameters": loop.parameters,
                        "seconds": loop.seconds,
                        "steps": [asdict(s) for s in loop.steps],
                    }
                    for loop in self.loops
                ],
            }
        )

    def save(self, directory: Path) -> Dict[str, Path]:
        """Write the report, histories and best parameters into ``directory``."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        written: Dict[str, Path] = {}

        report = directory / "fit_report.json"
        report.write_text(json.dumps(self.to_dict(), indent=2))
        written["report"] = report

        history = directory / "history.csv"
        self.history().to_csv(history, index=False)
        written["history"] = history

        loops = directory / "loop_summary.csv"
        self.loop_summary().to_csv(loops, index=False)
        written["loops"] = loops

        table = directory / "parameters.csv"
        self.parameter_table().to_csv(table, index=False)
        written["parameters_csv"] = table

        best = directory / "best_parameters.toml"
        best.write_text(self._parameters_toml())
        written["parameters_toml"] = best
        return written

    def _parameters_toml(self) -> str:
        lines = [
            "# pyModEst estimated parameters",
            f"# study = {self.study!r}",
            f"# cost = {self.cost:.6g}   loops = {self.n_loops}   converged = {self.converged}",
            "",
            "[parameters]",
        ]
        for name in sorted(self.parameters):
            lines.append(f"{name} = {self.parameters[name]!r}")
        return "\n".join(lines) + "\n"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"FitResult(cost={self.cost:.6g}, loops={self.n_loops}, "
            f"converged={self.converged}, parameters={len(self.parameters)})"
        )
