"""Experimental data: what was measured, when, and with what uncertainty.

A dataset is a table of time points and measured variables. Two layouts are
accepted:

``wide``  ``time, A, B, ...`` (optionally ``A_sigma``, ``B_sigma``)
``long``  ``time, variable, value`` (optionally ``sigma``)

Each variable is stored as its own :class:`Measurement` so that a module can
pick out only the variables it scores, and so that variables measured at
different time points need not be padded onto a common grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .config import DatasetSpec

__all__ = ["DataError", "Measurement", "ExperimentData", "load_dataset", "load_datasets"]

SIGMA_SUFFIXES = ("_sigma", "_sd", "_std", "_err")


class DataError(ValueError):
    """Raised when an experimental data table cannot be interpreted."""


@dataclass(frozen=True)
class Measurement:
    """One variable's time course within one dataset."""

    variable: str
    times: np.ndarray
    values: np.ndarray
    sigma: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.times.shape != self.values.shape:
            raise DataError(
                f"variable '{self.variable}': {self.times.size} time points but "
                f"{self.values.size} values"
            )
        if self.times.size == 0:
            raise DataError(f"variable '{self.variable}': no usable measurements")
        if self.sigma is not None:
            if self.sigma.shape != self.values.shape:
                raise DataError(f"variable '{self.variable}': sigma has the wrong length")
            if np.any(self.sigma <= 0):
                raise DataError(f"variable '{self.variable}': sigma values must be positive")

    @property
    def n(self) -> int:
        return int(self.times.size)

    @property
    def scale(self) -> float:
        """A representative magnitude, used by ``max_normalized`` scaling."""
        peak = float(np.max(np.abs(self.values)))
        return peak if peak > 0 else 1.0


@dataclass
class ExperimentData:
    """All measurements from a single experiment, tied to one model."""

    id: str
    model: str
    measurements: Dict[str, Measurement]
    weight: float = 1.0
    conditions: Dict[str, float] = None  # type: ignore[assignment]
    initial_conditions: Dict[str, float] = None  # type: ignore[assignment]
    source: Optional[Path] = None

    def __post_init__(self) -> None:
        self.conditions = dict(self.conditions or {})
        self.initial_conditions = dict(self.initial_conditions or {})
        if not self.measurements:
            raise DataError(f"dataset '{self.id}': no measured variables found")

    @property
    def variables(self) -> List[str]:
        return list(self.measurements)

    @property
    def n_points(self) -> int:
        return sum(m.n for m in self.measurements.values())

    def has(self, variable: str) -> bool:
        return variable in self.measurements

    def select(self, variables: Iterable[str]) -> List[Measurement]:
        """Measurements for the requested variables, silently skipping absent ones."""
        return [self.measurements[v] for v in variables if v in self.measurements]

    def time_grid(self, variables: Optional[Iterable[str]] = None) -> np.ndarray:
        """Sorted union of the time points of the requested variables."""
        chosen = self.select(variables) if variables is not None else list(self.measurements.values())
        if not chosen:
            return np.empty(0, dtype=float)
        return np.unique(np.concatenate([m.times for m in chosen]))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"ExperimentData(id={self.id!r}, model={self.model!r}, "
            f"variables={self.variables}, points={self.n_points})"
        )


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def _read_table(spec: DatasetSpec) -> pd.DataFrame:
    if spec.inline is not None:
        lengths = {k: len(v) for k, v in spec.inline.items()}
        if len(set(lengths.values())) > 1:
            raise DataError(f"dataset '{spec.id}': inline columns have unequal lengths {lengths}")
        return pd.DataFrame(spec.inline)

    path = Path(spec.file)  # type: ignore[arg-type]
    sep = "\t" if path.suffix.lower() in (".tsv", ".tab") else ","
    try:
        frame = pd.read_csv(path, sep=sep, comment="#")
    except Exception as exc:
        raise DataError(f"dataset '{spec.id}': could not read {path} ({exc})") from exc
    frame.columns = [str(c).strip() for c in frame.columns]
    return frame


def _sigma_column(frame: pd.DataFrame, variable: str) -> Optional[str]:
    for suffix in SIGMA_SUFFIXES:
        candidate = f"{variable}{suffix}"
        if candidate in frame.columns:
            return candidate
    return None


def _clean(times: np.ndarray, values: np.ndarray, sigma: Optional[np.ndarray]):
    """Drop rows where the time or the measured value is missing."""
    keep = np.isfinite(times) & np.isfinite(values)
    if sigma is not None:
        sigma = sigma[keep]
    times, values = times[keep], values[keep]
    order = np.argsort(times, kind="stable")
    if sigma is not None:
        sigma = sigma[order]
    return times[order], values[order], sigma


def _from_wide(frame: pd.DataFrame, spec: DatasetSpec) -> Dict[str, Measurement]:
    tcol = spec.time_column
    if tcol not in frame.columns:
        raise DataError(
            f"dataset '{spec.id}': no time column '{tcol}'; columns are {list(frame.columns)}"
        )
    times_all = pd.to_numeric(frame[tcol], errors="coerce").to_numpy(dtype=float)

    sigma_cols = {
        c for v in frame.columns for c in [_sigma_column(frame, v)] if c is not None
    }
    measurements: Dict[str, Measurement] = {}
    for column in frame.columns:
        if column == tcol or column in sigma_cols:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        scol = _sigma_column(frame, column)
        sigma = (
            pd.to_numeric(frame[scol], errors="coerce").to_numpy(dtype=float)
            if scol is not None
            else None
        )
        t, v, s = _clean(times_all, values, sigma)
        if t.size == 0:
            continue
        if s is not None and not np.all(np.isfinite(s)):
            s = None
        measurements[column] = Measurement(column, t, v, s)
    return measurements


def _from_long(frame: pd.DataFrame, spec: DatasetSpec) -> Dict[str, Measurement]:
    tcol = spec.time_column
    required = (tcol, "variable", "value")
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise DataError(
            f"dataset '{spec.id}': long format needs column(s) {missing}; "
            f"columns are {list(frame.columns)}"
        )
    measurements: Dict[str, Measurement] = {}
    for name, group in frame.groupby("variable", sort=False):
        times = pd.to_numeric(group[tcol], errors="coerce").to_numpy(dtype=float)
        values = pd.to_numeric(group["value"], errors="coerce").to_numpy(dtype=float)
        sigma = None
        if "sigma" in group.columns:
            sigma = pd.to_numeric(group["sigma"], errors="coerce").to_numpy(dtype=float)
        t, v, s = _clean(times, values, sigma)
        if t.size == 0:
            continue
        if s is not None and not np.all(np.isfinite(s)):
            s = None
        measurements[str(name)] = Measurement(str(name), t, v, s)
    return measurements


def load_dataset(spec: DatasetSpec) -> ExperimentData:
    """Read one dataset described by a :class:`~pymodest.config.DatasetSpec`."""
    frame = _read_table(spec)
    if frame.empty:
        raise DataError(f"dataset '{spec.id}': the table is empty")
    builder = _from_wide if spec.format == "wide" else _from_long
    measurements = builder(frame, spec)
    if not measurements:
        raise DataError(f"dataset '{spec.id}': no variable columns with usable values")
    return ExperimentData(
        id=spec.id,
        model=spec.model,
        measurements=measurements,
        weight=spec.weight,
        conditions=dict(spec.conditions),
        initial_conditions=dict(spec.initial_conditions),
        source=spec.file,
    )


def load_datasets(specs: Sequence[DatasetSpec]) -> Dict[str, ExperimentData]:
    """Read every dataset in a study, keyed by dataset id."""
    return {spec.id: load_dataset(spec) for spec in specs}


def write_dataset(
    data: Mapping[str, Measurement] | ExperimentData,
    path: Path,
    time_column: str = "time",
) -> Path:
    """Write measurements to a wide CSV (used by the example data generator)."""
    measurements = data.measurements if isinstance(data, ExperimentData) else dict(data)
    grid = np.unique(np.concatenate([m.times for m in measurements.values()]))
    frame = pd.DataFrame({time_column: grid})
    for name, m in measurements.items():
        lookup = dict(zip(m.times, m.values))
        frame[name] = [lookup.get(t, np.nan) for t in grid]
        if m.sigma is not None:
            slookup = dict(zip(m.times, m.sigma))
            frame[f"{name}_sigma"] = [slookup.get(t, np.nan) for t in grid]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path
