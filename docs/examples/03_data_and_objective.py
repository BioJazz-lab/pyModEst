"""How measurements are read, and how residuals are scaled.

Two independent things, shown together because they are what connect a model
to data: the shape your measurements arrive in, and how differences between
simulation and measurement are turned into one number.

    uv run python docs/examples/03_data_and_objective.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from pymodest.config import DatasetSpec, ObjectiveSpec
from pymodest.data import Measurement, load_dataset
from pymodest.objective import _scale_residuals

tmp = Path(tempfile.mkdtemp())

# --------------------------------------------------------------------------
# data layouts
# --------------------------------------------------------------------------
print("=" * 66)
print("DATA LAYOUTS -- three ways to say the same thing")
print("=" * 66)

wide = tmp / "wide.csv"
pd.DataFrame({"time": [0.0, 1.0, 2.0], "A": [0.0, 1.0, 1.8], "B": [0.0, 0.2, 0.7]}
             ).to_csv(wide, index=False)

long = tmp / "long.csv"
pd.DataFrame({
    "time": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
    "variable": ["A", "A", "A", "B", "B", "B"],
    "value": [0.0, 1.0, 1.8, 0.0, 0.2, 0.7],
}).to_csv(long, index=False)

layouts = {
    "wide file":  DatasetSpec(id="d", model="m", file=wide),
    "long file":  DatasetSpec(id="d", model="m", file=long, format="long"),
    "inline":     DatasetSpec(id="d", model="m", inline={
        "time": [0.0, 1.0, 2.0], "A": [0.0, 1.0, 1.8], "B": [0.0, 0.2, 0.7]}),
}
for label, spec in layouts.items():
    data = load_dataset(spec)
    print(f"  {label:<11} variables={data.variables}  points={data.n_points}  "
          f"A={np.round(data.measurements['A'].values, 2).tolist()}")

# --------------------------------------------------------------------------
# measurement error
# --------------------------------------------------------------------------
print()
print("=" * 66)
print("MEASUREMENT ERROR -- a matching _sigma column is picked up automatically")
print("=" * 66)

with_sigma = tmp / "sigma.csv"
pd.DataFrame({"time": [0.0, 1.0, 2.0], "A": [0.1, 1.0, 1.8],
              "A_sigma": [0.05, 0.10, 0.20]}).to_csv(with_sigma, index=False)
data = load_dataset(DatasetSpec(id="d", model="m", file=with_sigma))
print(f"  variables:  {data.variables}   <- A_sigma is not a variable")
print(f"  sigma:      {data.measurements['A'].sigma.tolist()}")
print("  suffixes accepted: _sigma, _sd, _std, _err")

# --------------------------------------------------------------------------
# ragged schedules
# --------------------------------------------------------------------------
print()
print("=" * 66)
print("RAGGED SCHEDULES -- variables need not be measured at the same times")
print("=" * 66)

ragged = tmp / "ragged.csv"
pd.DataFrame({"time": [0.0, 1.0, 2.0, 3.0],
              "A": [0.0, np.nan, 1.8, 2.4],
              "B": [0.0, 0.2, np.nan, 0.9]}).to_csv(ragged, index=False)
data = load_dataset(DatasetSpec(id="d", model="m", file=ragged))
for name, m in data.measurements.items():
    print(f"  {name}: {m.n} points at t={m.times.tolist()}")
print("  missing values are dropped per variable, not per row")

# --------------------------------------------------------------------------
# residual scaling
# --------------------------------------------------------------------------
print()
print("=" * 66)
print("RESIDUAL SCALING -- the same mismatch, four ways")
print("=" * 66)

# Two variables on very different scales, each simulated 10% too high.
observed = Measurement("x", np.array([0.0, 1.0]), np.array([1.0, 100.0]),
                       np.array([0.5, 5.0]))
simulated = np.array([1.1, 110.0])

print(f"  observed   {observed.values.tolist()}   (sigma {observed.sigma.tolist()})")
print(f"  simulated  {simulated.tolist()}")
print(f"  both are 10% too high\n")
print(f"  {'scaling':<16}{'residuals':>26}   what it does")
rows = [
    ("absolute", "plain difference; the large variable dominates"),
    ("relative", "divided by |observed|; both count equally  <- default"),
    ("sigma", "divided by measurement error; chi-square"),
    ("max_normalized", "divided by that variable's peak value"),
]
for scaling, note in rows:
    r = _scale_residuals(simulated, observed, ObjectiveSpec(scaling=scaling, epsilon=1e-9))
    print(f"  {scaling:<16}{str(np.round(r, 4).tolist()):>26}   {note}")

print("\n  With 'absolute', the second point contributes 10000x the first and the")
print("  fit effectively ignores the small variable. That is why 'relative' is")
print("  the default. Raise `epsilon` above the noise floor when measurements")
print("  approach zero, or a near-zero observation will dominate instead.")
