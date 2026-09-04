"""Generate the synthetic measurements used by this example.

Simulates the shared Antimony model at two stimulus doses, samples time
points log-spaced from 0.01 to 6000 (so both the fastest tier, tau~0.1, and
the slowest, tau~500, are resolved in the same trace), adds proportional
noise, and writes one CSV per dose plus the generating parameters.

    python generate_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pymodest.config import ModelSpec, SimulationSpec
from pymodest.model import SimulationModel

HERE = Path(__file__).parent

# the values the fits are expected to recover
TRUE = {
    "k1on": 3.0, "k1off": 1.5,
    "k2cat": 1.2, "Km2a": 0.15, "k2f": 1.0, "Km2b": 0.15,
    "k3cat": 0.35, "Km3a": 0.2, "k3f": 0.28, "Km3b": 0.2,
    "a0": 0.05, "aM": 1.0, "KM": 0.09, "n": 2.5, "dM": 0.08,
    "beta": 0.6, "dP": 0.02,
    "b0": 0.02, "bM": 1.0, "KM2": 80.0, "n2": 2.0, "dM2": 0.005,
    "beta2": 0.01, "dP2": 0.0025,
}

SPECIES = ("R", "X1", "X2", "M", "P", "M2", "P2")
TIMES = np.concatenate(([0.0], np.geomspace(0.01, 6000.0, 26)))

EXPERIMENTS = [
    # id,    L (stimulus dose),  noise
    ("high", 2.0, 0.04),
    ("low", 0.5, 0.04),
]


def main() -> None:
    rng = np.random.default_rng(20260903)
    antimony = (HERE / "models" / "cascade.ant").read_text()
    model = SimulationModel(ModelSpec(id="cascade", antimony=antimony), SimulationSpec())

    (HERE / "data").mkdir(exist_ok=True)
    for exp_id, dose, noise in EXPERIMENTS:
        trace = model.simulate(TIMES, parameters=TRUE, conditions={"L": dose}, use_cache=False)
        frame = pd.DataFrame({"time": TIMES})
        for variable in SPECIES:
            clean = trace[variable]
            sigma = noise * np.maximum(np.abs(clean), 0.05 * np.max(np.abs(clean)))
            frame[variable] = np.maximum(clean + rng.normal(0.0, sigma), 0.0)
            frame[f"{variable}_sigma"] = sigma
        path = HERE / "data" / f"cascade_{exp_id}.csv"
        frame.round(6).to_csv(path, index=False)
        print(f"wrote {path.relative_to(HERE)}  ({len(TIMES)} time points, L={dose})")

    truth = HERE / "true_parameters.toml"
    truth.write_text(
        "# values used to generate the synthetic data in data/\n"
        "[parameters]\n" + "".join(f"{k} = {v}\n" for k, v in TRUE.items())
    )
    print(f"wrote {truth.relative_to(HERE)}")


if __name__ == "__main__":
    main()
