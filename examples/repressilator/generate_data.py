"""Generate the synthetic measurements used by this example.

Simulates the shared Antimony model at two conditions -- the intact ring and
the "open" variant (gene 1's repression fixed off via a model override) --
samples time points, adds proportional noise, and writes one CSV per
experiment plus the generating parameters. Run it to regenerate the data:

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
    "a0": 0.22, "n": 2.1, "dm": 0.9, "dp": 0.18, "beta": 0.24,
    "a1": 200.0, "K1": 1.1,
    "a2": 240.0, "K2": 0.85,
    "a3": 180.0, "K3": 1.3,
}

SPECIES = ("M1", "P1", "M2", "P2", "M3", "P3")

EXPERIMENTS = [
    # id,     open_gene1, times,                                  noise
    ("ring", False, np.linspace(0.0, 114.0, 20), 0.04),
    ("open", True, np.linspace(0.0, 50.0, 14), 0.04),
]


def main() -> None:
    rng = np.random.default_rng(20260902)
    antimony = (HERE / "models" / "repressilator.ant").read_text()
    model = SimulationModel(ModelSpec(id="repressilator", antimony=antimony), SimulationSpec())

    (HERE / "data").mkdir(exist_ok=True)
    for exp_id, open_gene1, times, noise in EXPERIMENTS:
        params = dict(TRUE)
        if open_gene1:
            params["K1"] = 1e8  # matches the [models.overrides] used for the "open" model
        trace = model.simulate(times, parameters=params, use_cache=False)
        frame = pd.DataFrame({"time": times})
        for variable in SPECIES:
            clean = trace[variable]
            sigma = noise * np.maximum(np.abs(clean), 0.05 * np.max(np.abs(clean)))
            frame[variable] = np.maximum(clean + rng.normal(0.0, sigma), 0.0)
            frame[f"{variable}_sigma"] = sigma
        path = HERE / "data" / f"repressilator_{exp_id}.csv"
        frame.round(6).to_csv(path, index=False)
        print(f"wrote {path.relative_to(HERE)}  ({len(times)} time points)")

    truth = HERE / "true_parameters.toml"
    truth.write_text(
        "# values used to generate the synthetic data in data/\n"
        "[parameters]\n" + "".join(f"{k} = {v}\n" for k, v in TRUE.items())
    )
    print(f"wrote {truth.relative_to(HERE)}")


if __name__ == "__main__":
    main()
