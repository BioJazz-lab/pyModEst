"""Generate the synthetic measurements used by this example.

Simulates both models at the known true parameter values, samples a handful of
time points, adds proportional noise, and writes one CSV per experiment. Run it
to regenerate the data:

    python generate_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pymodest.config import ModelSpec, SimulationSpec
from pymodest.model import SimulationModel

HERE = Path(__file__).parent

# the values the fit is expected to recover
TRUE = {
    "Vmax1": 2.4,
    "Km1": 3.0,
    "k2": 0.55,
    "Vmax3": 1.30,
    "Km3": 0.80,
    "k4": 0.22,
    "Ki": 1.60,
}

EXPERIMENTS = [
    # id,        model,      S0,   measured variables,     noise
    ("wt_high",  "wt",       12.0, ("A", "B", "C"), 0.04),
    ("wt_low",   "wt",        3.0, ("A", "B", "C"), 0.04),
    ("fb_high",  "feedback", 12.0, ("A", "B", "C"), 0.04),
]

TIMES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 13.0, 16.0, 20.0])


def main() -> None:
    rng = np.random.default_rng(20260901)
    models = {
        name: SimulationModel(
            ModelSpec(id=name, antimony=(HERE / "models" / f"{name}.ant").read_text()),
            SimulationSpec(),
        )
        for name in ("wt", "feedback")
    }

    (HERE / "data").mkdir(exist_ok=True)
    for exp_id, model_id, s0, variables, noise in EXPERIMENTS:
        model = models[model_id]
        params = {k: v for k, v in TRUE.items() if model.has_parameter(k)}
        trace = model.simulate(
            TIMES, parameters=params, initial_conditions={"S": s0}, use_cache=False
        )
        frame = pd.DataFrame({"time": TIMES})
        for variable in variables:
            clean = trace[variable]
            sigma = noise * np.maximum(np.abs(clean), 0.05 * np.max(np.abs(clean)))
            frame[variable] = np.maximum(clean + rng.normal(0.0, sigma), 0.0)
            frame[f"{variable}_sigma"] = sigma
        path = HERE / "data" / f"{exp_id}.csv"
        frame.round(6).to_csv(path, index=False)
        print(f"wrote {path.relative_to(HERE)}  ({len(TIMES)} time points)")

    truth = HERE / "true_parameters.toml"
    truth.write_text(
        "# values used to generate the synthetic data in data/\n[parameters]\n"
        + "".join(f"{k} = {v}\n" for k, v in TRUE.items())
    )
    print(f"wrote {truth.relative_to(HERE)}")


if __name__ == "__main__":
    main()
