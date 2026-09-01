"""One parameter set, several models, several datasets.

The usual situation in practice: a wild type and a mutant are described by
related models that share most of their kinetics. pyModEst fits one parameter
set against all of them at once, and a parameter that exists in only some
models is written only to those -- so it is identified only by their data.

    uv run python docs/examples/02_shared_parameters.py
"""

from __future__ import annotations

from pymodest import fit
from pymodest.config import DatasetSpec, ModelSpec, StudyConfig
from pymodest.objective import Problem

from _toy import FEEDBACK, TRUE, WT, fitting, modules, report, simulate_truth

# Two models. `feedback` has an extra parameter, Ki, that `wt` knows nothing
# about; everything else is shared between them.
config = StudyConfig(
    name="shared-parameters",
    models=[ModelSpec(id="wt", antimony=WT), ModelSpec(id="fb", antimony=FEEDBACK)],
    datasets=[
        DatasetSpec(id="wt_exp", model="wt", inline=simulate_truth(WT),
                    initial_conditions={"S": 10.0}),
        DatasetSpec(id="fb_exp", model="fb", inline=simulate_truth(FEEDBACK),
                    initial_conditions={"S": 10.0}),
        # the same model under a different starting condition is just another
        # dataset -- this one is what pins down the saturating uptake
        DatasetSpec(id="wt_low", model="wt", inline=simulate_truth(WT, s0=2.0),
                    initial_conditions={"S": 2.0}),
    ],
    modules=modules(with_ki=True),
    fitting=fitting(max_loops=8),
)
config.validate()

problem = Problem(config)
wt, fb = problem.models["wt"], problem.models["fb"]

print("which model has which parameter")
print(f"{'parameter':<10}{'wt':>6}{'feedback':>10}")
for name in ("Vmax1", "Km1", "k2", "k3", "Ki"):
    print(f"{name:<10}{str(wt.has_parameter(name)):>6}{str(fb.has_parameter(name)):>10}")

# The shared vector is filtered per model on the way into the simulator, so a
# model is never handed a parameter it does not have.
print("\nparameters actually written to each model:")
print(f"  wt       {sorted(problem.parameters_for(wt, TRUE))}")
print(f"  feedback {sorted(problem.parameters_for(fb, TRUE))}")

print("\ndatasets, each attached to one model:")
for dataset in config.datasets:
    print(f"  {dataset.id:<8} model={dataset.model:<4} S0={dataset.initial_conditions['S']}")

result = fit(config)
print(f"\n{report(result)}")
print(f"\n{'parameter':<10}{'true':>8}{'estimate':>11}")
for name in sorted(TRUE):
    print(f"{name:<10}{TRUE[name]:>8.3f}{result.parameters[name]:>11.4f}")
print("\nKi is recovered from the feedback dataset alone -- no other model has it.")
