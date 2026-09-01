# Runnable examples

Each script stands alone and prints what it demonstrates. They are the source
of the output quoted in the documentation, so they are also how that output is
kept honest — `run_all.py` re-runs everything.

| script | shows |
| ------ | ----- |
| `01_first_fit.py` | a complete study in one file: model, data, modules, fit, recovery |
| `02_shared_parameters.py` | one parameter set across several models and datasets |
| `03_data_and_objective.py` | data layouts, measurement error, ragged schedules, residual scaling |
| `04_optimizers.py` | the five backends compared on an easy and a hard problem; adding your own |
| `05_loop_control.py` | module order, stopping rules, acceptance policy, fixed parameters, callbacks |

```bash
uv run python docs/examples/01_first_fit.py     # one of them
uv run python docs/examples/run_all.py          # all of them
uv run python docs/examples/run_all.py --show   # all, with output
```

`01` is deliberately self-contained and is the one to copy from. The rest
import shared setup from `_toy.py` so each stays focused on its subject.
