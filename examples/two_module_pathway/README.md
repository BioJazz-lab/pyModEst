# Example: a linear pathway fitted in two modules

```
S --J1--> A --J2--> B --J3--> C --J4--> out
```

Two models share one kinetic parameter set:

- `models/wt.ant` — the wild-type pathway.
- `models/feedback.ant` — the same pathway with product inhibition of uptake by
  C. It shares every kinetic parameter with the wild type and adds one of its
  own, `Ki`, which only feedback-strain data can identify.

Three experiments (`data/`) measure A, B and C: the wild type at saturating
(S=12) and sub-saturating (S=3) substrate, and the feedback strain at S=12. The
sub-saturating experiment is what pins down `Km1`.

The parameters are partitioned into two modules:

| module | parameters | scored on |
| ------ | ---------- | --------- |
| `upstream` | `Vmax1`, `Km1`, `k2` | A, B |
| `downstream` | `Vmax3`, `Km3`, `k4`, `Ki` | C |

The feedback strain couples them: C acts back on the uptake that produces A, so
the upstream fit depends on a downstream parameter. Resolving that coupling is
what the repeated loops are for.

## Running it

```bash
pymodest validate config.toml     # check everything lines up
pymodest fit config.toml          # ~7 seconds
```

Results land in `results/`. To simulate at the parameters that generated the
data instead:

```bash
pymodest simulate config.toml --parameters true_parameters.toml
```

## What to expect

The data carry 4% proportional noise, so the generating parameters themselves
score a total cost of **0.0469** — the noise floor. Starting from 237, the fit
converges to about **0.060** in five loops.

`Ki` and `k4` come back close to the truth. The `Vmax`/`Km` pairs stay
correlated, as they do in any fit at this noise level: several combinations
describe the data almost equally well. Running with `--optimizer scatter_search`
or `--optimizer particle_swarm` reaches the same answer, which is the useful
check that the estimator has actually converged rather than stopped early.

## Regenerating the data

```bash
python generate_data.py
```

Edit `TRUE`, `EXPERIMENTS` or the noise level at the top of that script to make
the problem easier or harder.
