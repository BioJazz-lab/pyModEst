# Example: a six-tier cascade, and where divide-and-conquer ties global fitting

```
L -> R -| -> X1 -| -> X2 -| -> M -| -> P -| -> M2 -| -> P2
   (receptor)  (kinase1) (kinase2)  (mRNA)  (protein) (mRNA2)  (protein2)
```

A strictly feed-forward chain -- no feedback loop onto any earlier tier --
combining receptor kinetics, a Goldbeter-Koshland zero-order ultrasensitive
switch, and cooperative (Hill) gene activation across six tiers. Tier
response times are separated by construction, roughly `tau ~ 0.1, 1, 2, 10,
50, 150, 500` (a ~5000-fold span): by the time a downstream tier's dynamics
become significant, the tier upstream of it has already settled near its
own steady value. Two experiments (`data/`) vary the stimulus dose `L`
(2.0 and 0.5); both measure all seven species on one log-spaced time grid
spanning the full range of tiers.

The 24 parameters are partitioned one module per tier:

| module | parameters | tier (tau) |
| --- | --- | --- |
| `receptor` | `k1on`, `k1off` | 1 (~0.1) |
| `kinase1` | `k2cat`, `Km2a`, `k2f`, `Km2b` | 2 (~1) |
| `kinase2` | `k3cat`, `Km3a`, `k3f`, `Km3b` | 3 (~2) |
| `transcription1` | `a0`, `aM`, `KM`, `n`, `dM` | 4 (~10) |
| `protein1` | `beta`, `dP` | 5 (~50) |
| `tier6` | `b0`, `bM`, `KM2`, `n2`, `dM2`, `beta2`, `dP2` | 6 (~150-500) |

## This is a companion to `../repressilator`

That example is the case where divide-and-conquer fitting loses decisively,
because a ring's circular coupling means no module's local fit approximates
its role in the joint optimum. This cascade is the opposite structural
extreme -- purely hierarchical, timescale-separated coupling -- and the two
methods land in a statistical tie:

```bash
uv run pymodest fit config_modular.toml   # 6 modules
uv run pymodest fit config_global.toml    # 1 module, everything jointly
```

## What to expect

The data carry 4% proportional noise. Both configs start from the same
point with identical optimizer settings (differential evolution,
`popsize=15, maxiter=60`), 8 max loops, `patience=2`.

| | `config_modular.toml` | `config_global.toml` |
| --- | --- | --- |
| final cost / noise floor | ~0.936 | ~0.935 |
| function evaluations | ~73,000 | ~75,000 |
| wall time | ~45s | ~54s |

Both land within measurement noise of the noise floor on every seed tried.
Modular is modestly faster in wall time (~16%) but *not* dramatically
cheaper in evaluation count -- worth noting, since it's easy to assume
"smaller sub-problems must mean fewer evaluations." Differential
evolution's population size scales with dimension; the summed population
across one sweep of modules of dimension 2, 4, 4, 5, 2, 7 equals
`popsize x 24`, the same as the single 24-dimensional global search.
Splitting into modules doesn't reduce a single sweep's cost by itself --
the real source of an efficiency edge (see `../repressilator`) is needing
fewer *sweeps*, and here both methods need a similarly small number (3-4),
so raw counts converge too.

**Objective setting worth noting**: this config uses
`scaling = "max_normalized"`, not `"relative"`. The model's seven species
span magnitudes from ~0.1 (`X2`) to ~800 (`P2`); a single relative-scaling
`epsilon` can't serve all of them without blowing up near-zero points on
the small-magnitude species. `max_normalized` divides each variable's
residuals by that variable's own peak observed value instead, which is the
right choice whenever a model's species span more than a couple of orders
of magnitude relative to each other.

## Regenerating the data

```bash
uv run python generate_data.py
```

Edit `TRUE` or the noise level at the top of that script to make the
problem easier or harder.
