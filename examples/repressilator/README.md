# Example: a repressilator, and where divide-and-conquer stops helping

```
gene1 -| gene2 -| gene3 -| gene1
```

The classic Elowitz & Leibler (2000) 3-gene ring oscillator, parameterised
per-gene rather than symmetrically. Each gene produces mRNA translated into
protein, and each protein represses the *next* gene's transcription. At the
parameters in `models/repressilator.ant`, the ring has no stable fixed
point and instead settles onto a limit cycle: sustained, self-generated
oscillation in every species.

One Antimony model backs two study entries:

- `ring` -- the intact loop.
- `open` -- gene 1's repression threshold `K1` fixed to a huge constant via
  a model-level override (`[models.overrides]` in the config), which
  flattens gene 1's Hill term to a near-constant rate and turns the ring
  into a feed-forward chain that relaxes to a fixed point instead of
  oscillating. `K1` is therefore identifiable only from the `ring` data; the
  `open` strain constrains everything else.

The 11 parameters are partitioned into four modules:

| module | parameters | scored on |
| --- | --- | --- |
| `gene1` | `a1`, `K1` | M1, P1 |
| `gene2` | `a2`, `K2` | M2, P2 |
| `gene3` | `a3`, `K3` | M3, P3 |
| `kinetics` | `a0`, `n`, `dm`, `dp`, `beta` (shared by every gene) | all six species |

## This is a companion to `../two_module_pathway`

That example is the case where divide-and-conquer fitting helps: a
directional pathway with one feedback arrow, where holding downstream
parameters fixed while fitting upstream ones is a good local approximation.
This example is the case where it does *not* help, and both configs are
here specifically so you can run the comparison yourself:

```bash
uv run pymodest fit config_modular.toml   # 4 modules -- ~35-40s
uv run pymodest fit config_global.toml    # 1 module, everything jointly -- ~20-25s
```

## What to expect

The data carry 4% proportional noise; the generating parameters score a
noise floor of **0.0584**. Both configs start from the same point
(`cost 7.19`) with identical optimizer settings (differential evolution,
`popsize=15, maxiter=60`), 8 max loops, `patience=2`.

| | `config_modular.toml` | `config_global.toml` |
| --- | --- | --- |
| final cost / noise floor | **~5.3-5.8x** | **~0.82x** (at the floor) |
| function evaluations | ~65,000 | ~33,000 |
| wall time | ~37s | ~22s |
| loops to converge | 8 (never fully converges) | 3 |

Global fitting wins on both fidelity and efficiency, consistently across
random seeds (verified across 6). Modular fitting is not merely slow to
converge here -- it converges (by the estimator's own stopping rule) to a
real local optimum well above the noise floor, confirmed by running with
`max_loops=20` (unchanged past loop ~9) and by trying
`fitting.accept = "total"` in `config_modular.toml` (converges even faster
to an *even worse* plateau).

**Why**: in the linear pathway example, coupling is one-directional, so
"hold downstream fixed, fit upstream" is a reasonable local approximation.
In this ring, oscillation phase, period and amplitude are one emergent
property of all three genes *together* -- no module's own species can be
fit correctly while the other two modules are still wrong, because they are
mutually phase-locked. Differential evolution's population size scales
with dimension either way (`popsize x sum-of-module-dims` here equals
`popsize x 11`, the same total population as the global search), so the
efficiency gap comes from needing far fewer total sweeps (3 vs. 8-9), not
from smaller per-sweep populations -- a nuance worth keeping in mind before
assuming "more modules = cheaper."

## Regenerating the data

```bash
uv run python generate_data.py
```

Edit `TRUE` or the noise level at the top of that script to make the
problem easier or harder.
