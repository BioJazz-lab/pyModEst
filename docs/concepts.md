# Concepts

## The problem

Fitting a kinetic model means searching parameter space for values that
reproduce measurements. A model with thirty parameters means a thirty-
dimensional search, and global optimizers degrade badly as dimension grows —
the volume to be searched grows exponentially while the budget does not.

Biology usually offers a way out that pure mathematics does not: you often know
*which measurements constrain which parameters*. Uptake kinetics are visible in
the upstream metabolite. A feedback constant is visible in the species doing
the inhibiting. That knowledge is what pyModEst asks you to write down.

## Modules

A **module** is a group of parameters fitted together, plus the measured
variables that score them. The modules must partition the parameter set: every
fitted parameter belongs to exactly one module, and pyModEst refuses to load a
configuration where they overlap or where one is left out.

```toml
[[modules]]
id = "upstream"
variables = ["A", "B"]      # what scores this module

[[modules.parameters]]
name = "Vmax1"
lower = 0.05
upper = 50.0
scale = "log"
```

## The loop

```
theta <- initial values
repeat up to max_loops times:
    for each module m (in the configured order):
        theta[m] <- argmin cost_m(theta[m] ; theta[not m] held fixed)
    stop when the total cost stops improving
```

Each module fit sees only a handful of free parameters, so a global optimizer
can solve it properly. Two modules of four parameters each are two
four-dimensional searches, not one eight-dimensional one.

## What this costs you

This is the part worth understanding before trusting a result.

**The total cost is not monotone.** Each module minimises *its own* objective,
not the joint one. Modules are coupled through the shared model, so a step that
helps one can hurt another. In the shipped example, four of the module fits
raise the total cost, and one loop makes it 11% worse before later loops
recover. This is expected, not a bug — pyModEst tracks the best parameter set
across the whole run and returns that, never simply the last one.

**The fixed point is not the joint optimum.** Because each module optimises a
different objective, the point the loop converges to differs slightly from what
a joint least-squares fit would find. On the shipped example the loop reaches a
cost of 0.0604 where the generating parameters score 0.0469. That gap is a
property of the method, not a failure to converge — three independent
optimizers agree on the same answer.

**A bad partition fights itself.** If two parameters are only identifiable
together, splitting them across modules means each fit keeps undoing the
other's work. The symptom is a total cost that oscillates instead of settling;
`loop_summary.csv` is where you see it. The fix is to merge those modules.

## Choosing a partition

- Put a parameter in the module whose variables respond to it most directly.
- Parameters that trade off against each other — a `Vmax`/`Km` pair, a
  synthesis/degradation pair — belong in the same module.
- Give a module enough variables to constrain its parameters, but not so many
  that it becomes the joint problem again.
- Coupled modules need more loops, not a different partition. The example's
  feedback strain makes upstream variables depend on a downstream parameter,
  and the repeated loops are exactly what resolves that.

## Several models, one parameter set

Studies rarely involve a single strain. pyModEst fits one shared parameter set
against any number of models and datasets at once. A parameter that exists in
only some models is written only to those, so it is identified only by their
data — a feedback constant present in one mutant is constrained by that
mutant's measurements and nothing else.

## When not to use this

If your model has few parameters, or you have no idea which measurements
constrain which parameters, fit jointly instead — declare one module containing
everything. The loop then reduces to a single global optimization, which is the
right thing to do when the structure this method exploits is not there.
