# Contributing

New languages and new variants are very welcome. The value of this benchmark is
entirely in its fairness, so most of these rules are about keeping every
implementation solving *exactly* the same problem.

## The rules that keep the comparison honest

1. **Same algorithm, same arithmetic.** Six-stage fifth-order Runge–Kutta,
   `y0 = (2, 2, 0, -1)`, `h = 0.2`, `T = 10000`, double precision throughout.
   Copy the stage expressions from an existing implementation rather than
   rederiving them — in particular `h` multiplies the *stage arguments*
   (`yn + h*9*k1/4 - ...`), not the stages, and the right-hand side uses the
   two-`sincos` form with trigonometric identities. Both choices are worth
   5–8% and must be identical everywhere or the numbers mean nothing.
2. **No relaxed floating point.** No `-ffast-math`, `@fastmath`, `-Ofast`, or
   reduced precision. Exact-algebra rewrites are fine; approximations are not.
3. **Single threaded.** No OpenMP, no `Threads.@threads`, no BLAS threads. The
   problem is sequential by construction.
4. **No `-march=native` or other machine-specific flags** in the committed
   build. (It makes every implementation here *slower* — see the README.)
5. **One file, idiomatic, readable.** Half the point of this repository is what
   the code looks like. Write what a competent user of the language would
   actually write, and let the file stand on its own.

## Output format

Print one timing per line, ten repetitions, and nothing else that looks like a
number. Any of these is fine — `scripts/bench.py` scrapes them all:

```
Time 0.010502 seconds.          C, C++, Fortran
  0.010424 seconds (5 allocations: 1.527 MiB)   Julia @time
Elapsed time is 0.103766 seconds.               MATLAB tic/toc
0.014466047286987305            a bare number, as the Python files print
```

For anything with a JIT, compile once before the timing loop and say so in a
comment.

## Naming and layout

Sources live in `c/`, `cpp/`, `fortran/`, `julia/`, `matlab/`, `python/`. The
straightforward version of a language is `pendulum.<ext>`; every variant adds
one descriptive word, as in `pendulum_views.jl` or `pendulum_colref.cpp`.

## Before you open a pull request

```sh
make validate      # every implementation must agree to ~15 digits at T = 10
make bench-quick   # sanity-check the timing
```

`make validate` is the important one. It rewrites `T = 10000` to `T = 10` in a
temporary copy of each file, appends a checksum print, and compares: 50 steps is
long enough that a wrong coefficient shows in the first digit, and short enough
that chaos has not yet amplified last-bit differences. If you add a file, add it
to the tables in `scripts/bench.py` and `scripts/validate.py` so it is covered.

Then add a row to the timing table in `README.md`. Please say what hardware and
compiler you measured on, and if you are reporting new numbers for the existing
implementations, re-measure all of them together on the same machine — the
absolute values move by 30% with the CPU governor alone.

## Reporting a result that contradicts the README

Very welcome, and please include the machine, compiler versions, CPU governor,
and whether the runs were pinned to a core. Several of the orderings here are
within a few percent, and at least two of them flip depending on the clock.
