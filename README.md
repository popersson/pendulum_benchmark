# Pendulum benchmark

The same double-pendulum integration written in C, C++, Fortran, Julia, MATLAB and
Python, as a way of comparing both **speed** and **beauty** on a problem that resists
the usual tricks.

<p align="center">
  <img src="pendulum.gif" width="360" alt="double pendulum animation">
</p>

Every implementation solves an identical problem with identical arithmetic, and
`make validate` proves it: every one of them agrees to fifteen digits. The interesting result
is not that the compiled languages are fast — it is *which* of them are, by how little,
and what the beautiful versions cost.

## The problem

A double pendulum: two rigid links of length 1 carrying a unit mass at each joint, in
unit gravity. The configuration is given by the two angles $\theta_1,\theta_2$ measured
from straight down, and the equations of motion are

$$
\theta_1'' = \frac{-3\sin\theta_1 - \sin(\theta_1-2\theta_2) - 2\sin(\theta_1-\theta_2)\left(\theta_2'^2 + \theta_1'^2\cos(\theta_1-\theta_2)\right)}{3-\cos(2\theta_1-2\theta_2)},
$$

$$
\theta_2'' = \frac{2\sin(\theta_1-\theta_2)\left(2\theta_1'^2 + 2\cos\theta_1 + \theta_2'^2\cos(\theta_1-\theta_2)\right)}{3-\cos(2\theta_1-2\theta_2)}.
$$

Introducing the angular velocities $\omega_1=\theta_1'$, $\omega_2=\theta_2'$ turns this
into a first-order system $\dot{\boldsymbol y} = \boldsymbol f(\boldsymbol y)$ for the
four-component state $\boldsymbol y = (\theta_1,\theta_2,\omega_1,\omega_2)$. That
function is `fpend` in every file here; the animation above is the actual trajectory it
produces.

The benchmark integrates it with a **six-stage, fifth-order explicit Runge–Kutta
method** — the one with weights $(17,0,100,2,-50,75)/144$ — from
$\boldsymbol y_0 = (2,2,0,-1)$ with $h = 0.2$ to $T = 10000$. That is **50 000
sequential steps**, six right-hand-side evaluations each, on a **four-element state**.

Every step depends on the previous one, so there is nothing to vectorize, nothing to
parallelize, and nothing to batch. That is the point: it is the shape of a great deal of
real scientific computing, and it is exactly the shape that array-oriented languages are
worst at. The system is also chaotic, which turns out to be useful — see
[Validation](#validation).

## Layout

```
c/pendulum.c                 explicit loops and an indexing macro
cpp/pendulum.cpp             std::vector<md::sarray>, an array of states
cpp/pendulum_mdxarray.cpp    md::array, columns read and written through page()
cpp/md.h                     mdxarray: the owning and static array types C++ lacks
fortran/pendulum.f90         external subroutines, the classic style
fortran/pendulum_modern.f90  module, associate, abstract interface, pure
julia/pendulum.jl            the straightforward version; allocates per call
julia/pendulum_inline.jl     + StaticArrays, @view and @inline  <- fastest overall
julia/pendulum_svector.jl    the same without @inline, to show what it costs
julia/pendulum_views.jl      preallocated buffers and broadcast; fast but ugly
julia/pendulum_cstyle.jl     Julia written as if it were C
julia/pendulum_tuple.jl      tuples plus broadcast; slow, but far less so on Julia 1.13
matlab/pendulum.m            straightforward MATLAB
python/pendulum.py           straightforward NumPy
python/pendulum_tuple.py     a 4-tuple value type, no NumPy in the hot loop
python/pendulum_numba.py     pendulum.py plus @njit, a six-line diff
python/pendulum_jax.py       jit + lax.fori_loop, the functional rewrite
animate.py                   makes pendulum.gif from python/pendulum.py
scripts/bench.py             runs everything, reports timings
scripts/validate.py          proves they all compute the same trajectory
```

## Building and running

```sh
make                # configure and build every compiled target this machine supports
make bench          # run everything available and print a timing table
make bench-quick    # the same, skipping the implementations that take seconds
make validate       # check that every implementation agrees
make animate        # regenerate pendulum.gif
make help
```

Nothing is required. Missing compilers, missing Julia, missing MATLAB and missing Python
packages are all reported as skipped rather than treated as errors, so `make && make bench`
does something useful on a bare machine with only a C compiler.

`numba` and `jax` are optional: `scripts/bench.py` and `scripts/validate.py` look for
them in the current interpreter and then in `./.venv` and `~/.venv`, and skip those two
files if neither has them. Only NumPy and Matplotlib are needed for the rest.

**`<mdspan>` is the one fussy dependency.** Both C++ variants use `cpp/md.h`
([mdxarray](https://github.com/popersson/mdxarray)), which is built on `std::mdspan`, and as
of today that needs clang with libc++. The Makefile therefore prefers `clang++` when it is
installed; without it both are skipped with a message and everything else still builds.
Force a compiler with `make CXX=g++` to see that happen.

The sources are ordinary single-file programs, so you never have to use the build system:

```sh
gcc -O3 -o pendulum c/pendulum.c -lm
clang++ -O3 -fno-math-errno -std=c++23 -stdlib=libc++ -Icpp -o pendulum cpp/pendulum_mdxarray.cpp
gfortran -O3 -o pendulum fortran/pendulum_modern.f90
julia julia/pendulum_inline.jl
python3 python/pendulum.py
cd matlab && matlab -batch pendulum
```

Each program prints ten timings, one per line. `scripts/bench.py` scrapes those numbers
back out and reports the best and the mean of the last five, which is why the
implementations themselves need no common output format or timing harness — the code
stays as plain as it would be if you wrote it for yourself.

**To get numbers you can trust**, fix the clock and pin to one core:

```sh
sudo cpupower frequency-set -g performance
taskset -c 0 make bench
```

Absolute times move by ~30% with the CPU governor alone, and on a hybrid CPU an unpinned
run may land on an efficiency core and read 40% slow.

## Conclusions

**1. Vanilla Julia, MATLAB and Python are beautiful and expressive** — exactly the code
people want to write. And they are slow, because every one of the 300 000 right-hand-side
evaluations allocates. The ranking is Julia 37 ms, MATLAB 112 ms, Python 1266 ms: Julia is
3× better than MATLAB, MATLAB is 11× better than Python, and Python is **187× off the best
implementation here**. Python is in a class of its own. Julia 1.13 moved this row
noticeably — the same file took 52 ms on 1.12 — so the vanilla-Julia penalty is shrinking
with the runtime, while MATLAB and Python are where they have always been.

**2. Julia can be brought *past* C speed without giving up the look of the code.** Views,
`StaticArrays` and `@inline` are enough; `julia/pendulum_inline.jl` differs from the
beautiful `julia/pendulum.jl` by four marks — `using StaticArrays`, `@view` on the column
read, `SVector` in the return, and `@inline` on `fpend` — and it is the **fastest
implementation here**, 2% ahead of the best C++, 12% ahead of C, 11% ahead of the classic
Fortran, and 13% ahead of the ugly hand-buffered Julia. The 2% is the honest margin: the
committed C++ has been given the one inlining hint clang does not take by itself, without
which it is 6% behind. A 5.4× speedup over the vanilla version for four annotations, with
the integrator loop character-for-character unchanged.
MATLAB and Python have no equivalent. This is the strongest result in the set.

**3. Python can be rescued from outside the language, at a cost.** numba is remarkably
clean — `diff python/pendulum.py python/pendulum_numba.py` is six lines, mostly `@njit` —
and gets to 1.8× of the best, its strongest showing yet. JAX reaches 3.8×, but only after
the entire time loop is rewritten as a functional `fori_loop` with no mutation. Both close a
186× gap to within a small factor, which is the real point; neither is as clean or as fast as
Julia's native tools, and both are a separate compiler you have to keep happy — numba
0.64 will not even import against NumPy 2.5, and measuring this row needed an upgrade to
numba 0.67.

**4. C is ugly and no longer even fast.** Explicit index loops everywhere, macros for 2-D
indexing — and on this machine it is the **slowest of the compiled implementations**, 12%
behind Julia and 5% behind C++, beaten by both Fortrans and by Julia written in C's own
style. It buys nothing over the alternatives any more.

**5. Fortran gets real benefit from native multidimensional arrays and array arithmetic** —
whole-array expressions with no library, no template machinery, and column slices as
first-class values, and it sits at the front of the pack with nothing but `-O3`. Modern
Fortran (module with explicit interfaces, `associate`, an `abstract interface` for the
right-hand side, `pure` throughout) removes almost all of the verbosity of the old style
at no cost in speed. The one concession to performance is that the right-hand side writes
into an out-argument instead of returning an array — `call f(yn + h*k1/5, k2)` rather than
`k2 = f(yn + h*k1/5)` — which is worth 11%. Slightly worse syntax, and it puts modern
Fortran within 3% of the best C++ and 2% ahead of C, with no library, no templates and no
hand-written vector class.

**6. Modern C++ recovers both beauty and speed, but has to bring its own vocabulary.** The
language still has no owning multidimensional array (`std::mdarray` is C++26) and no
arithmetic on `std::array`, so both C++ variants here lean on
[mdxarray](https://github.com/popersson/mdxarray) (`cpp/md.h`): `md::sarray` for the state,
`md::array` for the trajectory. With that vocabulary in place the integrator reads like the
Julia version and runs *faster* than C — second only to Julia. Eigen or a couple of hundred
lines of your own would do the same job. Every other language here has this out of the box;
it is C++'s one real handicap, and it is worth being blunt about.

What that vocabulary has to provide is worth spelling out, because the first port of these
files to mdxarray needed a workaround for each item and the library has since grown all of
them: element-wise construction, so `return {θ1dot, θ2dot, ω1dot, ω2dot};` compiles; a
tuple protocol, so `const auto [θ1, θ2, ω1, ω2] = y;` does; a converting constructor from a
view, so a column can start an expression; and assignment to a *slice* writing through
(`y.page(n + 1) = ...`) while assignment to a named view rebinds, as `std::mdspan` does.
Miss any one of them and the integrator picks up a helper function or a `md::assign` call
at exactly the place where the mathematics should be.

**7. The headline result nobody expects: this benchmark is ~three fifths a `libm`
benchmark.** With the arithmetic equalized across languages, Julia, C, C++ and Fortran land
within 12% of each other — because they all spend ~4.4 ms of their ~7 ms inside the same
`glibc` `sin`/`cos`. They converge because the language is largely not what is being
measured; the visible spread is the ~2.8 ms that is left. That reframes the whole exercise,
and it is the most interesting thing to say about it.

## Timings

Minimum over 8 interleaved rounds, each round taking the best of 10 in-run iterations,
every run pinned with `taskset -c 0`.

The minimum rather than the median, because the implementations that allocate the
trajectory on the heap turn out to be **bimodal**: the same binary, run repeatedly, lands
either near 6.93 ms or near 7.74 ms with nothing in between — 16 runs of one binary split
8/8. It is not the CPU governor, not ASLR (it survives `setarch -R`), and not SMT
contention (deliberately loading the sibling hyperthread costs 33%, a third and much
larger mode). It is a property of the memory the process happens to be given;
`fortran/pendulum_modern.f90`, whose trajectory is a static array, is immune and measures
to ±0.03 ms. Taking the minimum over many runs selects the good layout consistently, which
is the only way the 1–3% differences at the top of this table mean anything.

| Implementation | ms | vs fastest | Notes |
|---|---|---|---|
| **`julia/pendulum_inline.jl`** | **6.69** | **1.00×** | **beautiful *and* fastest — SVector + `@view` + `@inline`** |
| `cpp/pendulum.cpp` | 6.84 | 1.02× | `std::vector<md::sarray>`; passes a lambda so clang inlines |
| `cpp/pendulum_mdxarray.cpp` | 6.95 | 1.04× | `md::array`, columns via `page()` |
| `fortran/pendulum_modern.f90` | 7.23 | 1.08× | modern Fortran, `pure subroutine` right-hand side |
| `fortran/pendulum.f90` | 7.38 | 1.10× | the classic style |
| `c/pendulum.c` | 7.44 | 1.11× | |
| `julia/pendulum_cstyle.jl` | 7.45 | 1.11× | Julia written as C |
| `julia/pendulum_views.jl` | 7.56 | 1.13× | fast-but-ugly Julia: preallocated buffers, `@.` everywhere |
| `julia/pendulum_svector.jl` | 9.11 | 1.36× | `pendulum_inline.jl` without `@inline` |
| `python/pendulum_numba.py` | 12.7 | 1.90× | `pendulum.py` + `@njit`, a six-line diff |
| `python/pendulum_jax.py` | 24.3 | 3.63× | jit + `fori_loop`, held to one thread like the others |
| `julia/pendulum.jl` | 37.0 | 5.5× | the beautiful Julia baseline: allocates per call |
| `julia/pendulum_tuple.jl` | 65.2 | 9.7× | tuples plus broadcast; 24× faster on Julia 1.13 than on 1.12 |
| `matlab/pendulum.m` | 108 | 16.1× | the beautiful MATLAB baseline; this is an old MATLAB (R2018b), but R2026a measured 13.8× elsewhere, so the row is about right |
| `python/pendulum_tuple.py` | 580 | 86.8× | plain Python 4-tuples, no NumPy in the hot loop |
| `python/pendulum.py` | 1192 | 178× | the beautiful Python baseline |
| JAX without `jit` | ~200 000 | ~30 000× | ~4 ms *per step*; ~150× slower than plain NumPy |

Three orderings are worth pausing on. **Julia is first, ahead of C, C++ and Fortran** — by
2% over the best C++, and that is the margin after handing clang the one inlining hint it
does not take by default. **The beautiful Julia beats the ugly Julia by 13%**
(`pendulum_inline.jl` 6.69 ms against `pendulum_views.jl` 7.56 ms), which reverses the usual
assumption that the buffer-juggling version must be faster. And **C is at the back of the
compiled group**, behind both Fortrans and both C++ variants, and level with
`pendulum_cstyle.jl` — Julia written in C's own style.

## Where the time actually goes

Measured by recording the real `sin`/`cos` arguments from a live trajectory and timing
only the `libm` calls on them:

| | libm time | total | libm share |
|---|---|---|---|
| original formulation (6 trig per RHS → 4 `libm` calls) | 9.4 ms | ~12.2 ms | ~77% |
| trig identities (4 trig per RHS → 2 `sincos` calls) | 4.4 ms | 7.2 ms | ~61% |

Reducing the library work made everything ~1.7× faster; three fifths of what is left is
still inside `glibc`. The arithmetic, the loop and all the memory traffic together are
~2.8 ms of the 7.2. (The libm figures include streaming the recorded arguments back from
memory, which the real code does not, so they are slight over-estimates.)

**No AVX is involved.** Default `-O3` targets baseline x86-64, i.e. SSE2: 128-bit packed
doubles (two at a time), no FMA, no `ymm`. The 4-element state arithmetic *is* vectorized,
just two-wide. And `-march=native`, which really does emit AVX2 + FMA, makes everything
**slower**: C 7.57 → 10.61 ms, C++ 7.17 → 10.78 ms, Fortran 7.39 → 10.19 ms. Wider vectors
buy nothing on 4-element states, and keeping wide vector state dirty across 1.2 M `libm`
calls costs more than the arithmetic saves. That held on both machines this was measured on,
including this one with AVX-512 available. So the codes are essentially optimal, and there
is no compiler-flag magic left.

**And Julia is not quietly cheating.** Julia JIT-compiles for the machine it is running on,
so unlike the `-O3` binaries it *can* use AVX and AVX-512 — its generated code contains 35
`ymm` and 4 `zmm` instructions where the compiled binaries contain none, though no FMA
anywhere, which is why the arithmetic stays bit-identical. That turns out not to be where
its lead comes from: forcing baseline codegen with `julia --cpu-target=generic` makes it
*slightly faster* (6.56 ms), not slower. The compiled languages are not being shortchanged
by their flags.

## Findings worth mentioning

**The one big speedup left is mathematical, not linguistic.** Two `sincos` calls plus
trigonometric identities give every value the equations need —
$\sin(\theta_1-\theta_2) = s_1c_2 - c_1s_2$, $\cos(\theta_1-\theta_2) = c_1c_2 + s_1s_2$,
$\sin(\theta_1-2\theta_2) = s_\Delta c_2 - c_\Delta s_2$, and
$3-\cos(2\theta_1-2\theta_2) = 2 + 2s_\Delta^2$ — halving the library calls for a uniform
**~1.7× in every language**. Exact algebra, no `-ffast-math`, no threads. Applied
everywhere in these files.

**Two fairness bugs were hiding in the original numbers.** The Fortran was factoring `h`
into the stage arguments (`yn + h*9*k1/4`), which lets the compiler hoist `h*9/4` out of
the loop, while C, C++ and Julia scaled `k` by `h` immediately. That was worth 6–7% and
was the entire reason Fortran once looked like the winner; the same trick was worth about
as much in C++. All files now use one convention. Separately, the Fortran
had `real(dp), parameter :: h = 0.2`, which takes the **single-precision** literal and
integrates with h = 0.200000002980232 — a different problem from every other file. Both
files now write `0.2_dp`. It is a classic Fortran gotcha and a good argument for
`make validate`.

**Is Julia's win an artifact of `@inline`? No — inlining is available to all of them, and
only one of the five gains from it.** None of the compiled versions inline `fpend` by
default: C makes six direct calls, C++ six indirect ones (the right-hand side arrives as a
function reference through the template parameter), and Fortran six indirect ones through
`procedure(rhs) :: f`. Given the same treatment, the results split:

| | as written | inlined | |
|---|---|---|---|
| `cpp/pendulum.cpp` | 7.10 | **6.87** | lambda instead of a function reference; **kept** |
| `cpp/pendulum_mdxarray.cpp` | 7.10 | **6.98** | same change; **kept** |
| `c/pendulum.c` | 7.40 | 7.70 | `always_inline`, 4% *slower*; not used |
| `fortran/pendulum_modern.f90` | 7.21 | 7.22 | `-flto`, no change; not used |

Both C++ variants want it and neither C nor Fortran does. It used to matter much more for
`pendulum_mdxarray.cpp` — 11% rather than 2% — because the column write went through a copy
loop whose trip count was a runtime value, and inlining `fpend` was what let the compiler
see that the loop was four iterations long. mdxarray now takes that count from whichever
operand has static extents, which removes the cliff: the un-inlined version went from 7.78
to 7.10 ms on that change alone. gcc, incidentally, *refuses* to inline `fpend` even when it
is `static` and the inline budget is raised to 3000 instructions, and the measurement says
it is right to. Three ways of getting clang to inline it — a lambda at the call site,
defining `fpend` as a `constexpr` lambda, or a function pointer as a template parameter —
all measure the same, so the files use the one that leaves `fpend` and `runge5` untouched.
Even at its best, C++ is 2% behind Julia.

The reason `@inline` is worth 1.4× in Julia and at most 3% elsewhere is that Julia was paying a
penalty the others never pay: un-inlined, `fpend` returns its `SVector` by value through
the ABI — a memory round-trip six times per step — and that also blocks the `@view` from
being optimized away. C, C++ and Fortran hand the result back through a pointer or
out-argument by convention, so their calls were already nearly free. `@inline` does not
give Julia a favour; it brings Julia up to the calling convention the others get for
nothing.

**Julia 1.13 moved two rows a long way, which is a reason to date these tables.** Going from
Julia 1.12 to 1.13, with the files untouched: `pendulum_tuple.jl` went from 1584 ms to
64.5 ms, a **24× improvement**, and `pendulum.jl` from 52 ms to 37 ms. The allocation counts
only halved (7.6 M to 3.6 M for the tuple version), so this is not simply less garbage — 208
ns per allocation on 1.12 against 18 ns on 1.13 says the old figure was a pathology in the
tuple-broadcast path rather than an inherent cost. The tuple version is still 9.5× off the
best and still the wrong way to write this, but "tuples defeat Julia" was too strong a
lesson to draw from one release.

**Fortran's array-valued function results cost 11% here.** Writing the stages as
`k2 = f(yn + h*k1/5)` — a `pure function` returning `real(dp) :: f(neq)` — reads
beautifully and measures 8.28 ms, because gfortran does not elide the copy of the
result. The same file with a `pure subroutine` and an out-argument is 7.38 ms — 11% — so
`pendulum_modern.f90` uses that form; everything else about the modernization is
unaffected. Calling `fpend` directly instead of through the procedure argument recovers
almost nothing, so it is the result copy, not the indirect call.

**A Fortran finding that did *not* survive re-measurement.** An earlier version of this
README claimed that declaring the trajectory as
`real(dp), intent(out), contiguous :: y(:,:)` and taking `nsteps` from `size(y,2)` costs
40%. On this machine it costs nothing at all — 7.32 ms against 7.38 — and the generated
code is equivalent (133 vs 136 packed instructions in `runge5`). The 40% was real but
mis-attributed: it only appears in combination with the array-valued *function* form, where
it is worth 18% here (9.77 ms against 8.28), and that is the form the file used when the
measurement was taken. With the `pure subroutine` the assumed-shape interface is free, so
`nsteps` could be dropped for a cleaner signature; it stays for now only because that has
been measured on one machine and one compiler.

**NumPy is the problem, not the solution, at four elements.** `python/pendulum_tuple.py`
drops NumPy out of the inner loop in favour of a 4-tuple subclass with element-wise
operators, and is 2.2× faster than the NumPy version while looking better. Each NumPy
binary op costs ~0.5 µs of dispatch regardless of size. Related: transposing the result
array so states are contiguous changes `pendulum.py` by 0.4% — the layout was never the
problem. In numba, where the dispatch overhead is gone, the same change is worth 5%.

**numba is clean; JAX is a minefield.** numba's fastest form is nearly its naive form —
array-expression fusion means the obvious code beats hand-written scratch buffers, and
cleaning up the file made it 22% faster. JAX needed several traps disarmed: `h` hardcoded
as a literal instead of a traced argument makes XLA lose the in-place update of the loop
carry and turns the integrator **O(N²)** (15 → 50 → 262 µs per step at
N = 2 000 / 10 000 / 50 000 — this one line cost 84×, 2.08 s per run); materializing the
transpose on return costs 13 ms of 37 ms; and `scan(..., unroll=4)` is slower *and* changes
the numerical results. XLA's thread pool is a wash on a sequential loop, so the file pins
it to one thread.

**What JAX actually buys you.** Portability is real: the same source runs on a GPU with no
changes. Speed on a GPU is not — the loop is sequential over a 4-element state, so there is
nothing for thousands of cores to do, and float64 is throttled on most of them. What would
pay off is an *ensemble*: `jax.vmap` over many initial conditions is free to write and
embarrassingly parallel, though on the CPU it buys almost nothing here (463 → 438
ns/step/trajectory from batch 1 to batch 1024). Automatic differentiation is real and works
on the file as shipped, forward and reverse, agreeing to 14 digits — but on a chaotic system
the answer is useless: $|\partial\theta_1(T)/\partial\boldsymbol y_0|$ is 4.7 at T = 10,
5.8e17 at T = 100 and 3.7e133 over the full run, and reverse mode costs 87× the primal.
Julia has the same capability via ForwardDiff and Enzyme.

**Chaos makes a great validation tool, and a great slide.** Two correct implementations
agree to 15 digits at T = 10 and share no digits at all at T = 10000. JAX against numba, as
the run lengthens:

| T | steps | relative difference |
|---|---|---|
| 10 | 50 | 3.7e-15 |
| 20 | 100 | 6.1e-14 |
| 30 | 150 | 2.7e-13 |
| 50 | 250 | 1.7e-10 |
| 100 | 500 | 9.1e-02 |

A clean exponential with a Lyapunov time near 3, seeded by XLA's `sin`/`cos` differing from
glibc's by an ulp or two.

## Validation

```
$ make validate
Validating every implementation at T = 10 (50 steps).

  pendulum.c               -27.1127168681825
  pendulum.cpp             -27.1127168681825
  ...
  OK: all 15 implementations agree.
```

`scripts/validate.py` copies each file to a temporary directory, rewrites `T = 10000` to
`T = 10`, appends a checksum print, runs it, and compares everything against the median.
Fifty steps is long enough that a wrong Runge–Kutta coefficient or a mistyped step size
shows up in the first digit, and short enough that chaos has not yet amplified last-bit
differences between compilers — at the full `T = 10000` the checksums cannot be compared
at all. The largest deviation across all fifteen is 2.5e-15, which is summation order:
Julia's and MATLAB's `sum` are pairwise where C's loop is sequential.

Two stage typos turned up in the Julia files during exactly this cross-check —
`pendulum_views.jl` and `pendulum_cstyle.jl` both had `2k1/5` where the k3 stage needs
`2k2/5`, and `9k4/75` where k6 needs `8k4/75` — so they had been integrating a different
method all along, at the same speed. Both fixed.

## Environment

The numbers above: AMD Ryzen 9 7900X (12 cores, AVX-512 and FMA available), Linux, an idle
desktop with `cpupower frequency-set -g performance` and every run pinned with
`taskset -c 0`. gcc 15.2.0, clang 21.1.8 with libc++, gfortran 15.2.0, Julia 1.13.0,
Python 3.14.4 (NumPy 2.5.2, numba 0.67.0), JAX 0.11.1, MATLAB R2018b. Medians of 6 interleaved rounds, each
round taking the best of the last 5 of the 10 timings a program prints.

Pinning matters more than the governor: on a fixed-clock desktop the governor changed little,
but an unpinned run wanders between cores and reads several percent slow. On the laptop this
was originally developed on, `powersave` moved absolute times by ~30% and reshuffled the
leading group entirely.

A handful of secondary figures were measured on that laptop and have not been re-measured
here: the JAX internals (the O(N²) `h`-as-a-literal trap, the transpose cost, the un-jitted
per-step time), the automatic-differentiation costs, the `vmap` batching numbers, and the
NumPy/numba array-layout experiments. They are quoted as ratios, which is what they are
about; the absolute milliseconds would shift by roughly the same factor as everything else.

## Contributing

New languages and variants are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for the
rules that keep the comparison fair (same arithmetic everywhere, no fast-math, no threads)
and for the output format `scripts/bench.py` expects. Results that contradict the tables
above are especially welcome; please say what hardware, compiler and CPU governor you used.

## Acknowledgements

The benchmark is mine: the problem comes from my Math 128A course at UC Berkeley, and the
original C, C++, Fortran, Julia, MATLAB and Python implementations, the choice of variants
to compare, and the conclusions this repository exists to illustrate are all my own work
from using these codes in talks over several years.

[Claude Code](https://claude.com/claude-code) was then used extensively on top of that, and
deserves credit for a good deal of what is here now:

- **Performance work.** The `sincos`-plus-identities rewrite of the right-hand side
  (~1.5× everywhere), equalizing where `h` multiplies across all implementations, and
  tracking down why `@inline` was worth 1.4× in Julia.
- **New variants.** The `std::vector<State>` and column-reference C++ versions (originally
  on a pair of hand-written headers, since ported to mdxarray), the modern Fortran rewrite,
  `julia/pendulum_inline.jl`, and `python/pendulum_tuple.py`. The numba and JAX files were
  AI-written from the start and were rewritten here.
- **Debugging.** Two wrong Runge–Kutta stage coefficients that had been sitting in the Julia
  files, a single-precision `0.2` in the Fortran, and an accidentally quadratic JAX loop that
  was costing 84×.
- **Measurement and analysis.** Finding that two thirds of the runtime is `glibc` `sin`/`cos`,
  that `-march=native` makes everything slower, and that no compiler here inlines the
  right-hand side by default.
- **Build and tooling.** The CMake setup with its graceful skipping, `scripts/bench.py`,
  `scripts/validate.py`, `animate.py`, and this README.

Every number quoted above was measured rather than estimated, but I have not personally
re-derived all of them; corrections are welcome, and any mistakes that remain are mine.

## License

MIT, see [LICENSE](LICENSE). The problem and the reference MATLAB formulation come from
Math 128A at UC Berkeley.
