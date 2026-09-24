#!/usr/bin/env python3
"""Run every available implementation and report the best of its last timings.

Each implementation prints one line per repetition and nothing else, in whatever
format is natural for its language.  Rather than teach every language a common
output format -- which would clutter code whose readability is the whole point --
this script scrapes the numbers back out and averages the last few, by which
point caches are warm and any JIT has compiled.

Anything that is not installed is reported as skipped, never as a failure.
"""

import argparse
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# A timing line is either "<something> 0.0123 seconds <something>" or a bare
# number on its own line.  Fortran prints ".01065" with no leading zero, and
# Julia prints "  0.010424 seconds (5 allocations: 1.527 MiB)".
NUMBER = r"(\d*\.\d+(?:[eE][-+]?\d+)?|\d+\.?\d*(?:[eE][-+]?\d+)?)"
SECONDS_RE = re.compile(NUMBER + r"\s*seconds")
BARE_RE = re.compile(r"^\s*" + NUMBER + r"\s*$")


def parse_times(output):
    """Every timing printed by an implementation, in order, as seconds."""
    times = []
    for line in output.splitlines():
        match = SECONDS_RE.search(line) or BARE_RE.match(line)
        if match:
            try:
                times.append(float(match.group(1)))
            except ValueError:
                pass
    return times


def python_with(module, extra_interpreters=()):
    """The first interpreter that can import `module`, or None."""
    candidates = [sys.executable, "python3"]
    candidates.extend(str(p) for p in extra_interpreters)
    for interpreter in candidates:
        if not interpreter:
            continue
        path = shutil.which(interpreter) if os.sep not in interpreter else interpreter
        if not path or not os.path.exists(path):
            continue
        probe = subprocess.run([path, "-c", f"import {module}"],
                               capture_output=True, text=True)
        if probe.returncode == 0:
            return path
    return None


def implementations(build):
    """(name, language, command, cwd, note) for everything we know how to run."""
    venvs = [Path.home() / ".venv/bin/python", ROOT / ".venv/bin/python"]
    numba = python_with("numba", venvs)
    jax = python_with("jax", venvs)
    plain = shutil.which("python3") or sys.executable
    julia = shutil.which("julia")
    matlab = shutil.which("matlab")

    items = [
        ("pendulum_c", "C", [build / "pendulum_c"], None, ""),
        ("pendulum_cpp", "C++", [build / "pendulum_cpp"], None, "std::vector<State>"),
        ("pendulum_cpp_mdxarray", "C++", [build / "pendulum_cpp_mdxarray"], None, "md::array"),
        ("pendulum_f", "Fortran", [build / "pendulum_f"], None, ""),
        ("pendulum_f_modern", "Fortran", [build / "pendulum_f_modern"], None, ""),
    ]
    for name in ("pendulum_inline", "pendulum_svector", "pendulum_views",
                 "pendulum_cstyle", "pendulum", "pendulum_tuple"):
        items.append((f"{name}.jl", "Julia",
                      [julia, ROOT / "julia" / f"{name}.jl"] if julia else None,
                      None, "" if julia else "julia not found"))
    items.append(("pendulum_numba.py", "Python",
                  [numba, ROOT / "python/pendulum_numba.py"] if numba else None,
                  None, "" if numba else "numba not installed"))
    items.append(("pendulum_jax.py", "Python",
                  [jax, ROOT / "python/pendulum_jax.py"] if jax else None,
                  None, "" if jax else "jax not installed"))
    items.append(("pendulum_tuple.py", "Python", [plain, ROOT / "python/pendulum_tuple.py"], None, ""))
    items.append(("pendulum.py", "Python", [plain, ROOT / "python/pendulum.py"], None, ""))
    items.append(("pendulum.m", "MATLAB",
                  [matlab, "-batch", "pendulum"] if matlab else None,
                  ROOT / "matlab", "" if matlab else "matlab not found"))
    return items


# Implementations that take seconds rather than milliseconds per run.
SLOW = {"pendulum.py", "pendulum_tuple.py", "pendulum_tuple.jl", "pendulum.m", "pendulum.jl"}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", default="build", help="cmake build directory")
    parser.add_argument("--last", type=int, default=5,
                        help="average the last N timings of each run (default 5)")
    parser.add_argument("--quick", action="store_true",
                        help="skip the implementations that take seconds per run")
    parser.add_argument("--timeout", type=float, default=600.0,
                        help="per-implementation timeout in seconds (default 600)")
    parser.add_argument("--only", default=None,
                        help="substring filter on the implementation name")
    args = parser.parse_args()

    build = (ROOT / args.build) if not os.path.isabs(args.build) else Path(args.build)
    results, skipped = [], []

    for name, language, command, cwd, note in implementations(build):
        if args.only and args.only not in name:
            continue
        if args.quick and name in SLOW:
            skipped.append((name, "slow, skipped by --quick"))
            continue
        if command is None:
            skipped.append((name, note or "not available"))
            continue
        if isinstance(command[0], Path) and not command[0].exists():
            skipped.append((name, "not built"))
            continue

        print(f"  running {name} ...", end="", flush=True)
        started = time.time()
        try:
            done = subprocess.run([str(c) for c in command], cwd=cwd, timeout=args.timeout,
                                  capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            print(" timed out")
            skipped.append((name, f"timed out after {args.timeout:.0f}s"))
            continue
        except OSError as exc:
            print(" failed")
            skipped.append((name, str(exc)))
            continue

        times = parse_times(done.stdout)
        if not times:
            print(" no timings found")
            skipped.append((name, "produced no timings"
                            + (f"; exit {done.returncode}" if done.returncode else "")))
            continue
        tail = times[-args.last:]
        results.append((min(tail), statistics.mean(tail), len(tail), name, language, note))
        print(f" {min(tail) * 1e3:.2f} ms  ({time.time() - started:.0f}s wall)")

    if not results:
        print("\nNothing ran.  Build first with `make`, and see the skip list above.")
        return 1

    results.sort()
    fastest = results[0][0]
    width = max(len(r[3]) for r in results)
    print(f"\n  {'implementation'.ljust(width)}  {'best':>10}  {'mean':>10}   vs fastest")
    print("  " + "-" * (width + 38))
    for best, mean, count, name, language, note in results:
        unit = lambda t: f"{t * 1e3:.2f} ms" if t < 1 else f"{t:.3f} s"
        print(f"  {name.ljust(width)}  {unit(best):>10}  {unit(mean):>10}   "
              f"{best / fastest:8.2f}x" + (f"   {note}" if note else ""))
    print(f"\n  Best and mean of the last {args.last} timings of each run, ranked by best.")
    print("  Best is the fairer number here: a mean includes whatever the runtime happened")
    print("  to do between iterations, and Julia's @time in particular charges garbage")
    print("  collection of earlier iterations to later ones.")
    print("  For stable numbers: fix the clock (`cpupower frequency-set -g performance`),")
    print("  pin to one core (`taskset -c 0 make bench`), and close everything else.")

    if skipped:
        print("\n  skipped:")
        for name, why in skipped:
            print(f"    {name} -- {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
