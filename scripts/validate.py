#!/usr/bin/env python3
"""Check that every implementation really solves the same problem.

The implementations are deliberately plain: the final time is a literal in each
file and none of them takes command-line arguments.  So this script copies each
one into a temporary directory, rewrites `T = 10000` to `T = 10`, appends a
checksum print, runs it, and compares the results.

Ten time units is 50 steps -- long enough that a wrong Runge-Kutta coefficient
or a mistyped step size shows up in the first digit, and short enough that the
chaos has not yet amplified last-bit differences between compilers.  Over the
full T = 10000 the checksums cannot be compared at all: two correct
implementations share no digits by then.
"""

import argparse
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHECKSUM_RE = re.compile(r"CHECKSUM\s+(\S+)")
TOLERANCE = 1e-10   # relative; comfortably above summation-order differences


def cache_value(build, key, default=None):
    """Read a variable out of CMakeCache.txt."""
    cache = build / "CMakeCache.txt"
    if not cache.exists():
        return default
    for line in cache.read_text().splitlines():
        name, _, rest = line.partition(":")
        if name == key and "=" in rest:
            return rest.split("=", 1)[1]
    return default


def patch(text, rules, what):
    for old, new in rules:
        if text.count(old) != 1:
            raise RuntimeError(f"{what}: expected exactly one occurrence of {old!r}, "
                               f"found {text.count(old)}")
        text = text.replace(old, new)
    return text


CPP_CHECK = ('std::cout << "Time "',
             'std::cout << "CHECKSUM " << std::setprecision(17) << benchmark_sink\n'
             '                  << "\\n" << std::setprecision(6) << "Time "')

JULIA_TAIL = """
let yy = runge5(fpend, y0, h, N)
    println("CHECKSUM ", sum(yy))
end
"""


def cases(build):
    """Each case: name, source file, patch rules, extra files, how to build/run."""
    cc = cache_value(build, "CMAKE_C_COMPILER", shutil.which("cc"))
    cxx = cache_value(build, "CMAKE_CXX_COMPILER", shutil.which("c++"))
    fc = cache_value(build, "CMAKE_Fortran_COMPILER", shutil.which("gfortran"))
    libcxx = cache_value(build, "HAVE_MDSPAN_LIBCXX") == "1"
    has_mdspan = libcxx or cache_value(build, "HAVE_MDSPAN") == "1"

    out = []
    if cc:
        out.append(dict(
            name="pendulum.c", src=ROOT / "c/pendulum.c",
            rules=[("double T = 10000.0;", "double T = 10.0;"),
                   ("        free(y);",
                    '        { double s = 0; for (int i = 0; i < 4*(N+1); i++) s += y[i];\n'
                    '          printf("CHECKSUM %.17g\\n", s); }\n        free(y);')],
            build=lambda d, f, cc=cc: [cc, "-O3", "-o", str(d / "a.out"), str(f), "-lm"],
            run=lambda d: [str(d / "a.out")]))
    if cxx:
        cpp_files = [("pendulum.cpp", "pendulum.cpp", True)]
        if has_mdspan:
            cpp_files += [("pendulum_mdxarray.cpp",) * 2 + (True,)]
        for name, filename, _ in cpp_files:
            flags = ["-O3", "-fno-math-errno", "-std=c++23"]
            if libcxx:
                flags.append("-stdlib=libc++")
            out.append(dict(
                name=name, src=ROOT / "cpp" / filename,
                rules=[("const double T = 10000.0;", "const double T = 10.0;"), CPP_CHECK],
                copy=[ROOT / "cpp/md.h"],
                build=lambda d, f, flags=flags, cxx=cxx: [cxx, *flags, "-I", str(d),
                                                          "-o", str(d / "a.out"), str(f)],
                run=lambda d: [str(d / "a.out")]))
    if fc:
        out.append(dict(
            name="pendulum.f90", src=ROOT / "fortran/pendulum.f90",
            rules=[("real(dp), parameter :: T = 10000", "real(dp), parameter :: T = 10"),
                   ("     call cpu_time(finish)",
                    "     call cpu_time(finish)\n"
                    "     print '(\"CHECKSUM \",es25.17)', sum(y)")],
            build=lambda d, f, fc=fc: [fc, "-O3", "-J", str(d), "-o", str(d / "a.out"), str(f)],
            run=lambda d: [str(d / "a.out")]))
        out.append(dict(
            name="pendulum_modern.f90", src=ROOT / "fortran/pendulum_modern.f90",
            rules=[("real(dp), parameter :: T = 10000", "real(dp), parameter :: T = 10"),
                   ("     sink = sum(y)",
                    "     print '(\"CHECKSUM \",es25.17)', sum(y)\n     sink = sum(y)")],
            build=lambda d, f, fc=fc: [fc, "-O3", "-J", str(d), "-o", str(d / "a.out"), str(f)],
            run=lambda d: [str(d / "a.out")]))

    julia = shutil.which("julia")
    if julia:
        for stem in ("pendulum", "pendulum_inline", "pendulum_svector",
                     "pendulum_views", "pendulum_cstyle", "pendulum_tuple"):
            out.append(dict(
                name=f"{stem}.jl", src=ROOT / "julia" / f"{stem}.jl",
                rules=[("\nT = 10000\n", "\nT = 10\n")], append=JULIA_TAIL,
                run=lambda d, julia=julia: [julia, str(d / "src")]))

    def python_with(module):
        for interpreter in (sys.executable, shutil.which("python3"),
                            str(Path.home() / ".venv/bin/python"),
                            str(ROOT / ".venv/bin/python")):
            if interpreter and os.path.exists(interpreter):
                if subprocess.run([interpreter, "-c", f"import {module}"],
                                  capture_output=True).returncode == 0:
                    return interpreter
        return None

    plain = shutil.which("python3") or sys.executable
    for stem, interpreter in (("pendulum", plain), ("pendulum_tuple", plain),
                              ("pendulum_numba", python_with("numba")),
                              ("pendulum_jax", python_with("jax"))):
        if not interpreter:
            continue
        out.append(dict(
            name=f"{stem}.py", src=ROOT / "python" / f"{stem}.py",
            rules=[("    T = 10000;", "    T = 10;")],
            append='\nprint("CHECKSUM %.17g" % float(__import__("numpy").asarray(y).sum()))\n',
            run=lambda d, interpreter=interpreter: [interpreter, str(d / "src")]))
    return out


def run_case(case, keep, timeout):
    directory = Path(tempfile.mkdtemp(prefix="pendulum-validate-"))
    try:
        source = case["src"].read_text()
        source = patch(source, case.get("rules", []), case["name"])
        source += case.get("append", "")
        target = directory / ("src" + case["src"].suffix)
        target.write_text(source)
        (directory / "src").write_text(source)   # extension-free name for interpreters
        for extra in case.get("copy", []):
            shutil.copy(extra, directory)

        if "build" in case:
            command = case["build"](directory, target)
            compiled = subprocess.run([str(c) for c in command], capture_output=True, text=True)
            if compiled.returncode != 0:
                return None, "build failed: " + compiled.stderr.strip().splitlines()[-1][:90]

        done = subprocess.run([str(c) for c in case["run"](directory)], cwd=directory,
                              capture_output=True, text=True, timeout=timeout)
        found = CHECKSUM_RE.findall(done.stdout)
        if not found:
            tail = (done.stderr or done.stdout).strip().splitlines()
            return None, "no checksum printed" + (f": {tail[-1][:90]}" if tail else "")
        return float(found[-1].replace("D", "E")), None
    except subprocess.TimeoutExpired:
        return None, f"timed out after {timeout:.0f}s"
    except Exception as exc:                       # noqa: BLE001 -- report, do not crash
        return None, f"{type(exc).__name__}: {exc}"
    finally:
        if keep:
            print(f"    (kept {directory})")
        else:
            shutil.rmtree(directory, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", default="build", help="cmake build directory")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--keep", action="store_true", help="keep the temporary directories")
    args = parser.parse_args()

    build = (ROOT / args.build) if not os.path.isabs(args.build) else Path(args.build)
    print("Validating every implementation at T = 10 (50 steps).\n")

    values, failures = {}, {}
    for case in cases(build):
        print(f"  {case['name']:24s}", end="", flush=True)
        value, error = run_case(case, args.keep, args.timeout)
        if error:
            failures[case["name"]] = error
            print(f" -- {error}")
        else:
            values[case["name"]] = value
            print(f" {value:.15g}")

    if not values:
        print("\nNothing could be checked.")
        return 1

    reference = statistics.median(values.values())
    worst = max(abs(v - reference) / max(abs(reference), 1e-300) for v in values.values())
    print(f"\n  reference (median): {reference:.15g}")
    print(f"  largest relative deviation: {worst:.2e}   (tolerance {TOLERANCE:.0e})")

    disagreeing = {n: v for n, v in values.items()
                   if abs(v - reference) / max(abs(reference), 1e-300) > TOLERANCE}
    if disagreeing:
        print("\n  DISAGREE:")
        for name, value in sorted(disagreeing.items()):
            print(f"    {name:24s} {value:.15g}")
    else:
        print(f"\n  OK: all {len(values)} implementations agree.")

    if failures:
        print("\n  not checked:")
        for name, why in sorted(failures.items()):
            print(f"    {name:24s} {why}")
    return 1 if disagreeing else 0


if __name__ == "__main__":
    sys.exit(main())
