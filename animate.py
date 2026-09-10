#!/usr/bin/env python3
"""Animate the double pendulum and write pendulum.gif.

The trajectory comes from the benchmark's own vanilla Python implementation --
`python/pendulum.py` is imported and its `runge5`/`fpend` used unchanged -- so
the picture shows exactly the problem being timed.  The drawing follows the
`pendplot.m` utility from Math 128A: a base bar, two rigid links, and a mass at
each joint, on equal axes.

    python3 animate.py [--time 50] [--step 0.02] [--stride 3] [--out pendulum.gif]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "python"))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from pendulum import fpend, runge5   # the vanilla implementation, unmodified


def solve(y0, h, T):
    """Integrate to time T with the benchmark's own Runge-Kutta routine."""
    return runge5(fpend, y0, h, int(round(T / h)))


def animate(y, h, stride, out, dpi=100):
    theta1, theta2 = y[0, ::stride], y[1, ::stride]
    x1, y1 = np.sin(theta1), -np.cos(theta1)
    x2, y2 = x1 + np.sin(theta2), y1 - np.cos(theta2)

    figure, axes = plt.subplots(figsize=(3.6, 3.6))
    axes.set_aspect("equal")
    axes.set_xlim(-2.1, 2.1)
    axes.set_ylim(-2.1, 2.1)
    axes.set_xticks(range(-2, 3)), axes.set_yticks(range(-2, 3))
    axes.set_xticklabels([]), axes.set_yticklabels([])
    axes.tick_params(length=0)
    axes.grid(True, linewidth=0.4, alpha=0.3)
    for spine in axes.spines.values():
        spine.set_alpha(0.3)

    axes.plot([-0.2, 0.2], [0, 0], linewidth=5, color="tab:blue", solid_capstyle="round")
    trail, = axes.plot([], [], linewidth=1, color="tab:red", alpha=0.3)
    links, = axes.plot([], [], linewidth=2.5, color="black", solid_capstyle="round")
    masses, = axes.plot([], [], linestyle="none", marker="o", markersize=9, color="tab:red")
    clock = axes.text(0.03, 0.97, "", transform=axes.transAxes, va="top",
                      fontsize=8, alpha=0.6)
    figure.tight_layout(pad=0.2)

    tail = 60   # frames of tip trail

    def frame(n):
        links.set_data([0, x1[n], x2[n]], [0, y1[n], y2[n]])
        masses.set_data([x1[n], x2[n]], [y1[n], y2[n]])
        start = max(0, n - tail)
        trail.set_data(x2[start:n + 1], y2[start:n + 1])
        clock.set_text(f"t = {n * stride * h:5.1f}")
        return links, masses, trail, clock

    frames = len(theta1)
    animation = FuncAnimation(figure, frame, frames=frames, blit=True)
    animation.save(out, writer=PillowWriter(fps=30), dpi=dpi)
    plt.close(figure)
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--time", type=float, default=50.0, help="final time (default 50)")
    parser.add_argument("--step", type=float, default=0.02, help="step size (default 0.02)")
    parser.add_argument("--stride", type=int, default=3, help="keep every Nth step (default 3)")
    parser.add_argument("--out", default="pendulum.gif", help="output file")
    parser.add_argument("--y0", type=float, nargs=4, metavar=("TH1", "TH2", "OM1", "OM2"),
                        default=[2.0, 2.0, 0.0, -1.0],
                        help="initial state (default: the benchmark's 2 2 0 -1)")
    args = parser.parse_args()

    print(f"solving to t = {args.time} with h = {args.step} ...")
    y = solve(args.y0, args.step, args.time)
    frames = animate(y, args.step, args.stride, args.out)
    size = Path(args.out).stat().st_size / 1e6
    print(f"wrote {args.out}: {frames} frames, {size:.1f} MB")


if __name__ == "__main__":
    main()
