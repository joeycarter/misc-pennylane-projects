#!/usr/bin/env python3

"""
Plot "Simple Circuit" Benchmark Data
====================================

Plot the "simple circuit" benchmark data, which shows the "compilation time" of
performing cancel inverses and merge rotations in Catalyst vs core PennyLane.

Authors:

  - Joey Carter <joseph.carter@xanadu.ai>
  - Paul Wang <paul.wang@resident.xanadu.ai>
"""


import argparse
import os
import sys

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.image as mpimg
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

import catalyst

# Add parent directory to sys.path to import plotutils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import colourmaps as cmaps
import plotutils


def main():
    try:
        args = parse_args()
        plot(args)

    except KeyboardInterrupt:
        return 1


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description=_docstring(__doc__))
    parser.add_argument("--version", action="version", version="%(prog)s 0.1")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="print verbose messages; multiple -v result in more verbose messages",
    )
    parser.add_argument(
        "-o", "--outdir", default="plots", help="path to output directory"
    )
    # parser.add_argument("path", help="path to benchmarks .npz file")

    args = parser.parse_args()

    return args


def _docstring(docstring):
    """Return summary of docstring"""
    return " ".join(docstring.split("\n")[4:6]) if docstring else ""


def plot(args):
    """Plot the benchmarks

    Args:
        args (argparse.Namespace): Command-line arguments from argparse
    """
    data = np.load("timeit_peephole_benchmark_data_geom25_err.npz")

    loopsizes = data["loopsizes"] * 2  # each loop has 4 gates
    walltimes = data["walltimes"]
    cputimes = data["cputimes"]
    programsizes = data["programsizes"]
    core_PL_times = data["core_PL_times"]
    walltime_errs = data["walltime_errs"]
    cputime_errs = data["cputime_errs"]
    programsize_errs = data["programsize_errs"]
    core_PL_time_errs = data["core_PL_time_errs"]

    fig, (ax1, ax2) = plotutils.create_benchmark_fig_and_axes(
        title="Simple Quantum Circuit Optimizations"
    )

    blue = cmaps.pl4[0]
    magenta = cmaps.pl4[1]

    ax1.errorbar(
        loopsizes,
        core_PL_times,
        yerr=core_PL_time_errs,
        marker="s",
        label="PennyLane",
        c=blue,
        ls="--",
        zorder=2,
    )
    ax1.errorbar(
        loopsizes,
        cputimes,
        yerr=cputime_errs,
        marker="o",
        label="Catalyst",
        c=magenta,
        zorder=2,
    )

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_ylabel("Compilation Time [ms]", fontsize=14)
    ax1.set_ylim(-0.00001, 2e6)

    # # Add extra space at top of axes
    # ymin, ymax = ax1.get_ylim()
    # ax1.set_ylim(ymin, ymax * 1.10)

    # Image annotation
    img = mpimg.imread("auto_peephole_comp_horizontal.png")
    imagebox = OffsetImage(img, zoom=0.32)
    ab = AnnotationBbox(
        imagebox, (0.39, 0.69), zorder=1, frameon=False, xycoords=ax1.transAxes
    )
    ax1.add_artist(ab)

    plotutils.draw_benchmark_plot_legend(ax1, where="right")

    # Plot speed-up factor in lower panel
    ax2.plot(loopsizes, core_PL_times / cputimes, c=magenta, marker="o")
    ax2.set_xlabel("Circuit Gate Depth, N")
    ax2.set_ylabel("Speedup Factor")
    ax2.set_xscale("log")
    ax2.set_yscale("log")

    ax2.grid(axis="y", zorder=0)

    fname_prefix = f"simple_circuit_benchmarks.catalyst_0.9.0"
    plotutils.savefig_and_close(fig, fname_prefix, args)


if __name__ == "__main__":
    sys.exit(main())
