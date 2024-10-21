#!/usr/bin/env python3

"""
Plot XAS Benchmark Data
=======================

Plot the XAS benchmark data.

Author: Joey Carter <joey.carter@cern.ch>
"""


import argparse
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import catalyst

# Add parent directory to sys.path to import plotutils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
    parser.add_argument("path", help="path to benchmarks JSON file")

    args = parser.parse_args()

    return args


def _docstring(docstring):
    """Return summary of docstring"""
    return " ".join(docstring.split("\n")[4:5]) if docstring else ""


def plot(args):
    """Plot the benchmarks

    Args:
        args (argparse.Namespace): Command-line arguments from argparse
    """
    df, machine_info = plotutils.parse_benchmarks(
        args.path,
        index="num_steps",
        params=["N1", "ncas"],
        modes=["pennylane", "compile_and_execution"],
        args=args,
    )

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=[6.4, 6.4], height_ratios=[3, 1], sharex=True
    )

    # Iterate over all molecules found in the dataframe
    n1_ncas_combos = df.columns.levels[0].unique()

    # Use the 'tab20' colormap, which is a list of paired light/dark colours,
    # and select the colour by the index in the colormap. See:
    # https://matplotlib.org/stable/users/explain/colors/colormaps.html
    color_index = 0

    for n1_ncas in n1_ncas_combos:
        # Plot benchmarking data in main (upper) panel
        ax1.errorbar(
            df.index,
            df[n1_ncas]["pennylane"]["mean"],
            yerr=df[n1_ncas]["pennylane"]["stderr"],
            label=f"{n1_ncas} (Native PennyLane)",
            marker="o",
            c=plt.cm.tab20(color_index + 1),
            ls="--",
        )
        ax1.errorbar(
            df.index,
            df[n1_ncas]["compile_and_execution"]["mean"],
            yerr=df[n1_ncas]["compile_and_execution"]["stderr"],
            label=f"{n1_ncas} (Compile+Execution)",
            marker="s",
            c=plt.cm.tab20(color_index),
        )

        color_index += 2

    # Add extra space at top of axes
    ymin, ymax = ax1.get_ylim()
    ax1.set_ylim(ymin, ymax * 1.2)

    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_ylabel("Mean Runtime [s]")

    ax1.legend(
        loc="upper left", bbox_to_anchor=(0.01, 0.88), frameon=False, fontsize=10
    )

    # Text annotations
    plotutils.pennylane_label(ax1, qualifier="Internal")
    plotutils.label(
        ax1,
        0.98,
        0.98,
        f"XAS Benchmarks\n{machine_info['cpu']['brand_raw']}\nCatalyst v{catalyst.__version__}",
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
    )

    # Plot speed-up factor in lower panel
    color_index = 0
    for n1_ncas in n1_ncas_combos:
        speedup_factor = (
            df[n1_ncas]["pennylane"]["mean"]
            / df[n1_ncas]["compile_and_execution"]["mean"]
        )
        ax2.plot(
            df.index, speedup_factor, c=plt.cm.tab10(color_index), marker="o", zorder=10
        )
        color_index += 1

    ax2.grid(axis="y", zorder=0)

    ax2.set_ylabel("Compilation Speedup")
    ax2.set_xlabel("Number of Iterations")

    plt.tight_layout()

    # Save figure
    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    fname_prefix = f"xas_benchmarks.catalyst_{catalyst.__version__}"
    for ext in ["png", "svg", "pdf"]:
        fname = os.path.join(args.outdir, f"{fname_prefix}.{ext}")
        if args.verbose:
            print(f"Saving figure to file {fname}")
        plt.savefig(fname)

    # Free up resources
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
