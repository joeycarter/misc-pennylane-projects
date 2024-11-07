#!/usr/bin/env python3

"""
Plot VQE Benchmark Data
=======================

Plot the VQE benchmark data.

Author: Joey Carter <joseph.carter@xanadu.ch>
"""


import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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
        index="it_max",
        params=["mol", "basis_set"],
        modes=["pennylane", "compile_and_execution"],
        args=args,
    )

    fig, (ax1, ax2) = plotutils.create_benchmark_fig_and_axes(
        title="Variational Quantum Eigensolver (VQE)"
    )

    # Iterate over all molecules found in the dataframe
    molecules = df.columns.levels[0].unique()

    for mol in molecules:
        # Plot benchmarking data in main (upper) panel
        ax1.errorbar(
            df.index,
            df[mol]["pennylane"]["mean"],
            yerr=df[mol]["pennylane"]["stderr"],
            label=f"{mol} (PennyLane)",
            marker="o",
            c=cmaps.pl8[color_index + 1],
            ls="--",
        )
        ax1.errorbar(
            df.index,
            df[mol]["compile_and_execution"]["mean"],
            yerr=df[mol]["compile_and_execution"]["stderr"],
            label=f"{mol} (Catalyst)",
            marker="s",
            c=cmaps.pl8[color_index],
        )

        color_index += 2

    # Add extra space at top of axes
    ymin, ymax = ax1.get_ylim()
    ax1.set_ylim(ymin, ymax * 1.10)

    # ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax1.set_ylabel("Total Runtime [s]")

    plotutils.draw_benchmark_plot_legend(ax1)

    # Plot speed-up factor in lower panel
    color_index = 0
    for mol in molecules:
        speedup_factor = (
            df[mol]["pennylane"]["mean"] / df[mol]["compile_and_execution"]["mean"]
        )
        ax2.plot(
            df.index, speedup_factor, c=cmaps.pl4[color_index], marker="o", zorder=10
        )
        color_index += 1

    ax2.grid(axis="y", zorder=0)

    ax2.set_ylabel("Speedup Factor")
    ax2.set_xlabel("Number of Iterations")

    fname_prefix = f"vqe_benchmarks.catalyst_0.9.0"
    plotutils.savefig_and_close(fig, fname_prefix, args)


if __name__ == "__main__":
    sys.exit(main())
