#!/usr/bin/env python3

"""
Plot VQE Benchmark Data
=======================

Plot the VQE benchmark data.

Author: Joey Carter <joey.carter@cern.ch>
"""


import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import catalyst


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
    if args.verbose > 1:
        print(f"Reading benchmarks data from file '{args.path}'")

    with open(args.path, "r") as fin:
        data = json.load(fin)

    machine_info = data["machine_info"]
    benchmarks = data["benchmarks"]

    df = parse_benchmarks(benchmarks, args)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=[6.4, 6.4], height_ratios=[3, 1], sharex=True
    )

    # Iterate over all molecules found in the dataframe
    molecules = df.columns.levels[0].unique()

    # Use the 'tab20' colormap, which is a list of paired light/dark colours,
    # and select the colour by the index in the colormap. See:
    # https://matplotlib.org/stable/users/explain/colors/colormaps.html
    color_index = 0

    for mol in molecules:
        # Plot benchmarking data in main (upper) panel
        ax1.errorbar(
            df.index,
            df[mol]["pennylane"]["mean"],
            yerr=df[mol]["pennylane"]["stderr"],
            label=f"{mol} (Native PennyLane)",
            marker="o",
            c=plt.cm.tab20(color_index + 1),
            ls="--",
        )
        ax1.errorbar(
            df.index,
            df[mol]["compile_and_execution"]["mean"],
            yerr=df[mol]["compile_and_execution"]["stderr"],
            label=f"{mol} (Compile+Execution)",
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
    ax1.text(
        0.02,
        0.98,
        r"$\mathbfit{PennyLane}$ Internal",
        transform=ax1.transAxes,
        fontsize=16,
        verticalalignment="top",
    )
    ax1.text(
        0.98,
        0.98,
        f"VQE Benchmarks\n{machine_info['cpu']['brand_raw']}\nCatalyst v{catalyst.__version__}",
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
    )

    # Plot speed-up factor in lower panel
    color_index = 0
    for mol in molecules:
        speedup_factor = (
            df[mol]["pennylane"]["mean"] / df[mol]["compile_and_execution"]["mean"]
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

    fname_prefix = f"vqe_benchmarks.catalyst_{catalyst.__version__}"
    for ext in ["png", "svg", "pdf"]:
        fname = os.path.join(args.outdir, f"{fname_prefix}.{ext}")
        if args.verbose:
            print(f"Saving figure to file {fname}")
        plt.savefig(fname)

    # Free up resources
    plt.close(fig)


def parse_benchmarks(benchmarks: list, args):
    """Parse benchmarking data and return as a pandas DataFrame

    This function parses the benchmarking data from the list of individual
    benchmarks that were parsed from the JSON pytest benchmarking output.

    Call as, e.g.

        >>> import json
        >>> with open("path/to/benchmarks.json", "r") as fin:
        ...     data = json.load(fin)
        >>> df = parse_benchmarks(data["benchmarks"], args)

        The returned data frame is in the format, e.g.

                        H2-STO-3G                                    HeH+-STO-3G
            compile_and_execution       pennylane          compile_and_execution       pennylane
                            mean  stderr    mean  stderr                   mean  stderr    mean  stderr
        it_max
        10                  0.3008  0.0036  0.1878  0.0456                 0.3438  0.0022  0.1394  0.0587
        50                  0.3023  0.0025  0.4898  0.1054                 0.3389  0.0045  0.6232  0.0756
        100                 0.2966  0.0026  1.2519  0.2025                 0.3426  0.0055  1.5131  0.1293
        200                 0.3075  0.0030  2.3756  0.1830                 0.3561  0.0047  2.8572  0.2344
        300                 0.3113  0.0027  3.9498  0.2859                 0.3704  0.0089  3.5509  0.2496
        400                 0.3288  0.0187  2.9637  0.2230                 0.3755  0.0054  5.1274  0.4936
        500                 0.3289  0.0045  3.8653  0.3746                 0.3799  0.0061  4.2845  0.2262

    Args:
        benchmarks (list): List of individual benchmarks from JSON
        args (argparse.Namespace): Command-line arguments from argparse
    """
    # Build dataframe dynamically as a dictionary and construct pandas dataframe at end
    df_dict = {}

    for benchmark in benchmarks:
        name = benchmark["name"]

        it_max = benchmark["params"]["it_max"]
        mol = benchmark["params"]["mol"]
        basis_set = benchmark["params"]["basis_set"]

        # "Mode" is the PennyLane workflow used for a particular set of benchmarks, e.g.
        #   - 'pennylane' for the runtime of the native PennyLane workflow
        #   - 'compile_and_execution' for the compilation+execution runtime of the Catalyst workflow
        modes_to_plot = ["pennylane", "compile_and_execution"]
        mode = parse_mode_from_benchmark_name(name, modes_to_plot)

        if mode is None:
            if args.verbose:
                print(f"skipping benchmark '{name}'; not in list of modes to plot")

            continue

        if it_max not in df_dict:
            df_dict[it_max] = {}

        key = f"{mol}-{basis_set}"

        df_dict[it_max][(key, mode, "mean")] = benchmark["stats"]["mean"]
        df_dict[it_max][(key, mode, "stderr")] = benchmark["stats"]["stddev"] / np.sqrt(
            benchmark["stats"]["rounds"]
        )

    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.from_dict.html
    df = pd.DataFrame.from_dict(df_dict, orient="index")

    # Clean up dataframe
    df.index.name = "it_max"
    df.sort_index(axis="index", inplace=True)
    df.sort_index(axis="columns", inplace=True)

    return df


def parse_mode_from_benchmark_name(name: str, allowed_modes: list):
    """Parse the workflow mode from the benchmark name.

    This function will only search for the modes listed in `allowed_modes`, in
    the order in which they are give, and return the first match. If no match is
    found, return None.

    For example:

        >>> parse_mode_from_benchmark_name(
        ...     "test_VQE_pennylane_fast[10-HeH+-STO-3G]",
        ...     ["pennylane", "compile_and_execution"])
        'pennylane'

        >>> parse_mode_from_benchmark_name(
        ...     "test_VQE_execution_only_fast[10-HeH+-STO-3G]",
        ...     ["pennylane", "compile_and_execution"])
        None

    Args:
        name (str): Benchmark name, as it appears in the benchmark["name"] field
        allowed_modes (list): Modes to search for

    Returns:
        str: Matched workflow mode, or None if no match is found
    """
    result = None
    for mode in allowed_modes:
        if mode in name:
            result = mode

    return result


if __name__ == "__main__":
    sys.exit(main())
