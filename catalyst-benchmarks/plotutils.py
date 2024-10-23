import json

import numpy as np
import pandas as pd
import matplotlib


def parse_benchmarks(
    fpath: list, index: str, params: list, modes: list, args, machine_info=True
):
    """Parse benchmarking data and return as a pandas DataFrame

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
        fpath (str, path-like): Path to the benchmarks JSON file.
        index (str): Name of the parameter to use as the index in the output
            DataFrame, e.g. 'num_iter'.
        params (list): List parameter names that were used to parameterize the
            benchmarks. The top-level DataFrame column names are formed from
            each unique set of parameters used. For example, if there are two
            parameters ['foo', 'bar'], with possible values ['foo1', 'foo1']
            for parameter 'foo', and 'bar1' for parameter 'bar', then there are
            two unique sets of parameters used: 'foo1-bar1' and 'foo2-bar1'.
        modes (list): List of benchmarking modes, where a "mode" is the
            PennyLane workflow used for a particular set of benchmarks, e.g.
              - 'pennylane' for the runtime of the native PennyLane workflow
              - 'compile_and_execution' for the compilation+execution runtime of
                the Catalyst workflow
        args (argparse.Namespace): Command-line arguments from argparse.
        machine_info (bool): If True, also return the dictionary of information
            about the machine on which the benchmarks were run.
    """
    json_data = _get_benchmark_data_from_json(fpath, args)

    benchmarks = json_data["benchmarks"]
    machine_info_ = json_data["machine_info"]

    # Build dataframe dynamically as a dictionary and construct pandas dataframe at end
    df_dict = {}

    for benchmark in benchmarks:
        name_ = benchmark["name"]
        index_ = benchmark["params"][index]

        params_ = {}

        for param in params:
            params_[param] = benchmark["params"][param]

        mode = _parse_mode_from_benchmark_name(name_, modes)

        if mode is None:
            if args.verbose:
                print(f"skipping benchmark '{name_}'; not in list of modes to parse")

            continue

        if index_ not in df_dict:
            df_dict[index_] = {}

        key = "-".join(params_.values())

        df_dict[index_][(key, mode, "mean")] = benchmark["stats"]["mean"]
        df_dict[index_][(key, mode, "stderr")] = benchmark["stats"]["stddev"] / np.sqrt(
            benchmark["stats"]["rounds"]
        )

    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.from_dict.html
    df = pd.DataFrame.from_dict(df_dict, orient="index")

    # Clean up dataframe
    df.index.name = index
    df.sort_index(axis="index", inplace=True)
    df.sort_index(axis="columns", inplace=True)

    if args.verbose >= 2:
        print("Benchmark dataframe:")
        print(df.head())

    if machine_info:
        return df, machine_info_
    else:
        return df


def _get_benchmark_data_from_json(fpath, args):
    """Get benchmark data from JSON.

    Args:
        fpath (str, path-like): Path to the benchmarks JSON file
        args (argparse.Namespace): Command-line arguments from argparse

    Returns:
        list: benchmarks
        dict: machine_info
    """
    if args.verbose > 1:
        print(f"Reading benchmarks data from file '{fpath}'")

    with open(args.path, "r") as fin:
        data = json.load(fin)

    return data


def _parse_mode_from_benchmark_name(name: str, allowed_modes: list):
    """Parse the workflow mode from the benchmark name.

    This function will only search for the modes listed in `allowed_modes`, in
    the order in which they are give, and return the first match. If no match is
    found, return None.

    For example:

        >>> _parse_mode_from_benchmark_name(
        ...     "test_VQE_pennylane_fast[10-HeH+-STO-3G]",
        ...     ["pennylane", "compile_and_execution"])
        'pennylane'

        >>> _parse_mode_from_benchmark_name(
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


def pennylane_label(
    ax: matplotlib.axes.Axes,
    x: float = 0.02,
    y: float = 0.98,
    fontsize=16,
    qualifier: str = None,
):
    """Add a PennyLane header label to the given axes.

    Args:
        ax (plt.axes.Axes): Handle to the axes where the label is drawn.
        qualifier (str): A "qualifier" string to append to the PennyLane header
            label, e.g. 'Internal', 'Preliminary', etc.
    """
    if not qualifier:
        label_str = rf"$\mathbfit{{PennyLane}}$"
    else:
        label_str = rf"$\mathbfit{{PennyLane}}$ {qualifier}"

    ax.text(
        x,
        y,
        label_str,
        transform=ax.transAxes,
        fontsize=fontsize,
        verticalalignment="top",
    )


def label(ax: matplotlib.axes.Axes, x: float, y: float, s, **kwargs):
    """A wrapper for matplotlib.axes.Axes.text()"""
    if not s:
        return

    ax.text(x, y, s, transform=ax.transAxes, **kwargs)
