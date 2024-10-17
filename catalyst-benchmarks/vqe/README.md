# Catalyst Benchmarks: VQE

Catalyst benchmarking using the Variational Quantum Eigensolver (VQE) program.
These benchmarks were inspired by the [VQE PennyLane benchmarks](https://github.com/PennyLaneAI/pennylane-benchmarks/tree/main/benchmarks/VQE).

## Usage

First-time setup:

```console
$ python -m venv venv-benchmarks
$ source ./venv-benchmarks/bin/activate
$ pip install -r requirements.txt
```

Decide which release of PennyLane/Lightning/Catalyst to use and install with, for example:

```console
$ pip install PennyLane-Catalyst==0.8.1
```

To use a pre-release version, you can do, for example:

```console
$ pip install -i https://test.pypi.org/simple/ PennyLane-Catalyst==0.9.0.dev37
```

To run the benchmarks, do:

```console
$ make benchmarks
```

and to plot the results:

```console
$ make plots
```

By default, this will create a new directory `plots` where the results are saved.