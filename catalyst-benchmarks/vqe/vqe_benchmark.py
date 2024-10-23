from typing import Callable

import pennylane as qml
import pytest
from conftest import create_device, device_name
from pennylane import numpy as np

import catalyst
from jax import numpy as jnp

mols_basis_sets = [
    ["H2", "STO-3G"],  #  4 /   15
    ["HeH+", "STO-3G"],  #  4 /   27
    ["H3+", "STO-3G"],  #  6 /   66
    ["He2", "6-31G"],  #  8 /  181
    ["H2", "6-31G"],  #  8 /  185
    ["H4", "STO-3G"],  #  8 /  185
    ["LiH", "STO-3G"],  # 12 /  631
    ["OH-", "STO-3G"],  # 12 /  631
    ["H3+", "6-31G"],  # 12 / 1403
    ["BeH2", "STO-3G"],  # 14 /  666
    ["H2O", "STO-3G"],  # 14 / 1086
    ["H2", "CC-PVDZ"],  # 20 / 2951
    ["C2H2", "STO-3G"],  # 24 / 6500
]


def workload_VQE(mol: str, basis_set: str, mode: str, it_max: int = 10):
    """Generate a VQE workload for a given molecule and basis set.

    Args:
        mol (str): molecule
        basis_set (str): basis set
        mode (str): Choose from "pennylane" to return the PennyLane native
            workload or "catalyst" to return the QJIT compiled workload
        it_max (int): Maximum number of iterations; if no stopping condition is
            given, the VQE algorithm will run exactly this many iterations
    """
    dataset = qml.data.load("qchem", molname=mol, basis=basis_set)[0]
    ham, _ = dataset.hamiltonian, len(dataset.hamiltonian.wires)
    hf_state = dataset.hf_state
    ham = dataset.hamiltonian
    wires = ham.wires
    dev = create_device(device_name, wires)
    n_electrons = dataset.molecule.n_electrons

    singles, doubles = qml.qchem.excitations(n_electrons, len(wires))

    @qml.qnode(dev, diff_method="adjoint")
    def cost(weights):
        qml.templates.AllSinglesDoubles(weights, wires, hf_state, singles, doubles)
        return qml.expval(ham)

    step_size = 0.2

    def workload():
        np.random.seed(42)
        params = np.random.normal(0, np.pi, len(singles) + len(doubles))

        opt = qml.GradientDescentOptimizer(stepsize=step_size)

        for _ in range(it_max):
            params, _ = opt.step_and_cost(cost, params)
        return params

    def workload_catalyst():

        def grad_descent_catalyst(params, cost_fn: Callable):
            diff = catalyst.grad(cost_fn)
            theta = params

            # for_loop can only be used in JIT mode
            @catalyst.for_loop(0, it_max, 1)
            def single_step(i, theta):
                h = diff(theta)
                return theta - h * step_size

            return single_step(theta)

        np.random.seed(42)
        params = jnp.array(np.random.normal(0, np.pi, len(singles) + len(doubles)))

        params = grad_descent_catalyst(params=params, cost_fn=cost)

        return params

    if mode in ["p", "pennylane"]:
        return workload
    elif mode in ["c", "catalyst"]:
        return qml.qjit(workload_catalyst)
    else:
        raise ValueError(f"unknown mode '{mode}'")


fast_indices = 4
extra_fast_indices = 2

# it_maxs = list(range(10, 101, 10))
it_maxs = [10, 50, 100, 200, 300, 400, 500]
# it_maxs = [10]

@pytest.mark.fast
@pytest.mark.pennylane
@pytest.mark.parametrize("mol, basis_set", mols_basis_sets[:extra_fast_indices])
@pytest.mark.parametrize("it_max", it_maxs)
def test_VQE_pennylane_fast(mol, basis_set, it_max, benchmark):
    workload = workload_VQE(mol, basis_set, mode="pennylane", it_max=it_max)
    benchmark(workload)
    

@pytest.mark.slow
@pytest.mark.pennylane
@pytest.mark.parametrize("mol, basis_set", mols_basis_sets[4:11])
def test_VQE_pennylane_slow(mol, basis_set, benchmark):
    workload = workload_VQE(mol, basis_set, mode="pennylane")
    benchmark(workload)


@pytest.mark.fast
@pytest.mark.catalyst_compilation
@pytest.mark.parametrize("mol, basis_set", mols_basis_sets[:extra_fast_indices])
@pytest.mark.parametrize("it_max", it_maxs)
def test_VQE_catalyst_compile_and_execution_fast(mol, basis_set, it_max, benchmark):

    def workflow():
        return workload_VQE(mol, basis_set, mode="catalyst", it_max=it_max)()

    benchmark(workflow)


@pytest.mark.slow
@pytest.mark.catalyst_compilation
@pytest.mark.parametrize("mol, basis_set", mols_basis_sets[4:11])
def test_VQE_catalyst_compile_and_execution_slow(mol, basis_set, benchmark):

    def workflow():
        workload_VQE(mol, basis_set, mode="catalyst")()

    benchmark(workflow)


@pytest.mark.fast
@pytest.mark.catalyst
@pytest.mark.parametrize("mol, basis_set", mols_basis_sets[:extra_fast_indices])
@pytest.mark.parametrize("it_max", it_maxs)
def test_VQE_catalyst_execution_only_fast(mol, basis_set, it_max, benchmark):
    workload = workload_VQE(mol, basis_set, mode="catalyst", it_max=it_max)
    benchmark(workload)


@pytest.mark.slow
@pytest.mark.catalyst
@pytest.mark.parametrize("mol, basis_set", mols_basis_sets[4:11])
def test_VQE_catalyst_execution_only_slow(mol, basis_set, benchmark):
    workload = workload_VQE(mol, basis_set, mode="catalyst")
    benchmark(workload)

