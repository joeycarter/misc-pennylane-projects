from itertools import product

import numpy as np
import pennylane as qml
import pytest
from conftest import device_name, selective_qjit
from pennylane.qchem.convert import _wfdict_to_statevector
from pyscf import gto, mcscf, scf
from pyscf.fci.cistring import addrs2str
from scipy.linalg import expm
from scipy.sparse import coo_matrix


def _casci_state(casci_solver, state=0, tol=1e-15):
    """Return a dictionary describing the initial state."""
    norb = casci_solver.mol.nao

    nelec_a = casci_solver.mol.nelectron // 2
    nelec_b = casci_solver.mol.nelectron // 2

    try:
        ncore_a, ncore_b = casci_solver.ncore
    except TypeError:
        ncore_a = casci_solver.ncore
        ncore_b = ncore_a

    ncas_a = norb - ncore_a
    ncas_b = norb - ncore_b

    nelecas_a = nelec_a - ncore_a
    nelecas_b = nelec_b - ncore_b

    ## extract the CI coeffs from the right state
    assert state in range(casci_solver.fcisolver.nroots), (
        f"State requested has not " f"been solved for. Re-run with larger nroots."
    )
    cascivec = casci_solver.ci[state] if casci_solver.fcisolver.nroots > 1 else casci_solver.ci

    # filter out small values based on preset tolerance to save more memory
    cascivec[abs(cascivec) < tol] = 0
    sparse_cascimatr = coo_matrix(cascivec, shape=np.shape(cascivec), dtype=float)
    row, col, dat = sparse_cascimatr.row, sparse_cascimatr.col, sparse_cascimatr.data

    ## turn indices into strings
    strs_row = addrs2str(ncas_a, nelecas_a, row)
    strs_col = addrs2str(ncas_b, nelecas_b, col)

    ## create the FCI matrix as a dict
    dict_fcimatr = dict(zip(list(zip(strs_row, strs_col)), dat))

    return dict_fcimatr


def circuit_two_body(U, Z, h):
    """Append a Trotter iteration to the operation queue."""
    ctrl_wires = 1
    trgt_wires = np.arange(U.shape[-1], dtype=int)

    qml.BasisRotation(unitary_matrix=U, wires=2 * trgt_wires + ctrl_wires)
    qml.BasisRotation(unitary_matrix=U, wires=2 * trgt_wires + ctrl_wires + 1)

    for sigma, tau in product(range(2), repeat=2):
        for i, k in product(trgt_wires, repeat=2):
            if i == k and sigma == tau:
                continue
            if ctrl_wires:
                qml.ctrl(
                    qml.MultiRZ(
                        Z[i, k] / 4.0 * h,
                        wires=[2 * i + sigma + ctrl_wires, 2 * k + tau + ctrl_wires],
                    ),
                    control=range(ctrl_wires),
                )
            else:
                qml.MultiRZ(Z[i, k] / 4.0 * h, wires=[2 * i + sigma, 2 * k + tau])

    qml.BasisRotation(unitary_matrix=U.T, wires=2 * trgt_wires + ctrl_wires)
    qml.BasisRotation(unitary_matrix=U.T, wires=2 * trgt_wires + ctrl_wires + 1)

    phase = -np.trace(Z) / 4 * h
    qml.PhaseShift(phase, wires=0)


def XAS_workload(ncas, N1, num_steps, mode):
    """Generate a workload for a circuit used in computing the X-ray absorption spectrum of an N2 molecule.
    The time evolution is approximated using a two-body cumulative distribution function.
    The circuit therefore consist in a ``StatePrep`` operation, followed be a loop over
    time steps, which itself contains a loop over two-body terms. The circuit therefore
    quickly becomes fairly deep. The number of qubits varies proportionally to the size
    of the active subspace.

    Args:
        ncas (int): size of the active subspace.
        N1 (int): multiplicative factor yielding the number of CDF blocks (N1 * ncas).
        num_steps (int): number of time steps used to divide the time evolution.
    """
    # Initialize system
    mol = gto.Mole(atom=[["N", (0, 0, 0)], ["N", (0, 0, 1.077)]], basis="6-31G", symmetry="d2h")
    mol.build()
    hf = scf.RHF(mol).run(verbose=0)
    nelecas = 4
    # Initialize state
    mycasci = mcscf.CASCI(hf, ncas=ncas, nelecas=nelecas)
    mycasci.run(verbose=0)
    wf = _casci_state(mycasci, tol=1e-3)
    wf_dict = _wfdict_to_statevector(wf, ncas)
    # Random integrals
    X = np.random.rand(N1 * ncas, ncas, ncas)
    Z = np.random.rand(N1 * ncas, ncas, ncas)
    U = expm(X)
    # Initialize device
    dev = qml.device(device_name, wires=2 * ncas + 1)
    dev_wires = dev.wires.tolist()

    h = 1.0 / num_steps

    @qml.qnode(dev)
    def circuit(U=U, Z=Z, h=h, wf_dip=wf_dict, num_steps=num_steps):
        qml.StatePrep(wf_dip, wires=dev_wires[1:])
        qml.Hadamard(dev_wires[0])
        for _ in range(num_steps):
            for uj, zj in zip(U, Z):
                circuit_two_body(uj, zj, h)
        return [qml.expval(op) for op in [qml.PauliX(dev_wires[0]), qml.PauliY(dev_wires[0])]]

    if mode in ["p", "pennylane"]:
        return circuit
    elif mode in ["c", "catalyst"]:
        return qml.qjit(circuit)
    else:
        raise ValueError(f"unknown mode '{mode}'")


@pytest.mark.dependency()
@pytest.mark.warmup
def test_warmup_XAS():
    """Warm-up XAS algorithm."""
    XAS_workload(5, 1, 1)()


num_steps = [1, 5, 10]

@pytest.mark.fast
@pytest.mark.pennylane
@pytest.mark.dependency(depends=["test_warmup_XAS"])
@pytest.mark.parametrize("ncas", [3])
@pytest.mark.parametrize("N1", [1, 2])
@pytest.mark.parametrize("num_steps", num_steps)
def test_XAS_pennylane_fast(benchmark, ncas, N1, num_steps):
    workflow = XAS_workload(ncas, N1, num_steps, mode="pennylane")
    benchmark(workflow)


@pytest.mark.fast
@pytest.mark.catalyst
@pytest.mark.dependency(depends=["test_warmup_XAS"])
@pytest.mark.parametrize("ncas", [3])
@pytest.mark.parametrize("N1", [1, 2])
@pytest.mark.parametrize("num_steps", num_steps)
def test_XAS_catalyst_execution_only_fast(benchmark, ncas, N1, num_steps):
    workflow = XAS_workload(ncas, N1, num_steps, mode="catalyst")
    benchmark(workflow)


@pytest.mark.fast
@pytest.mark.catalyst_compilation
@pytest.mark.dependency(depends=["test_warmup_XAS"])
@pytest.mark.parametrize("ncas", [3])
@pytest.mark.parametrize("N1", [1, 2])
@pytest.mark.parametrize("num_steps", num_steps)
def test_XAS_compile_and_execution_fast(benchmark, ncas, N1, num_steps):
    def workflow():
        return XAS_workload(ncas, N1, num_steps, mode="catalyst")()

    benchmark(workflow)


@pytest.mark.slow
@pytest.mark.catalyst
@pytest.mark.dependency(depends=["test_warmup_XAS"])
@pytest.mark.parametrize("ncas", [5, 7])
@pytest.mark.parametrize("N1", [1, 2])
@pytest.mark.parametrize("num_steps", num_steps)
def test_XAS_slow(benchmark, ncas, N1, num_steps):
    workflow = selective_qjit(XAS_workload(ncas, N1, num_steps))
    benchmark(workflow)


@pytest.mark.slow
@pytest.mark.catalyst_compilation
@pytest.mark.dependency(depends=["test_warmup_XAS"])
@pytest.mark.parametrize("ncas", [5, 7])
@pytest.mark.parametrize("N1", [1, 2])
@pytest.mark.parametrize("num_steps", num_steps)
def test_XAS_slow_with_compilation(benchmark, ncas, N1, num_steps):
    def workflow():
        selective_qjit(XAS_workload(ncas, N1, num_steps))()

    benchmark(workflow)
