# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Pytest configuration file for PennyLane-Benchmarks suite.
"""
import json
import os
from pathlib import Path
from typing import Callable

import pennylane as qml
import pytest

try:
    import catalyst
    import jax.numpy as jnp

    catalyst_available = True
except ImportError:
    catalyst_available = False


def selective_qjit(fn: Callable = None, *args, **kwargs) -> Callable:
    """This function qjit-annotates the provided function depending on Catalyst availability.

    Args:
        fn (Callable, optional): The function to be annotated or not. Defaults to None.

    Returns:
        Callable: If catalyst available, qjit-annotated function if not return the provided function.
    """
    return qml.qjit(fn, *args, **kwargs) if catalyst_available else fn


def catalyst_workload_dispatcher(fn: Callable = None, fn_catalyst: Callable = None) -> Callable:
    """This function dispatches to a non-Catalyst or a Catalyst compatible function depending of availability.

    Args:
        fn (Callable, optional): A function not using qjit annotations. Defaults to None.
        fn_catalyst (Callable, optional): A function using qjit annotations (Catalyst compatible). Defaults to None.

    Returns:
        Callable: The selected function.
    """
    return qml.qjit(fn_catalyst) if catalyst_available else fn


# Looking for the device for benchmarking.
default_device = "lightning.qubit"
supported_devices = {"default.qubit", "lightning.qubit"}
supported_devices.update({sb.replace(".", "_") for sb in supported_devices})


def get_device() -> str:
    """Return the pennylane device name.

    The device is ``lightning.qubit`` by default. Allowed values are:
    "default.qubit", and "lightning.qubit". An
    underscore can also be used instead of a dot. If the environment
    variable ``DEVICE`` is defined, its value is used. Underscores
    are replaced by dots upon exiting.

    Returns:
        str: The device name.
    """
    device = None
    if "DEVICE" in os.environ:
        device = os.environ.get("DEVICE", default_device)
        device = device.replace("_", ".")
    if device is None:
        device = default_device
    if device not in supported_devices:
        raise ValueError(f"Invalid backend {device}.")
    return device


device_name = get_device()
print("Benchmark default device: ", device_name)


def create_device(device_name: str, wires: int, shots: int = None):
    if device_name == "lightning.qubit":
        return qml.device(device_name, wires=wires, batch_obs=True, shots=shots)
    if device_name == "default.qubit":
        return qml.device(device_name, wires=wires, max_workers=8, shots=shots)


@pytest.fixture
def specs(request):
    def write_specs_to_json_inner(specs):
        path = Path("./footprint.json")
        if path.exists():
            with open("footprint.json", "r") as fid:
                all_specs = json.load(fid)
        else:
            all_specs = dict(test_footprint={})
        specs["resources"] = dict(
            (k, v.__dict__ if hasattr(v, "__dict__") else v)
            for k, v in specs["resources"].__dict__.items()
        )
        all_specs["test_footprint"][request.node.name] = specs
        with open("footprint.json", "w") as fid:
            json.dump(all_specs, fid)

    return write_specs_to_json_inner


def benchmark_and_specs(benchmark, specs, circuit, *args, **kwargs):
    if benchmark.disabled:
        circuit(*args, **kwargs)
        return
    benchmark.pedantic(circuit, args=args, kwargs=kwargs)
    if isinstance(circuit, qml.QNode):
        circuit_specs = qml.specs(circuit, level="device")(*args, **kwargs)
        specs(circuit_specs)
