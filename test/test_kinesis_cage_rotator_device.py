import pytest
import pennylane as qml
from pennylane import numpy as qnp

import numpy as np

import pennylane_kinesis as pk


def _setup_device(interface, analytic=False, domain_period=False):
    shots = 10000 if not analytic else None
    wires = [0]
    dev = None
    qwp_domain_period = np.pi / 2 if domain_period else None
    hwp_domain_period = np.pi / 2 if domain_period else None

    if interface == "class":
        dev = pk.KinesisSimulatorDevice(
            shots=shots,
            wires=wires,
            seed=37,
            qwp_domain_period=qwp_domain_period,
            hwp_domain_period=hwp_domain_period,
        )
    elif interface == "pennylane":
        dev = qml.device(
            "kinesis.simulator",
            shots=shots,
            wires=wires,
            seed=37,
            qwp_domain_period=qwp_domain_period,
            hwp_domain_period=hwp_domain_period,
        )
    else:
        raise ValueError("interface: '" + interface + "', unkown. Please use 'class' or 'pennylane'.")

    return dev


def _HWP(theta, wires):
    qml.RY(-1 * theta, wires=wires)
    qml.PauliZ(wires=wires)
    qml.RY(theta, wires=wires)


def _QWP(theta, wires):
    qml.RY(-1 * theta, wires=wires)
    qml.S(wires=wires)
    qml.RY(theta, wires=wires)
