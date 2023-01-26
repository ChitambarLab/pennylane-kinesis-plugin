import pytest
import pennylane as qml
from pennylane import numpy as qnp

import numpy as np

import pennylane_kinesis as pk


def _setup_device(interface):

    shots = 10000
    wires = [0]
    dev = None

    if interface == "class":
        dev = pk.KinesisCageRotatorDevice(shots=shots, wires=wires)
    elif interface == "pennylane":
        dev = qml.device("kinesis.cage.rotator", shots=shots, wires=wires)
    else:
        raise ValueError(
            "interface: '" + interface + "', unkown. Please use 'class' or 'pennylane'."
        )

    dev.test_mode = True

    return dev


@pytest.mark.parametrize("interface", ["class", "pennylane"])
class TestKinesisSimulatorDevice:
    def test_device_attributes(self, interface):

        dev = _setup_device(interface)

        assert dev.test_mode

        # pennylane-kinesis attributes
        assert dev.name == "Kinesis Cage Rotator Device"
        assert dev.short_name == "kinesis.cage.rotator"
        assert dev.pennylane_requires == "0.28.0"
        assert dev.version == pk.__version__
        assert dev.author == "Brian Doolittle"
        assert dev.operations == {"HWPRotStage", "QWPRotStage"}
        assert dev.observables == {"Projector", "Identity", "PauliZ"}
        assert not dev.analytic
        assert all(dev.init_state == [1, 0])

        # test-defined attributes
        assert dev.shots == 10000
        assert dev.num_wires == 1
        assert list(dev.wires) == [0]

        assert dev.integration_time == 1

        assert dev.device_list() == []

        assert dev.a_hwp_sn == "55253334"
        assert dev.a_qwp_sn == "55253344"
        assert dev.b_hwp_sn == "55250324"
        assert dev.b_qwp_sn == "55246004"

        assert dev.a_hwp_off == 100.29
        assert dev.a_qwp_off == 112.21
        assert dev.b_hwp_off == 86.64
        assert dev.b_qwp_off == 12.02

        assert dev.dev_a_hwp["sn"] == "55253334"
        assert dev.dev_a_qwp["sn"] == "55253344"

        assert dev.a_hwp_config["device_vel"] == 5
        assert dev.a_qwp_config["device_vel"] == 5

        assert dev.buffer == {}

    def test_qnode_execution(self, interface):
        dev = _setup_device(interface)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circ(settings):
            pk.HWPRotStage(settings[0], wires=dev.wires[0])
            pk.QWPRotStage(settings[1], wires=dev.wires[0])

            return qml.expval(qml.Projector([1], wires=[0]))

        assert circ([0.5, 0.5]) == 0.5

    def test_device_qnode_samples(self, interface):

        dev = _setup_device(interface)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circ(settings):
            pk.HWPRotStage(settings[0], wires=dev.wires[0])
            pk.QWPRotStage(settings[1], wires=dev.wires[0])

            return qml.sample(wires=dev.wires[0])

        samples = circ([0.5, 0.5])

        assert len(samples) == 10000
        assert all([s == 0 or s == 1 for s in samples])
