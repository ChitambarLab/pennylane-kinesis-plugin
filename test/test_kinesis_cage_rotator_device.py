import pytest
import pennylane as qml
from pennylane import numpy as qnp

import numpy as np

import pennylane_kinesis as pk


def _setup_device(interface, wimp_mode):
    dev_kwargs = {
        "shots": 500,
        "shot_rate": 1000,
        "wires": [0],
        "test_mode": True,
        "wimp_mode": wimp_mode,
    }

    dev = None

    if interface == "class":
        dev = pk.KinesisCageRotatorDevice(**dev_kwargs)
    elif interface == "pennylane":
        dev = qml.device("kinesis.cage.rotator", **dev_kwargs)
    else:
        raise ValueError(
            "interface: '" + interface + "', unkown. Please use 'class' or 'pennylane'."
        )

    print("Dev test mode : ", dev.test_mode)

    return dev


@pytest.mark.parametrize("interface", ["class", "pennylane"])
@pytest.mark.parametrize("wimp_mode", [True, False])
class TestKinesisSimulatorDevice:
    def test_device_attributes(self, interface, wimp_mode):
        dev = _setup_device(interface, wimp_mode)

        assert dev.test_mode

        # pennylane-kinesis attributes
        assert dev.name == "Kinesis Cage Rotator Device"
        assert dev.short_name == "kinesis.cage.rotator"
        assert dev.pennylane_requires == "0.28.0"
        assert dev.version == pk.__version__
        assert dev.author == "Brian Doolittle and Colin Lualdi"
        assert dev.operations == {"HWPRotStage", "QWPRotStage"}
        assert dev.observables == {"Projector", "Identity", "PauliZ"}
        assert not dev.analytic
        assert all(dev.init_state == [1, 0])

        # test-defined attributes
        assert dev.shots == 500
        assert dev.shot_rate == 1000
        assert dev.num_wires == 1
        assert list(dev.wires) == [0]

        assert dev.integration_time == 0.5

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

        assert dev.a_hwp_config["device_vel"] == 5 if wimp_mode else 10
        assert dev.a_qwp_config["device_vel"] == 5 if wimp_mode else 10

        assert dev.buffer == {}

        assert dev.wimp_mode == wimp_mode

    def test_qnode_execution(self, interface, wimp_mode):
        dev = _setup_device(interface, wimp_mode)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circ(settings):
            pk.HWPRotStage(settings[0], wires=dev.wires[0])
            pk.QWPRotStage(settings[1], wires=dev.wires[0])

            return qml.expval(qml.Projector([1], wires=[0]))

        assert circ([0.5, 0.5]) == 0.5

    def test_device_qnode_samples(self, interface, wimp_mode):
        dev = _setup_device(interface, wimp_mode)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circ(settings):
            pk.HWPRotStage(settings[0], wires=dev.wires[0])
            pk.QWPRotStage(settings[1], wires=dev.wires[0])

            return qml.sample(wires=dev.wires[0])

        samples = circ([0.5, 0.5])

        assert len(samples) == 500
        assert all([s == 0 or s == 1 for s in samples])
