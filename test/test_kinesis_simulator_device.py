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
        dev = pk.KinesisSimulatorDevice(shots=shots, wires=wires)
    elif interface == "pennylane":
        dev = qml.device("kinesis.simulator", shots=shots, wires=wires)
    else:
        raise ValueError(
            "interface: '" + interface + "', unkown. Please use 'class' or 'pennylane'."
        )

    return dev


def _HWP(theta, wires):
    qml.RY(-1 * theta, wires=wires)
    qml.PauliZ(wires=wires)
    qml.RY(theta, wires=wires)


def _QWP(theta, wires):
    qml.RY(-1 * theta, wires=wires)
    qml.S(wires=wires)
    qml.RY(theta, wires=wires)


@pytest.mark.parametrize("interface", ["class", "pennylane"])
class TestKinesisSimulatorDevice:
    def test_device_attributes(self, interface):

        dev = _setup_device(interface)

        # pennylane-kinesis attributes
        assert dev.name == "Kinesis Simulator Device"
        assert dev.short_name == "kinesis.simulator"
        assert dev.pennylane_requires == "0.28.0"
        assert dev.version == pk.__version__
        assert dev.author == "Brian Doolittle and Colin Lualdi"
        assert dev.operations == {
            "PauliZ",
            "PhaseShift",
            "RX",
            "RY",
            "RZ",
            "S",
            "HWPRotStage",
            "QWPRotStage",
        }
        assert dev.observables == {"Projector", "Identity", "PauliZ"}
        assert not dev.analytic
        assert all(dev.init_state == [1, 0])

        # test-defined attributes
        assert dev.shots == 10000
        assert dev.num_wires == 1
        assert list(dev.wires) == [0]

    @pytest.mark.parametrize(
        "ops,match",
        [
            ([qml.RY(0, [0]), qml.PauliZ([0]), qml.RY(0, [0])], [1, 0]),
            ([qml.RY(-np.pi, [0]), qml.S([0]), qml.RY(np.pi, [0])], [1j, 0]),
        ],
    )
    def test_device_apply(self, interface, ops, match):

        dev = _setup_device(interface)
        assert all(dev._state == dev.init_state)

        dev.apply(ops)
        assert np.allclose(dev._state, match)

        dev.reset()
        assert all(dev._state == dev.init_state)

    def test_device_qnode_projector_expval(self, interface):

        dev = _setup_device(interface)

        @qml.qnode(dev)
        def circ(theta):
            _HWP(theta, wires=[0])
            return qml.expval(qml.Projector([1], wires=[0]))

        assert np.isclose(circ(0), 0, atol=1e-2)
        assert np.isclose(circ(np.pi / 4), 0.5, atol=2e-2)
        assert np.isclose(circ(np.pi / 2), 1, atol=1e-2)

    def test_device_qnode_probs(self, interface):

        dev = _setup_device(interface)

        @qml.qnode(dev)
        def circ(theta):
            _HWP(theta, wires=[0])
            return qml.probs(wires=[0])

        assert np.allclose(circ(0), [1, 0], atol=1e-2)
        assert np.allclose(circ(np.pi / 4), [0.5, 0.5], atol=2e-2)
        assert np.allclose(circ(np.pi / 2), [0, 1], atol=1e-2)

    def test_device_qnode_samples(self, interface):

        np.random.seed(41)

        dev = _setup_device(interface)

        @qml.qnode(dev)
        def circ(theta):
            _QWP(theta, wires=[0])
            return qml.sample(wires=[0])

        samples = circ(np.pi / 4)

        assert len(samples) == 10000
        assert all([s == 0 or s == 1 for s in samples])

    def test_device_parameter_shift_opt(self, interface):

        dev = _setup_device(interface)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circ(settings):
            theta, phi = settings
            _HWP(theta, wires=[0])
            _QWP(phi, wires=[0])

            # fiber transformation to invert
            qml.Rot(np.pi / 4, np.pi / 2, np.pi / 3, wires=[0])

            # minimize projection onto |1> state
            return qml.expval(qml.Projector([1], wires=[0]))

        opt = qml.GradientDescentOptimizer(stepsize=0.4)

        settings = qnp.zeros(2)
        cost_vals = []
        for i in range(40):
            settings, cost = opt.step_and_cost(circ, settings)
            cost_vals += [cost]

        assert np.isclose(min(cost_vals), 0, atol=5e-2)
