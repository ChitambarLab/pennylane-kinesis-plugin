import pytest
import pennylane as qml
from pennylane import numpy as qnp

import numpy as np

import pennylane_kinesis as pk


def _setup_device(interface, analytic=False):
    shots = 10000 if not analytic else None
    wires = [0]
    dev = None

    if interface == "class":
        dev = pk.KinesisSimulatorDevice(shots=shots, wires=wires, seed=37)
    elif interface == "pennylane":
        dev = qml.device("kinesis.simulator", shots=shots, wires=wires, seed=37)
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
    @pytest.mark.parametrize("analytic", [False, True])
    def test_device_attributes(self, interface, analytic):
        dev = _setup_device(interface, analytic)

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
        assert dev.analytic == analytic
        assert all(dev.init_state == [1, 0])

        # test-defined attributes
        assert dev.shots == None if analytic else 10000
        assert dev.num_wires == 1
        assert list(dev.wires) == [0]

    @pytest.mark.parametrize(
        "ops,match",
        [
            ([qml.RY(qnp.array(0.0), [0]), qml.PauliZ([0]), qml.RY(qnp.array(0.0), [0])], [1, 0]),
            ([qml.RY(qnp.array(-np.pi), [0]), qml.S([0]), qml.RY(qnp.array(np.pi), [0])], [1j, 0]),
        ],
    )
    def test_device_apply(self, interface, ops, match):
        dev = _setup_device(interface)
        assert all(dev._state == dev.init_state)

        dev.apply(ops)
        assert np.allclose(dev._state, match)

        dev.reset()
        assert all(dev._state == dev.init_state)

    @pytest.mark.parametrize(
        "analytic, atol",
        [
            (True, 1e-6),
            (False, 2e-2),
        ],
    )
    def test_device_qnode_projector_expval(self, interface, analytic, atol):
        dev = _setup_device(interface, analytic)

        @qml.qnode(dev)
        def circ(theta):
            _HWP(theta, wires=[0])
            return qml.expval(qml.Projector([1], wires=[0]))

        assert np.isclose(circ(0), 0, atol=atol)
        assert np.isclose(circ(np.pi / 4), 0.5, atol=atol)
        assert np.isclose(circ(np.pi / 2), 1, atol=atol)

    @pytest.mark.parametrize(
        "analytic, atol",
        [
            (True, 1e-6),
            (False, 2e-2),
        ],
    )
    def test_device_qnode_probs(self, interface, analytic, atol):
        dev = _setup_device(interface, analytic)
        atol = 1e-6 if analytic else 2e-2

        @qml.qnode(dev)
        def circ(theta):
            _HWP(theta, wires=[0])
            return qml.probs(wires=[0])

        assert np.allclose(circ(0), [1, 0], atol=atol)
        assert np.allclose(circ(np.pi / 4), [0.5, 0.5], atol=atol)
        assert np.allclose(circ(np.pi / 2), [0.0, 1], atol=atol)

    def test_device_qnode_samples(self, interface):
        dev = _setup_device(interface)

        @qml.qnode(dev)
        def circ(theta):
            _QWP(theta, wires=[0])
            return qml.sample(wires=[0])

        samples = circ(np.pi / 4)

        assert len(samples) == 10000
        assert all([s == 0 or s == 1 for s in samples])

    @pytest.mark.parametrize(
        "analytic, atol",
        [
            (True, 5e-2),
            (False, 5e-2),
        ],
    )
    def test_device_parameter_shift_opt(self, interface, analytic, atol):
        dev = _setup_device(interface, analytic)

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

        cost_vals += [circ(settings)]

        assert np.isclose(min(cost_vals), 0, atol=atol)

    @pytest.mark.parametrize("analytic, atol", [(True, 1e-6), (False, 5e-2)])
    def test_device_backprop_opt_match(self, interface, analytic, atol):
        dev_test = _setup_device(interface, analytic=analytic)
        dev_match = qml.device("default.qubit", wires=[0])

        def circ(settings):
            theta, phi = settings
            _HWP(theta, wires=[0])
            _QWP(phi, wires=[0])

            # fiber transformation to invert
            qml.Rot(np.pi / 4, np.pi / 2, np.pi / 3, wires=[0])

            # minimize projection onto |1> state
            return qml.expval(qml.Projector([1], wires=[0]))

        qnode_test = qml.QNode(circ, dev_test, diff_method="parameter-shift")
        qnode_match = qml.QNode(circ, dev_match, diff_method="backprop")

        opt = qml.GradientDescentOptimizer(stepsize=0.4)

        settings_test = qnp.zeros(2)
        settings_match = qnp.zeros(2)
        cost_vals_test = []
        cost_vals_match = []
        for i in range(20):
            settings_test, cost_test = opt.step_and_cost(qnode_test, settings_test)
            cost_vals_test += [cost_test]

            settings_match, cost_match = opt.step_and_cost(qnode_match, settings_match)
            cost_vals_match += [cost_match]

            assert np.allclose(settings_test, settings_match, atol=atol)
            assert np.isclose(cost_test, cost_match, atol=atol)
