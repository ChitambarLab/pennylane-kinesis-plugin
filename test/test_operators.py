import pytest
import numpy as np
import pennylane as qml
from pennylane import numpy as qnp

import pennylane_kinesis as pk


class TestHWPRotStage:
    def test_hwp_operator_attributes(self):
        op = pk.HWPRotStage(0.123, wires=[0])

        assert isinstance(op, qml.operation.Operation)

        assert op.num_wires == 1
        assert op.num_params == 1
        assert op.ndim_params == (0,)
        assert op.grad_method == "A"
        assert op.parameter_frequencies == [(4,)]
        assert op.id == None

    def match_hwp_operator_matrix(self, theta):
        return qml.RY(2 * theta, wires=[0]).matrix() @ qml.PauliZ(0).matrix() @ qml.RY(-2 * theta, wires=[0]).matrix()

    @pytest.mark.parametrize("theta", [theta for theta in np.arange(0, np.pi, 0.3)])
    def test_hwp_operator_computer_matrix(self, theta):
        op = pk.HWPRotStage(theta, wires=[0])
        op_pi_phase_shift = pk.HWPRotStage(theta + np.pi, wires=[0])
        op_pi2_phase_shift = pk.HWPRotStage(theta + np.pi / 2, wires=[0])

        assert np.allclose(op.matrix(), self.match_hwp_operator_matrix(theta))
        assert np.allclose(op.matrix(), op_pi_phase_shift.matrix())
        assert np.allclose(op.matrix(), -op_pi2_phase_shift.matrix())

    def match_hwp_grad(self, theta):
        dev = qml.device("default.qubit", wires=[0])

        @qml.qnode(dev, diff_method="backprop")
        def circ(phi):
            qml.RY(-2 * phi, wires=[0])
            qml.PauliZ(wires=[0])
            qml.RY(2 * phi, wires=[0])

            qml.S(wires=[0])
            qml.Hadamard(wires=[0])
            qml.ArbitraryUnitary([0.7, 0.2, 0.3], wires=[0])

            return qml.expval(qml.Projector([1], wires=[0]))

        return qml.grad(circ)(qnp.array(theta, requires_grad=True))

    @pytest.mark.parametrize("theta", np.arange(0, 2 * np.pi, 0.5))
    def test_hwp_grad(self, theta):
        dev = qml.device("kinesis.simulator", wires=[0], shots=None, seed=66)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circ(phi):
            pk.HWPRotStage(phi, wires=[0])

            qml.S(wires=[0])
            qml.Hadamard(wires=[0])
            qml.ArbitraryUnitary([0.7, 0.2, 0.3], wires=[0])

            return qml.expval(qml.Projector([1], wires=[0]))

        assert np.isclose(
            qml.grad(circ)(qnp.array(theta, requires_grad=True)),
            self.match_hwp_grad(theta),
            atol=1e-6,
        )


class TestQWPRotStage:
    def test_qwp_operator_attributes(self):
        op = pk.QWPRotStage(0.123, wires=[0])

        assert isinstance(op, qml.operation.Operation)

        assert op.num_wires == 1
        assert op.num_params == 1
        assert op.ndim_params == (0,)
        assert op.grad_method == "A"
        assert op.parameter_frequencies == [(2, 4)]
        assert op.id == None

    def match_qwp_operator_matrix(self, theta):
        return qml.RY(2 * theta, wires=[0]).matrix() @ qml.S(0).matrix() @ qml.RY(-2 * theta, wires=[0]).matrix()

    @pytest.mark.parametrize("theta", [theta for theta in np.arange(0, np.pi, 0.3)])
    def test_qwp_operator_computer_matrix(self, theta):
        op = pk.QWPRotStage(theta, wires=[0])
        op_pi_phase_shift = pk.QWPRotStage(theta + np.pi, wires=[0])
        op_pi2_phase_shift = pk.QWPRotStage(theta + np.pi / 2, wires=[0])

        assert np.allclose(op.matrix(), self.match_qwp_operator_matrix(theta))
        assert np.allclose(op.matrix(), op_pi_phase_shift.matrix())
        assert np.allclose(op.matrix().real, op_pi2_phase_shift.matrix().imag)
        assert np.allclose(op.matrix().imag, op_pi2_phase_shift.matrix().real)

    def match_qwp_grad(self, theta):
        dev = qml.device("default.qubit", wires=[0])

        @qml.qnode(dev, diff_method="backprop")
        def circ(phi):
            qml.RY(-2 * phi, wires=[0])
            qml.S(wires=[0])
            qml.RY(2 * phi, wires=[0])

            qml.S(wires=[0])
            qml.Hadamard(wires=[0])
            qml.ArbitraryUnitary([0.7, 0.2, 0.3], wires=[0])

            return qml.expval(qml.Projector([1], wires=[0]))

        return qml.grad(circ)(qnp.array(theta, requires_grad=True))

    @pytest.mark.parametrize("theta", np.arange(0, 2 * np.pi, 0.1))
    def test_qwp_grad(self, theta):
        dev = qml.device("kinesis.simulator", wires=[0], shots=None, seed=20)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circ(phi):
            pk.QWPRotStage(phi, wires=[0])

            qml.S(wires=[0])
            qml.Hadamard(wires=[0])
            qml.ArbitraryUnitary([0.7, 0.2, 0.3], wires=[0])

            return qml.expval(qml.Projector([1], wires=[0]))

        assert np.isclose(
            qml.grad(circ)(qnp.array(theta, requires_grad=True)),
            self.match_qwp_grad(theta),
            atol=1e-6,
        )
