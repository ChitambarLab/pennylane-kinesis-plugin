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
        assert op.parameter_frequencies == [(1, 2)]
        assert op.id == None

    def match_hwp_operator_matrix(self, theta):
        return (
            qml.RY(theta, wires=[0]).matrix()
            @ qml.PauliZ(0).matrix()
            @ qml.RY(-theta, wires=[0]).matrix()
        )

    @pytest.mark.parametrize("theta", [theta for theta in np.arange(0, 2 * np.pi, 0.5)])
    def test_hwp_operator_computer_matrix(self, theta):
        op = pk.HWPRotStage(theta, wires=[0])

        assert np.allclose(op.matrix(), self.match_hwp_operator_matrix(theta))

    def match_hwp_grad(self, theta):
        dev = qml.device("default.qubit", wires=[0])

        @qml.qnode(dev, diff_method="parameter-shift")
        def circ(phi):
            qml.RY(-phi, wires=[0])
            qml.PauliZ(wires=[0])
            qml.RY(phi, wires=[0])

            return qml.expval(qml.Projector([1], wires=[0]))

        return qml.grad(circ)(qnp.array(theta, requires_grad=True))

    @pytest.mark.parametrize("theta", np.arange(0, 2 * np.pi, 0.5))
    def test_hwp_grad(self, theta):
        dev = qml.device("kinesis.simulator", wires=[0], shots=100000)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circ(phi):
            pk.HWPRotStage(phi, wires=[0])

            return qml.expval(qml.Projector([1], wires=[0]))

        assert np.isclose(
            qml.grad(circ)(qnp.array(theta, requires_grad=True)),
            self.match_hwp_grad(theta),
            atol=5e-3,
        )


class TestQWPRotStage:
    def test_qwp_operator_attributes(self):

        op = pk.QWPRotStage(0.123, wires=[0])

        assert isinstance(op, qml.operation.Operation)

        assert op.num_wires == 1
        assert op.num_params == 1
        assert op.ndim_params == (0,)
        assert op.grad_method == "A"
        assert op.parameter_frequencies == [(1, 2)]
        assert op.id == None

    def match_qwp_operator_matrix(self, theta):
        return (
            qml.RY(theta, wires=[0]).matrix()
            @ qml.S(0).matrix()
            @ qml.RY(-theta, wires=[0]).matrix()
        )

    @pytest.mark.parametrize("theta", [theta for theta in np.arange(0, 2 * np.pi, 0.5)])
    def test_qwp_operator_computer_matrix(self, theta):
        op = pk.QWPRotStage(theta, wires=[0])

        assert np.allclose(op.matrix(), self.match_qwp_operator_matrix(theta))

    def match_qwp_grad(self, theta):
        dev = qml.device("default.qubit", wires=[0])

        @qml.qnode(dev, diff_method="parameter-shift")
        def circ(phi):
            qml.RY(-phi, wires=[0])
            qml.S(wires=[0])
            qml.RY(phi, wires=[0])

            return qml.expval(qml.Projector([1], wires=[0]))

        return qml.grad(circ)(qnp.array(theta, requires_grad=True))

    @pytest.mark.parametrize("theta", np.arange(0, 2 * np.pi, 0.5))
    def test_qwp_grad(self, theta):
        dev = qml.device("kinesis.simulator", wires=[0], shots=100000)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circ(phi):
            pk.QWPRotStage(phi, wires=[0])

            return qml.expval(qml.Projector([1], wires=[0]))

        assert np.isclose(
            qml.grad(circ)(qnp.array(theta, requires_grad=True)),
            self.match_qwp_grad(theta),
            atol=5e-3,
        )
