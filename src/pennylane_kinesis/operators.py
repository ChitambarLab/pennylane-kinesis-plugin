import pennylane as qml
from pennylane.operation import Operation


class HWP(Operation):

    num_wires = 1
    grad_method = "A"

    parameter_frequencies = [(2,)]

    def __init__(self, theta, wires, do_queue=True, id=None):
        super().__init__(theta, wires=qml.wires.Wires(wires), do_queue=do_queue, id=id)

    @property
    def num_params(self):
        return 1

    @property
    def ndim_params(self):
        return (0,)

    @staticmethod
    def compute_decomposition(theta, wires):
        return [
            qml.RY(theta, wires=wires[0]),
            qml.PauliZ(wires=wires[0]),
            qml.RY(-1 * theta, wires=wires[0]),
        ]

    # def adjoint(self):
    #     return HWP(-self.parameters[0], self.wires[0], self.wires[1], do_flip=self.hyperparameters["do_flip"])

    @staticmethod
    def compute_matrix(theta):
        interface = qml.math.get_interface(theta)

        c = qml.math.cos(theta)
        s = qml.math.sin(theta)

        # If anything is not tensorflow, it has to be casted and then
        if interface == "tensorflow":
            c = qml.math.cast_like(qml.math.asarray(c, like=interface), 1j)
            s = qml.math.cast_like(qml.math.asarray(s, like=interface), 1j)

        mat = [
            [c**2 - s**2, 2 * c * s],
            [2 * c * s, -(c**2) + s**2],
        ]

        return qml.math.stack([stack_last(row) for row in mat], axis=-2)

    def generator(self):
        return -1 * qml.PauliZ(wires=self.wires[0])


class QWP(Operation):

    num_wires = 1
    grad_method = "A"

    def __init__(self, theta, wires, do_queue=True, id=None):
        super().__init__(theta, wires=qml.wires.Wires(wires), do_queue=do_queue, id=id)

    @property
    def num_params(self):
        return 1

    @property
    def ndim_params(self):
        return (0,)

    @staticmethod
    def compute_decomposition(theta, wires):
        return [
            qml.RY(theta, wires=wires[0]),
            qml.S(wires=wires[0]),
            qml.RY(-1 * theta, wires=wires[0]),
        ]

    @staticmethod
    def compute_matrix(theta):
        interface = qml.math.get_interface(theta)

        c = qml.math.cos(theta)
        s = qml.math.sin(theta)

        # If anything is not tensorflow, it has to be casted and then
        if interface == "tensorflow":
            c = qml.math.cast_like(qml.math.asarray(c, like=interface), 1j)
            s = qml.math.cast_like(qml.math.asarray(s, like=interface), 1j)

        mat = [
            [c**2 + 1j * s**2, (1 - 1j) * c * s],
            [(1 - 1j) * c * s, 1j * c**2 + s**2],
        ]

        return qml.math.stack([stack_last(row) for row in mat], axis=-2)

    # def adjoint(self):
    #     # the adjoint operator of this gate simply negates the theta
    #     return FlipAndRotate(-self.parameters[0], self.wires[0], self.wires[1], do_flip=self.hyperparameters["do_flip"])
