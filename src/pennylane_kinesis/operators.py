import pennylane as qml
from pennylane.operation import Operation
import numpy as np
from pennylane import numpy as qnp
from enum import Enum

class WaveplateTypes(Enum):
    HWP = 0
    QWP = 1

QWP_RETARDANCE_A = 1.617103
QWP_RETARDANCE_B = 1.6551547
HWP_RETARDANCE_A = np.pi
HWP_RETARDANCE_B = 3.00416

class HWPRotStage(Operation):
    """A quarter wave plate (HWP) mounted on a rotational stage.

    The HWP decomposes as the following quantum function:

    .. code-block:: python

        def hwp_rot_stage(theta, wires):
            qml.RY(-2 * theta, wires=wires[0])
            qml.PauliZ(wires=wires[0])
            qml.RY(2 * theta, wires=wires[0])

    The HWP rotational stage corresponds to the following unitary matrix:

    .. math::

        HWP(\\theta) = \\begin{pmatrix}
            \\cos^2(\\theta) - \\sin^2(\\theta) & 2\\cos(\\theta)\\sin(\\theta) \\\\
            2\\cos(\\theta)\\sin(\\theta) & - \\cos^2(\\theta) + \\sin^2(\\theta) \\\\
        \\end{pmatrix}

    :param theta: The angle of rotation. 
    :type theta: Float

    :param wires: The wire to which the operator is applied
    :type wires: list-like

    :param id: An optional custom name for the operator. Defaults to ``None``.
    :type: string-like

    :returns: An instantiated and trainable ``HWPRotStage`` operator.
    """

    num_wires = 1
    grad_method = "A"
    parameter_frequencies = [(4,)]

    def __init__(self, angle, wires, id=None):
        super().__init__(angle, wires=qml.wires.Wires(wires), id=id)

    @property
    def num_params(self):
        return 1

    @property
    def ndim_params(self):
        return (0,)

    @staticmethod
    def compute_decomposition(theta, wires):
        return [
            qml.RY(-2 * theta, wires=wires),
            qml.PauliZ(wires=wires),
            qml.RY(2 * theta, wires=wires),
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
            [c**2 - s**2, 2 * c * s],
            [2 * c * s, -(c**2) + s**2],
        ]

        return qml.math.stack([qml.math.stack(row, axis=-1) for row in mat], axis=-2)


class QWPRotStage(Operation):
    """A quarter wave plate (QWP) mounted on a rotational stage.

    The QWP decomposes as the following quantum function:

    .. code-block:: python

        def qwp_rot_stage(theta, wires):
            qml.RY(-2 * theta, wires=wires[0])
            qml.S(wires=wires[0])
            qml.RY(2 * theta, wires=wires[0])

    The QWP rotational stage corresponds to the following unitary matrix:

    .. math::

        QWP(\\theta) = \\begin{pmatrix}
            \\cos^2(\\theta) + i \\sin^2(\\theta) & (1-i)\\cos(\\theta)\\sin(\\theta) \\\\
            (1-i)\\cos(\\theta)\\sin(\\theta) & i \\cos^2(\\theta) + \\sin^2(\\theta) \\\\
        \\end{pmatrix}

    :param theta: The angle of rotation. 
    :type theta: Float

    :param wires: The wire to which the operator is applied
    :type wires: list-like

    :param id: An optional custom name for the operator. Defaults to ``None``.
    :type: string-like

    :returns: An instantiated and trainable ``QWPRotStage`` operator.
    """

    num_wires = 1
    grad_method = "A"
    parameter_frequencies = [(2, 4)]

    def __init__(self, theta, wires, id=None):
        super().__init__(theta, wires=qml.wires.Wires(wires), id=id)

    @property
    def num_params(self):
        return 1

    @property
    def ndim_params(self):
        return (0,)

    @staticmethod
    def compute_decomposition(theta, wires):
        return [
            qml.RY(-2 * theta, wires=wires[0]),
            qml.S(wires=wires[0]),
            qml.RY(2 * theta, wires=wires[0]),
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

        return qml.math.stack([qml.math.stack(row, axis=-1) for row in mat], axis=-2)

class VariableWPRotStage(Operation):
    """A variable retardance waveplate mounted on a rotational stage.

    The WP decomposes as the following quantum function:

    .. code-block:: python

        def qwp_rot_stage(theta, wires, retardance):
            qml.RY(-2 * theta, wires=wires[0])
            qml.PhaseShift(retardance, wires=wires[0])
            qml.RY(2 * theta, wires=wires[0])

    The WP rotational stage corresponds to the following unitary matrix:

    .. math::

        QWP(\\theta) = \\exp{-i\\eta}\\begin{pmatrix}
            \\cos^2(\\theta) + \\exp{i\\eta}\\sin^2(\\theta) & (1-\\exp{i\\eta})\\cos(\\theta)\\sin(\\theta) \\\\
            (1-\\exp{i\\eta})\\cos(\\theta)\\sin(\\theta) & \\exp{i\\eta}\\cos^2(\\theta) + \\sin^2(\\theta) \\\\
        \\end{pmatrix}

    :param theta: The angle of rotation. 
    :type theta: Float

    :param wires: The wire to which the operator is applied
    :type wires: list-like

    :param id: An optional custom name for the operator. Defaults to ``None``.
    :type: string-like

    :returns: An instantiated and trainable ``QWPRotStage`` operator.
    """

    num_wires = 1
    grad_method = "A"
    parameter_frequencies = [(2, 4)]
    hyperparameters = {"retardance": None, "wp_type": None}

    def __init__(self, theta, wires, id=None, retardance=np.pi / 2, wp_type: WaveplateTypes = WaveplateTypes.QWP):
        super().__init__(theta, wires=qml.wires.Wires(wires), id=id)
        self.hyperparameters["retardance"] = retardance
        self.hyperparameters["wp_type"] = wp_type

    @property
    def num_params(self):
        return 1

    @property
    def ndim_params(self):
        return (0,)

    @staticmethod
    def compute_decomposition(theta, wires,retardance):
        return [
            qml.RY(-2 * theta, wires=wires[0]),
            qml.PhaseShift(retardance, wires=wires[0]),
            qml.RY(2 * theta, wires=wires[0]),
        ]

    @staticmethod
    def compute_matrix(theta, retardance=None):
        interface = qml.math.get_interface(theta)

        c = qml.math.cos(theta)
        s = qml.math.sin(theta)
        e = qml.math.exp(1.0j * retardance)

        # If anything is not tensorflow, it has to be casted and then
        if interface == "tensorflow":
            c = qml.math.cast_like(qml.math.asarray(c, like=interface), 1j)
            s = qml.math.cast_like(qml.math.asarray(s, like=interface), 1j)

        mat = np.exp(-1.0j * retardance / 2) * qnp.array(
            [
                [c**2 + e * s**2, (1 - e) * c * s],
                [(1 - e) * c * s, e * c**2 + s**2],
            ]
        )

        return qml.math.stack([qml.math.stack(row, axis=-1) for row in mat], axis=-2)