import pennylane as qml
from pennylane.operation import Operation


class HWPRotStage(Operation):
    """A quarter wave plate (HWP) mounted on a rotational stage.

    The HWP decomposes as the following quantum function:

    .. code-block:: python

        def hwp_rot_stage(\theta, wires):
            qml.RY(-\theta, wires=wires[0])
            qml.PauliZ(wires=wires[0])
            qml.RY(\theta, wires=wires[0])

    The HWP rotational stage corresponds to the following unitary matrix:

    .. math::

        HWP(\theta) = \begin{pmatrix}
            \cos^2(\theta/2) - \sin^2(\theta/2) & 2\cos(\theta/2)\sin(\theta/2) \\
            2\cos(\theta/2)\sin(\theta/2) & - \cos^2(\theta/2) + \sin^2(\theta/2) \\
        \end{pmatrix}

    :param theta: The angle of rotation. 
    :type theta: Float

    :param wires: The wire to which the operator is applied
    :type wires: list-like

    :param do_queue: Specifies whether or not the operator is queued in a tape context. Defaults to ``True``.
    :type do_queue: Boolean

    :param id: An optional custom name for the operator. Defaults to ``None``.
    :type: string-like

    :returns: An instantiated and trainable ``HWPRotStage`` operator.
    """

    num_wires = 1
    grad_method = "A"
    parameter_frequencies = [(1, 2)]

    def __init__(self, angle, wires, do_queue=True, id=None):
        super().__init__(angle, wires=qml.wires.Wires(wires), do_queue=do_queue, id=id)

    @property
    def num_params(self):
        return 1

    @property
    def ndim_params(self):
        return (0,)

    @staticmethod
    def compute_decomposition(theta, wires):
        return [
            qml.RY(-1 * theta, wires=wires),
            qml.PauliZ(wires=wires),
            qml.RY(theta, wires=wires),
        ]

    @staticmethod
    def compute_matrix(theta):
        interface = qml.math.get_interface(theta)

        c = qml.math.cos(theta / 2)
        s = qml.math.sin(theta / 2)

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

        def qwp_rot_stage(\theta, wires):
            qml.RY(-\theta, wires=wires[0])
            qml.S(wires=wires[0])
            qml.RY(\theta, wires=wires[0])

    The QWP rotational stage corresponds to the following unitary matrix:

    .. math::

        QWP(\theta) = \begin{pmatrix}
            \cos^2(\theta/2) + i \sin^2(\theta/2) & (1-i)\cos(\theta/2)\sin(\theta/2) \\
            (1-i)\cos(\theta/2)\sin(\theta/2) & i \cos^2(\theta/2) + \sin^2(\theta/2) \\
        \end{pmatrix}

    :param theta: The angle of rotation. 
    :type theta: Float

    :param wires: The wire to which the operator is applied
    :type wires: list-like

    :param do_queue: Specifies whether or not the operator is queued in a tape context. Defaults to ``True``.
    :type do_queue: Boolean

    :param id: An optional custom name for the operator. Defaults to ``None``.
    :type: string-like

    :returns: An instantiated and trainable ``QWPRotStage`` operator.
    """

    num_wires = 1
    grad_method = "A"
    parameter_frequencies = [(1, 2)]

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
            qml.RY(-1 * theta, wires=wires[0]),
            qml.S(wires=wires[0]),
            qml.RY(theta, wires=wires[0]),
        ]

    @staticmethod
    def compute_matrix(theta):
        interface = qml.math.get_interface(theta)

        c = qml.math.cos(theta / 2)
        s = qml.math.sin(theta / 2)

        # If anything is not tensorflow, it has to be casted and then
        if interface == "tensorflow":
            c = qml.math.cast_like(qml.math.asarray(c, like=interface), 1j)
            s = qml.math.cast_like(qml.math.asarray(s, like=interface), 1j)

        mat = [
            [c**2 + 1j * s**2, (1 - 1j) * c * s],
            [(1 - 1j) * c * s, 1j * c**2 + s**2],
        ]

        return qml.math.stack([qml.math.stack(row, axis=-1) for row in mat], axis=-2)
