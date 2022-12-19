from pennylane import QubitDevice
import pennylane as qml
import numpy as np

from .version import __version__


class KinesisSimulatorDevice(QubitDevice):
    """ """

    name = "Kinesis Simulator Device"
    short_name = "kinesis.simulator"
    pennylane_requires = "0.27.0"
    version = __version__
    author = "Brian Doolittle"

    operations = {"PauliZ", "PhaseShift", "S", "RY", "RZ"}
    observables = {"Projector", "Identity", "PauliZ"}

    incidence_rate = 100000
    init_state = np.array([1, 0])

    def __init__(
        self,
        wires,
        shots=1000,
        *,
        r_dtype=np.float64,
        c_dtype=np.complex128,
        analytic=None,
    ):
        super().__init__(wires, shots, r_dtype=r_dtype, c_dtype=c_dtype, analytic=analytic)
        self._state = self.init_state

    def reset(self):
        super().reset()
        self._state = self.init_state

    @classmethod
    def capabilities(self):
        capabilities = super().capabilities().copy()
        capabilities.update(
            model="qubit",
            supports_inverse_operations=False,
            supports_analytic_computation=False,
            supports_broadcasting=False,
            returns_state=False,
        )
        return capabilities

    def apply(self, operations, rotations=[]):
        all_ops = operations + rotations if rotations else operations
        for operation in all_ops:
            mat = qml.matrix(operation)
            self._state = mat @ self._state

    def generate_samples(self):
        p0 = np.linalg.norm(self._state[0]) ** 2

        return np.random.choice([0, 1], size=(self.shots, 1), p=[p0, 1 - p0])
