from pennylane import QubitDevice
import pennylane as qml
import numpy as np

from .operators import HWPRotStage, QWPRotStage
from .version import __version__


class KinesisSimulatorDevice(QubitDevice):
    """A PennyLane plugin device for simulating the integration with the ThorLabs Kinesis software.

    This plugin device performs a simple state vector simulation of optical qubits passing
    through components on rotational stages controlled by the Kinesis software.
    """

    name = "Kinesis Simulator Device"
    short_name = "kinesis.simulator"
    pennylane_requires = "0.34"
    version = __version__
    author = "Brian Doolittle, Michael Vayninger, Timur Javid, and Colin Lualdi"

    operations = {
        "PauliZ",
        "PhaseShift",
        "RX",
        "RY",
        "RZ",
        "S",
        "HWPRotStage",
        "QWPRotStage",
        "QubitChannel"
    }
    observables = {"Projector", "Identity", "PauliZ"}

    incidence_rate = 100000
    init_state = np.array([1, 0])

    def __init__(
        self,
        wires,
        shots=1000,
        background_counts=100,
        *,
        r_dtype=np.float64,
        c_dtype=np.complex128,
        shot_rate=[400000,400000],
        analytic=None,
        seed=None,
        qwp_domain_period=np.pi,
        hwp_domain_period=np.pi/2,
    ):
        super().__init__(wires, shots, r_dtype=r_dtype, c_dtype=c_dtype, analytic=analytic)
        self._state = self.init_state
        self.background_counts = background_counts
        if seed:
            np.random.seed(seed)

        self.qwp_domain_period = qwp_domain_period
        self.hwp_domain_period = hwp_domain_period
        self.shot_rate = np.array(shot_rate)
        self.background_ratio = background_counts/np.sum(self.shot_rate)

    def reset(self):
        super().reset()
        self._state = self.init_state

    @classmethod
    def capabilities(self):
        capabilities = super().capabilities().copy()
        capabilities.update(
            model="qubit",
            supports_inverse_operations=False,
            supports_analytic_computation=True,
            supports_broadcasting=False,
            returns_state=False,
        )
        return capabilities

    def apply(self, operations, rotations=[]):
        all_ops = operations + rotations if rotations else operations
        for operation in all_ops:
            if operation.name == "HWPRotStage":
                angle = (
                    np.mod(operation.parameters[0], self.hwp_domain_period)
                    if self.hwp_domain_period
                    else operation.parameters[0]
                )
                mat = qml.matrix(HWPRotStage)(angle, wires=[0])
                self._state = mat @ self._state
            elif operation.name == "QWPRotStage":
                angle = (
                    np.mod(operation.parameters[0], self.qwp_domain_period)
                    if self.qwp_domain_period
                    else operation.parameters[0]
                )

                mat = qml.matrix(QWPRotStage)(angle, wires=[0])
                self._state = mat @ self._state
            elif operation.name == "QubitChannel":
                kraus_ops = operation.kraus_matrices()
                output_state = np.zeros(self._state.shape,dtype=complex)
                for op in kraus_ops:
                    output_state += op @ self._state @ op.conj().T
                self._state = output_state
            else:
                mat = qml.matrix(operation)
                self._state = mat @ self._state



    def take_background(self):
        zero_background_counts = np.random.poisson(int((self.background_counts)*0.666))
        one_background_counts = np.random.poisson(int((self.background_counts)*0.333))

        return (zero_background_counts, one_background_counts)

    def analytic_probability(self, wires=None):
        if self._state is None:
            return None

        real = self._real(self._state)
        imag = self._imag(self._state)
        prob = self.marginal_prob(real**2 + imag**2, wires)
        return prob

    def generate_samples(self):
        p0 = np.linalg.norm(self._state[0]) ** 2

        counts = np.random.poisson(self.shots) + self.background_counts
        zero_background_counts = np.random.poisson(int((self.background_ratio * self.shots)*0.666))
        one_background_counts = np.random.poisson(int((self.background_ratio*self.shots)*0.333))
        detected_counts = self.shots - zero_background_counts - one_background_counts

        zero_counts = min(int(p0 * detected_counts) + zero_background_counts, self.shots)
        one_counts = self.shots - zero_counts

        return np.concatenate(
            (
                np.ones((one_counts, 1), dtype=int),
                np.zeros((zero_counts, 1), dtype=int),
            )
        )
