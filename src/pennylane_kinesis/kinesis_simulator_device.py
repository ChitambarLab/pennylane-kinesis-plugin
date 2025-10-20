from pennylane import QubitDevice
import numpy as np

from .operators import (
    VariableWPRotStage,
    WaveplateTypes,
    QWPRotStage,
    HWPRotStage,
    QWP_RETARDANCE_A,
    QWP_RETARDANCE_B,
    HWP_RETARDANCE_A,
    HWP_RETARDANCE_B,
)
import pennylane as qml
from .version import __version__
from enum import Enum

class DeviceIndex(Enum):
    aHWP = 0
    aQWP = 1
    bHWP = 2
    bQWP = 3

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
        "VariableWPRotStage",
        "QubitChannel",
    }
    observables = {"Projector", "Identity", "PauliZ"}

    init_state = np.array(
        [
            [1, 0],
            [0, 0],
        ],
        dtype=complex,
    )

    def __init__(
        self,
        wires,
        shots=10000,
        background_counts=[300, 100],
        *,
        r_dtype=np.float64,
        c_dtype=np.complex128,
        shot_rate=[300000, 300000],
        analytic=None,
        seed=None,
        qwp_domain_period=np.pi,
        hwp_domain_period=np.pi / 2,
        debug=False,
        h_efficiency=1,
        v_efficiency=1,
        waveplate_order = [DeviceIndex.aHWP, DeviceIndex.aQWP, DeviceIndex.bHWP, DeviceIndex.bQWP]
    ):
        super().__init__(wires, shots, r_dtype=r_dtype, c_dtype=c_dtype, analytic=analytic)
        self._state = self.init_state
        self.background_counts = background_counts
        if seed:
            np.random.seed(seed)

        self.qwp_domain_period = qwp_domain_period
        self.hwp_domain_period = hwp_domain_period
        self.shot_rate = np.array(shot_rate)
        self.h_efficiency = h_efficiency
        self.v_efficiency = v_efficiency

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
        for i, operation in enumerate(all_ops):
            device_index = self.waveplate_order[i]
            if operation.name == "HWPRotStage" or (
                operation.name == "VariableWPRotStage" and (device_index is DeviceIndex.aHWP or device_index is DeviceIndex.bHWP)
            ):
                angle = (
                    np.mod(operation.parameters[0], self.hwp_domain_period)
                    if self.hwp_domain_period
                    else operation.parameters[0]
                )
                # mat = qml.matrix(operation)
                if i == 0:
                    mat = VariableWPRotStage(
                        angle,
                        wires=[0],
                    ).compute_matrix(angle, HWP_RETARDANCE_A)
                else:
                    mat = VariableWPRotStage(
                        angle,
                        wires=[0],
                    ).compute_matrix(angle, HWP_RETARDANCE_B)
                self._state = mat @ self._state @ mat.conj().T
            elif operation.name == "QWPRotStage" or (
                operation.name == "VariableWPRotStage" and (device_index is DeviceIndex.aQWP or device_index is DeviceIndex.bQWP)
            ):
                angle = (
                    np.mod(operation.parameters[0], self.qwp_domain_period)
                    if self.qwp_domain_period
                    else operation.parameters[0]
                )
                # mat = qml.matrix(operation)
                if i == 1:
                    mat = VariableWPRotStage(
                        angle,
                        wires=[0],
                    ).compute_matrix(angle, retardance=QWP_RETARDANCE_A)
                else:
                    mat = VariableWPRotStage(
                        angle,
                        wires=[0],
                    ).compute_matrix(angle, retardance=QWP_RETARDANCE_B)
                self._state = mat @ self._state @ mat.conj().T
            elif operation.name == "QubitChannel":
                kraus_ops = operation.kraus_matrices()
                output_state = np.zeros(self._state.shape, dtype=complex)
                for op in kraus_ops:
                    output_state += op @ self._state @ op.conj().T
                self._state = output_state
            else:
                mat = qml.matrix(operation)
                self._state = mat @ self._state @ mat.conj().T

    def take_background(self):
        zero_background_counts = np.random.poisson(int((self.background_counts[0])))
        one_background_counts = np.random.poisson(int((self.background_counts[1])))

        return (zero_background_counts, one_background_counts)

    def analytic_probability(self, wires=None):
        if self._state is None:
            return None

        real = self._real(self._state)
        imag = self._imag(self._state)
        prob = self.marginal_prob(real**2 + imag**2, wires)
        return prob

    def generate_samples(self):
        zero_state = np.array([[1, 0], [0, 0]])
        one_state = np.array([[0, 0], [0, 1]])
        p0 = np.trace(self._state @ zero_state).real
        p1 = np.trace(self._state @ one_state).real

        # Poisson(background counts /s) * background_ratio
        zero_background_counts = int(
            (np.random.poisson(self.background_counts[0]) * self.background_counts[0] / self.shot_rate[0])
        )
        one_background_counts = int(
            (np.random.poisson(self.background_counts[1]) * self.background_counts[1] / self.shot_rate[1])
        )

        # Poisson(rate at input of PBS * prob of measurement) * measurement time + background
        zero_counts = (
            int(
                (
                    np.random.poisson(p0 * np.sum(self.shot_rate))
                    * self.h_efficiency
                    * self.shots
                    / np.sum(self.shot_rate)
                )
            )
            + zero_background_counts
        )
        one_counts = (
            int(
                (
                    np.random.poisson(p1 * np.sum(self.shot_rate))
                    * self.v_efficiency
                    * self.shots
                    / np.sum(self.shot_rate)
                )
            )
            + one_background_counts
        )

        return np.concatenate(
            (
                np.zeros((zero_counts, 1), dtype=int),
                np.ones((one_counts, 1), dtype=int),
            )
        )
