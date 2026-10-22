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
        background_counts=[400, 100],
        *,
        r_dtype=np.float64,
        c_dtype=np.complex128,
        shot_rate=[400000, 400000],
        analytic=None,
        seed=None,
        qwp_domain_period=np.pi,
        hwp_domain_period=np.pi / 2,
        debug=False,
        h_efficiency=1,
        v_efficiency=1,
        waveplate_order=[DeviceIndex.aHWP, DeviceIndex.aQWP, DeviceIndex.bQWP, DeviceIndex.bHWP],
        retardances=[HWP_RETARDANCE_A, QWP_RETARDANCE_A, HWP_RETARDANCE_B, QWP_RETARDANCE_B],
    ):
        super().__init__(wires, shots, r_dtype=r_dtype, c_dtype=c_dtype, analytic=analytic)
        self._state = self.init_state
        self.retardances = retardances
        self.waveplate_order = waveplate_order

        self.shots = shots

        self.background_counts = background_counts
        self.bqwp_angle = 0
        if seed:
            np.random.seed(seed)

        self.qwp_domain_period = qwp_domain_period
        self.hwp_domain_period = hwp_domain_period
        self.shot_rate = np.array(shot_rate)
        self.noise_percentage = 0
        self.h_efficiency = 1
        self.v_efficiency = 1
        # self.shots is None in analytic mode but photon-count sampling still needs a concrete
        # integration time, so fall back to the class's default shots in that case.
        integration_shots = self.shots if self.shots is not None else 10000
        self.integration_time = integration_shots / np.sum(self.shot_rate)
        self.crosstalk = np.eye(2)

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
        angles = []
        # need to keep track of waveplate index separately for applying
        # imperfect-retardance waveplates
        wp_index = 0
        for operation in all_ops:
            if operation.name in ("HWPRotStage", "QWPRotStage", "VariableWPRotStage"):
                device_index = self.waveplate_order[wp_index]
                wp_index += 1

            if operation.name == "HWPRotStage":
                angle = operation.parameters[0]
                if self.hwp_domain_period:
                    angle %= self.hwp_domain_period
                mat = HWPRotStage(angle, wires=[0]).compute_matrix(angle)
                self._state = mat @ self._state @ mat.conj().T
                angles.append(angle * 180 / np.pi)

            elif operation.name == "QWPRotStage":
                angle = operation.parameters[0]
                if self.qwp_domain_period:
                    angle %= self.qwp_domain_period
                mat = QWPRotStage(angle, wires=[0]).compute_matrix(angle)
                self._state = mat @ self._state @ mat.conj().T
                angles.append(angle * 180 / np.pi)
            elif operation.name == "VariableWPRotStage":
                angle = operation.parameters[0]
                if device_index.value == DeviceIndex.aQWP.value or device_index.value == DeviceIndex.bQWP.value:
                    domain_period = self.qwp_domain_period
                else:
                    domain_period = self.hwp_domain_period
                if self.hwp_domain_period:
                    angle %= domain_period
                mat = VariableWPRotStage(
                    angle,
                    wires=[0],
                ).compute_matrix(angle, retardance=self.retardances[device_index.value])
                self._state = mat @ self._state @ mat.conj().T
                angles.append(angle * 180 / np.pi)
            elif operation.name == "QubitChannel":
                kraus_ops = operation.kraus_matrices()
                output_state = np.zeros(self._state.shape, dtype=complex)
                for op in kraus_ops:
                    output_state += op @ self._state @ op.conj().T
                self._state = output_state
            else:
                mat = qml.matrix(operation)
                self._state = mat @ self._state @ mat.conj().T

        self.angles = angles

    def take_background(self):
        zero_background_counts = np.random.poisson(int((self.background_counts[0])))
        one_background_counts = np.random.poisson(int((self.background_counts[1])))

        return (zero_background_counts, one_background_counts)

    def analytic_probability(self, wires=None):
        if self._state is None:
            return None

        prob = self.marginal_prob(self._real(self._diag(self._state)), wires)
        return prob

    def generate_samples(self):
        zero_state = np.array([[1, 0], [0, 0]])
        one_state = np.array([[0, 0], [0, 1]])
        p0 = np.clip(np.trace(self._state @ zero_state).real, a_min=1e-16, a_max=None)
        p1 = np.clip(np.trace(self._state @ one_state).real, a_min=1e-16, a_max=None)

        # Poisson(background counts /s) * background_ratio
        zero_background_counts = int((np.random.poisson(self.background_counts[0]) * self.integration_time))
        one_background_counts = int((np.random.poisson(self.background_counts[1]) * self.integration_time))

        zero_incident_rate = p0 * np.sum(self.shot_rate)
        one_incident_rate = p1 * np.sum(self.shot_rate)

        crosstalk_zero_rate = self.crosstalk[0, 1] * one_incident_rate
        crosstalk_one_rate = self.crosstalk[1, 0] * zero_incident_rate

        crosstalk_zero_counts = int(np.random.poisson(crosstalk_zero_rate) * self.integration_time)
        crosstalk_one_counts = int(np.random.poisson(crosstalk_one_rate) * self.integration_time)

        noise_counts_zero = int(np.random.poisson(zero_incident_rate * self.noise_percentage) * self.integration_time)
        noise_counts_one = int(np.random.poisson(one_incident_rate * self.noise_percentage) * self.integration_time)

        # The counts are calculated as:
        # Poisson(rate at input of PBS * prob of measurement) * measurement time (= num_shots/shot_rate) + background + noise_counts
        zero_counts = (
            int((np.random.poisson(zero_incident_rate) * self.h_efficiency * self.integration_time))
            + zero_background_counts
            + noise_counts_zero
            + crosstalk_zero_counts
        )
        one_counts = (
            int((np.random.poisson(one_incident_rate) * self.v_efficiency * self.integration_time))
            + one_background_counts
            + noise_counts_one
            + crosstalk_one_counts
        )

        return np.concatenate(
            (
                np.zeros((zero_counts, 1), dtype=int),
                np.ones((one_counts, 1), dtype=int),
            )
        )
