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
        shot_rate=[300000, 300000],
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
        # wp_angles = np.linspace(0,2*np.pi, 3)
        # wedge_interp = np.linspace(0,2*np.pi, 1800)
        # wedge_vals = np.ones(3)
        # wedge_vals[1] = 0.2

        # print("Initializing")

        # self.aqwp_wedge = np.interp(wedge_interp, wp_angles, wedge_vals)

        # print(self.aqwp_wedge)
        self.shots = shots

        self.background_counts = background_counts
        self.bqwp_angle = 0
        if seed:
            np.random.seed(seed)

        self.qwp_domain_period = qwp_domain_period
        self.hwp_domain_period = hwp_domain_period
        self.shot_rate = np.array(shot_rate)
        self.noise_percentage = 8e-4
        self.h_efficiency = 1 - 1e-4
        self.v_efficiency = 1
        self.integration_time = self.shots / np.sum(self.shot_rate)
        self.crosstalk = np.array([[1, 1e-2], [1e-1, 1]])

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
        for i, operation in enumerate(all_ops):
            # print(operation)
            device_index = self.waveplate_order[i]
            # print("i", i)
            # print(self.waveplate_order[i])
            # print(operation.parameters[0])
            # print(operation.name)
            # print(device_index.value == DeviceIndex.aHWP.value)
            if operation.name == "HWPRotStage" or (
                operation.name == "VariableWPRotStage"
                and (device_index.value == DeviceIndex.aHWP.value or device_index.value == DeviceIndex.bHWP.value)
            ):
                angle = operation.parameters[0]
                mat = VariableWPRotStage(
                    angle % self.hwp_domain_period,
                    wires=[0],
                ).compute_matrix(angle, retardance=self.retardances[device_index.value])
                self._state = mat @ self._state @ mat.conj().T
            elif operation.name == "QWPRotStage" or (
                operation.name == "VariableWPRotStage"
                and (device_index.value == DeviceIndex.aQWP.value or device_index.value == DeviceIndex.bQWP.value)
            ):
                angle = operation.parameters[0]
                # mat = qml.matrix(operation)
                mat = VariableWPRotStage(
                    angle % self.qwp_domain_period,
                    wires=[0],
                ).compute_matrix(angle, retardance=self.retardances[device_index.value])
                self._state = mat @ self._state @ mat.conj().T
            elif operation.name == "QubitChannel":
                kraus_ops = operation.kraus_matrices()
                output_state = np.zeros(self._state.shape, dtype=complex)
                for op in kraus_ops:
                    output_state += op @ self._state @ op.conj().T
                self._state = output_state
            else:
                print("else")
                mat = qml.matrix(operation)
                self._state = mat @ self._state @ mat.conj().T

            # print(angle)
            angles.append(angle * 180 / np.pi)
        self.angles = angles
        # print(self.angles)

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
        p0 = np.clip(np.trace(self._state @ zero_state).real, a_min=1e-16, a_max=None)
        p1 = np.clip(np.trace(self._state @ one_state).real, a_min=1e-16, a_max=None)
        # p0 = np.trace(self._state @ zero_state).real
        # p1 = np.trace(self._state @ one_state).real

        # print(p0)
        # print(p1)
        # print(self.angles)
        # print(self.angles)
        # print(int(self.angles[self.waveplate_order[1].value]))
        # p0 *= self.aqwp_wedge[int(self.angles[self.waveplate_order[0].value])]
        # p1 *= self.aqwp_wedge[int(self.angles[self.waveplate_order[1].value])]
        # print(self.aqwp_wedge[int(self.angles[self.waveplate_order[1].value])])
        # p1 *= self.aqwp_wedge[int(self.angles[self.waveplate_order[1].value])]

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

        # print(zero_counts,one_countsg)

        return np.concatenate(
            (
                np.zeros((zero_counts, 1), dtype=int),
                np.ones((one_counts, 1), dtype=int),
            )
        )
