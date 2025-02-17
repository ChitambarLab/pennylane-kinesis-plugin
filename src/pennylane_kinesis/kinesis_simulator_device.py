from pennylane import QubitDevice
import pennylane as qml
import numpy as np

from .operators import HWPRotStage, QWPRotStage
from .version import __version__


fiber_ops = np.array([[[(0.0018772661051027382+0j), (-0.0027611995983203583+0.0010188753312217946j)], [(-0.002213517501836983+0.0019397585566256889j), (-0.0008962080534798236+0.0016495310170866926j)]], [[(-0.026409683178653796+0j), (-0.01630798158908431+0.003369984295831257j)], [(-0.01405007500683984+0.008938773263111127j), (0.01895201582815165-0.018392732433955786j)]], [[(0.08992855920493449+0j), (-0.06442924408549917+0.009838287344991347j)], [(0.049568586506210065-0.042318931691822916j), (0.05879575383289624-0.06804549557115137j)]], [[(-0.5789218081389755+0j), (-0.7724691369046199+0.23413445167935226j)], [(0.6835346549211395-0.429310923437477j), (-0.3798535015712914+0.4368773212738786j)]]])

fiber_ops2 = [[[(0.06207743763293914+0j), (-0.07964913383185647+0.015020717595874145j)], [(0.03459295492416963-0.032704185829270135j), (0.07511841451835383-0.08758499614306452j)]], [[(-0.5771281829228686+0j), (-0.7755710707383385+0.2344980875222249j)], [(0.6814851257996933-0.4273377771604468j), (-0.38126653311398817+0.43819646406643614j)]]]
fiber_ops_projected = np.array([[[(0.0018691098465687557+0j), (-0.0027522087074046534+0.0010141999562753743j)], [(-0.002208203873602957+0.0019305666959577885j), (-0.0008950003417570723+0.0016408938840532343j)]], [[(-0.026681795254364118+0j), (-0.016471112465215307+0.003335714418126235j)], [(-0.014178020792834782+0.009022661584289006j), (0.0192192145579343-0.018507851986120483j)]], [[(0.08872337087542657+0j), (-0.06360407100674283+0.009487824495710595j)], [(0.04896992088782897-0.04168265464129159j), (0.05833820096728807-0.06684681610585777j)]], [[(-0.5789862962078215+0j), (-0.7726431442384725+0.2340590967780792j)], [(0.6837394547292913-0.42925686219726406j), (-0.38004584177559386+0.43679547811917463j)]]])
fiber_ops_proj_1_20 = np.array([[[(0.001869300889533977+0j), (-0.002753327831004509+0.001015789419268893j)], [(-0.002207749235504094+0.0019335163825407577j), (-0.0008930415825510767+0.001642182252073776j)]], [[(-0.026610404781142634+0j), (-0.01642337886396826+0.0032854912732571046j)], [(-0.014127976451108805+0.008995671598809377j), (0.01920675029653697-0.018417773635206178j)]], [[(0.08900546336839853+0j), (-0.06384570698650913+0.009359106633608522j)], [(0.04911225920714969-0.041855145146700905j), (0.058652284798456-0.06694685948990188j)]], [[(-0.5789709236002231+0j), (-0.7725981748262815+0.2340900420702011j)], [(0.6836845899658486-0.4292801753899103j), (-0.3799862100711545+0.43682698008736776j)]]])

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
    init_state = np.array([[1,0],[0,0]])

    def __init__(
        self,
        wires,
        shots=10000,
        background_counts=100,
        *,
        r_dtype=np.float64,
        c_dtype=np.complex128,
        shot_rate=[400000,400000],
        analytic=None,
        seed=None,
        qwp_domain_period=np.pi,
        hwp_domain_period=np.pi/2,
        debug=False,
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
                self._state = mat @ self._state @ mat.conj().T

            elif operation.name == "QWPRotStage":
                angle = (
                    np.mod(operation.parameters[0], self.qwp_domain_period)
                    if self.qwp_domain_period
                    else operation.parameters[0]
                )

                mat = qml.matrix(QWPRotStage)(angle, wires=[0])
                self._state = mat @ self._state @ mat.conj().T

            elif operation.name == "QubitChannel":
                # print("QubitChannel")
                kraus_ops = operation.kraus_matrices()
                output_state = np.zeros(self._state.shape,dtype=complex)
                for op in kraus_ops:
                    output_state += op @ self._state @ op.conj().T
                self._state = output_state
                
            else:
                mat = qml.matrix(operation)
                self._state = mat @ self._state @ mat.conj().T

    def take_background(self):
        zero_background_counts = np.random.poisson(int((self.background_counts)*0.333))
        one_background_counts = np.random.poisson(int((self.background_counts)*0.666))

        return (zero_background_counts, one_background_counts)

    def analytic_probability(self, wires=None):
        if self._state is None:
            return None

        real = self._real(self._state)
        imag = self._imag(self._state)
        prob = self.marginal_prob(real**2 + imag**2, wires)
        return prob

    def generate_samples(self):
        zero_state = np.array([[1,0],[0,0]])
        
        # p0 = np.linalg.norm(self._state[0]) ** 2

        p0 = np.trace(zero_state@self._state).real
        counts = np.random.poisson(self.shots) + self.background_counts
        zero_background_counts = np.random.poisson(int((self.background_ratio * self.shots)*0.333))
        one_background_counts = np.random.poisson(int((self.background_ratio*self.shots)*0.666))
        detected_counts = self.shots - zero_background_counts - one_background_counts

        zero_counts = min(int(p0 * detected_counts) + zero_background_counts, self.shots)
        one_counts = self.shots - zero_counts

        return np.concatenate(
            (
                np.ones((one_counts, 1), dtype=int),
                np.zeros((zero_counts, 1), dtype=int),
            )
        )
