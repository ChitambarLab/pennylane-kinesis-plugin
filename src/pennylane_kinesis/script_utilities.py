import numpy as np
import qnetvo

import pennylane as qml
from .file_utilities import datetime_now_string
from .file_utilities import make_dir
from .kinesis_cage_rotator_device import KinesisMotors
from .operators import HWPRotStage
from .operators import QWPRotStage
from .optimization import optimize
import pennylane_kinesis as pk
from .elliptec_rotator_device import ElliptecMotors

import time

HWP_PERIOD = np.pi / 2
QWP_PERIOD = np.pi
fiber_ops_projected = np.array([[[(0.0018691098465687557+0j), (-0.0027522087074046534+0.0010141999562753743j)], [(-0.002208203873602957+0.0019305666959577885j), (-0.0008950003417570723+0.0016408938840532343j)]], [[(-0.026681795254364118+0j), (-0.016471112465215307+0.003335714418126235j)], [(-0.014178020792834782+0.009022661584289006j), (0.0192192145579343-0.018507851986120483j)]], [[(0.08872337087542657+0j), (-0.06360407100674283+0.009487824495710595j)], [(0.04896992088782897-0.04168265464129159j), (0.05833820096728807-0.06684681610585777j)]], [[(-0.5789862962078215+0j), (-0.7726431442384725+0.2340590967780792j)], [(0.6837394547292913-0.42925686219726406j), (-0.38004584177559386+0.43679547811917463j)]]])
fiber_ops_proj_1_20 = np.array([[[(0.001869300889533977+0j), (-0.002753327831004509+0.001015789419268893j)], [(-0.002207749235504094+0.0019335163825407577j), (-0.0008930415825510767+0.001642182252073776j)]], [[(-0.026610404781142634+0j), (-0.01642337886396826+0.0032854912732571046j)], [(-0.014127976451108805+0.008995671598809377j), (0.01920675029653697-0.018417773635206178j)]], [[(0.08900546336839853+0j), (-0.06384570698650913+0.009359106633608522j)], [(0.04911225920714969-0.041855145146700905j), (0.058652284798456-0.06694685948990188j)]], [[(-0.5789709236002231+0j), (-0.7725981748262815+0.2340900420702011j)], [(0.6836845899658486-0.4292801753899103j), (-0.3799862100711545+0.43682698008736776j)]]])
fiber_ops_proj_1_25 = np.array([[[(-1.8189161262355992e-05+0j), (1.1215280345683544e-05-1.4049221285203244e-05j)], [(6.134511718376058e-06-1.6508742104145237e-05j), (1.0701722052490562e-05-1.3596662144797063e-05j)]], [[(-0.0193775271996777+0j), (-0.006052928362413193+0.009189484725334422j)], [(-0.009092517099851483+0.01539699772352265j), (0.007231121421921657-0.005671038280377137j)]], [[(0.061828311130135705+0j), (-0.08045755210622296+0.02635880953784693j)], [(0.03903531780079182-0.013549659164153141j), (0.0815521753610482-0.07806185845636447j)]], [[(-0.5810499259287734+0j), (-0.7702076113785203+0.23303405963620938j)], [(0.6866599052701762-0.42971366991118737j), (-0.37849325690161+0.4348563743755882j)]]])

def two_state_discrimination(
    filepath,
    shots=10000,
    shot_rate=3000000,
    background_counts=300,
    two_wp=False,
    numerics=False,
    step_size=0.08,
    num_steps=20,
    optimizer="gradient-descent",
    verbose=True,
    fixed_meas_settings=[0, 0],
    hwp_period=HWP_PERIOD,
    qwp_period=QWP_PERIOD,
    test_mode=False,
    diff_method="parameter-shift",
):
    """
    Performs a hardware-compatible optimization of minimal error discrimination of two encoded states.
    By default, this function connects to the hardware backend and both the state preparations and measurements are optimized.

    :param filepath: The directory to save data to. Subdirectories will be created within this directory.
    :type filepath: String

    :param shots: The number of shots to use in calculating gradients.
    :type: Int

    :param shot_rate: The rate at which shots are emitted from the source. Used for hardware implementation.
    :type: Int

    :param two_wp: If True, the measurement waveplates will be fixed.
    :type twp_wp: Bool

    :param numerics: If True, a simulator backend is used in place of hardware.
    :type numerics: Bool

    :param step_size: The gradient descent step size.
    :type step_size: Float

    :param num_steps: The number of gradient descent steps.
    :type num_steps: Int

    :param verbose: If True, the optimizer runs in verbose mode.
    :type verbose: Bool

    :param fixed_settings: A two element array.
    :type fixed_settings: List[Float]

    :param use_adam: If `True`, use an adaptive step size in gradient descent.
    :type use_adam: Bool

    :param hwp_period: The domain of unique half wave-plate operations.
    :type hwp_period: Float

    :param qwp_period: The domain of unique quarter wave-plate operations.
    :type qwp_period: Float

    :param diff_method: The differentiation method to use in the QNode ('parameter-shift' or 'finite-diff').
    :type diff_method: String

    :returns: An optimization dictionary output from the `optimize` method.
    :rtype: Dict
    """
    data_path = make_dir(filepath)

    def arb_unitary(settings, wires):
        HWPRotStage(settings[0], wires=[wires[0]])
        QWPRotStage(settings[1], wires=[wires[0]])

    # Function to use for MeasureNode if we are not moving the measurement waveplates.
    def meas_fn(settings, wires):
        HWPRotStage(settings[0], wires=wires[0:1])
        QWPRotStage(settings[1], wires=wires[0:1])

    def noise_fn(settings, wires):
        kraus_ops = fiber_ops_proj_1_25
        qml.QubitChannel(kraus_ops, wires=wires[0])


    if numerics:
        data_path += "numerics/"
        dev_name = "kinesis.simulator"
        additional_dev_kwargs = {
            "shot_rate": shot_rate,
            "shots": shots,
            "background_counts": background_counts,
            "qwp_domain_period": qwp_period,
            "hwp_domain_period": hwp_period,
        }
    else:
        motors = ElliptecMotors(test_mode=test_mode)
        data_path += "hardware/"
        dev_name = "elliptec.rotator"
        additional_dev_kwargs = {
            "motors": motors,
            "test_mode": test_mode,
            "filepath": filepath,
        }

    # Legacy code: old thorlabs rotator mounts:
    # motors = KinesisMotors(test_mode=test_mode)
    # data_path += "hardware/"
    # dev_name = "kinesis.cage.rotator"
    # additional_dev_kwargs = {
    #     "motors": motors,
    #     "test_mode": test_mode,
    #     "filepath": filepath,
    # }

    make_dir(data_path)

    meas_node = None
    if two_wp:
        data_path += "two_wp/"
        meas_node = qnetvo.MeasureNode(
            num_in=1,
            num_out=2,
            wires=[0],
            ansatz_fn=lambda *args: meas_fn(fixed_meas_settings, args[1]),
            num_settings=0,
        )

    else:
        data_path += "four_wp/"
        meas_node = qnetvo.MeasureNode(num_in=1, wires=[0], ansatz_fn=arb_unitary, num_settings=2)

    make_dir(data_path)

    datetime_str = datetime_now_string()

    # network ansatz class manages circuit construction and settings
    # state_disc_ansatz = qnetvo.NetworkAnsatz(
    #     [qnetvo.PrepareNode(num_in=2, wires=[0], ansatz_fn=arb_unitary, num_settings=2)],
    #     [meas_node],
    #     dev_kwargs={
    #         "name": dev_name,
    #         "shot_rate": shot_rate,
    #         "shots": shots,
    #         "debug": verbose,
    #         **additional_dev_kwargs,
    #     },
    # )

    # network ansatz class manages circuit construction and settings
    state_disc_ansatz = qnetvo.NetworkAnsatz(
        [qnetvo.PrepareNode(num_in=2, wires=[0], ansatz_fn=arb_unitary, num_settings=2)],
        [qnetvo.NoiseNode(wires=[0], ansatz_fn=noise_fn)],
        [meas_node],
        dev_kwargs={
            "name": dev_name,
            "shot_rate": shot_rate,
            "shots": shots,
            **additional_dev_kwargs,
        },
    )

    device = qml.device(**state_disc_ansatz.dev_kwargs)
    shot_rate = device.shot_rate.tolist()
    background_rate = device.take_background()

    # construct a linear inequality cost function that
    # corresponds to the minimal qubit dimensionality witness.
    qnode_kwargs = {"diff_method": diff_method}
    if diff_method == "finite-diff":
        qnode_kwargs["h"] = 1e-3
        qnode_kwargs["strategy"] = "center"
        qnode_kwargs["approx_order"] = 2
    state_disc_cost = qnetvo.linear_probs_cost_fn(
        state_disc_ansatz, np.eye(2), qnode_kwargs=qnode_kwargs
    )
    behavior_fn = qnetvo.behavior_fn(state_disc_ansatz, qnode_kwargs=qnode_kwargs)

    opt_dict = optimize(
        lambda settings: state_disc_cost(*settings),
        np.array(state_disc_ansatz.rand_network_settings()),
        behavior_fn=behavior_fn,
        step_size=step_size,
        num_steps=num_steps,
        verbose=verbose,
        filepath=data_path,
        optimizer=optimizer,
        background_rate=background_rate,
        shot_rate=shot_rate,
        filename="optimization_" + datetime_str,
        domain_periods=[hwp_period, qwp_period] * 2 + [hwp_period, qwp_period],
    )
    if not numerics:
        motors.close()
    return opt_dict


def qubit_dimensionality_witnessing(
    filepath,
    shots=10000,
    shot_rate=3000000,
    background_counts=300,
    three_wp=False,
    numerics=False,
    step_size=0.02,
    num_steps=50,
    verbose=True,
    optimizer="gradient-descent",
    fixed_settings=[0, 0],
    hwp_period=HWP_PERIOD,
    qwp_period=QWP_PERIOD,
    test_mode=False,
    diff_method="parameter-shift",
):
    """
    Performs a hardware-compatible optimization of a quantum violation of the simplest qubit
    dimensionality witnessing test. A violation of the dimensionality witness requires two noncommuting
    measurement bases and non-orthogonal state encodings.

    By default, this function connects to the hardware backend and both the state preparations and
    measurements are optimized.

    :param filepath: The directory to save data to. Subdirectories will be created within this directory.
    :type filepath: String

    :param shots: The number of shots to use in calculating gradients.
    :type: Int

    :param shot_rate: The rate at which shots are emitted from the source. Used for hardware implementation.
    :type: Int

    :param two_wp: If True, the measurement waveplates will be fixed.
    :type twp_wp: Bool

    :param numerics: If True, a simulator backend is used in place of hardware.
    :type numerics: Bool

    :param step_size: The gradient descent step size.
    :type step_size: Float

    :param num_steps: The number of gradient descent steps.
    :type num_steps: Int

    :param verbose: If True, the optimizer runs in verbose mode.
    :type verbose: Bool

    :param fixed_settings: A two element array.
    :type fixed_settings: List[Float]

    :param use_adam: If `True`, use an adaptive step size in gradient descent.
    :type use_adam: Bool

    :param hwp_period: The domain of unique half wave-plate operations.
    :type hwp_period: Float

    :param qwp_period: The domain of unique quarter wave-plate operations.
    :type qwp_period: Float

    :returns: An optimization dictionary output from the `optimize` method.
    :rtype: Dict
    """
    data_path = make_dir(filepath)

    def arb_unitary(settings, wires):
        HWPRotStage(settings[0], wires=[wires[0]])
        QWPRotStage(settings[1], wires=[wires[0]])

    if numerics:
        data_path += "numerics/"
        dev_name = "kinesis.simulator"
        additional_dev_kwargs = {
            "shot_rate": shot_rate,
            "shots": shots,
            "background_counts": background_counts,
            "qwp_domain_period": qwp_period,
            "hwp_domain_period": hwp_period,
        }
    else:
        motors = ElliptecMotors(test_mode=test_mode)
        data_path += "hardware/"
        dev_name = "elliptec.rotator"
        additional_dev_kwargs = {
            "motors": motors,
            "test_mode": test_mode,
            "filepath": filepath,
        }

    make_dir(data_path)

    if three_wp:
        data_path += "three_wp/"
        fixed_setting_ids = [7, 9]
        fixed_settings = fixed_settings[0:2]  # these settings will be held constant
    else:
        data_path += "four_wp/"
        fixed_setting_ids = []
        fixed_settings = []

    make_dir(data_path)

    datetime_str = datetime_now_string()

    prep_num_in = 3
    meas_num_in = 2

    # network ansatz class manages circuit construction and settings
    dim_wit_ansatz = qnetvo.NetworkAnsatz(
        [qnetvo.PrepareNode(num_in=prep_num_in, wires=[0], ansatz_fn=arb_unitary, num_settings=2)],
        [qnetvo.MeasureNode(num_in=meas_num_in, wires=[0], ansatz_fn=arb_unitary, num_settings=2)],
        dev_kwargs={
            "name": dev_name,
            "shot_rate": shot_rate,
            "shots": shots,
            **additional_dev_kwargs,
        },
    )

    # construct a linear inequality cost function that
    # corresponds to the minimal qubit dimensionality witness.
    dim_wit_cost = qnetvo.linear_probs_cost_fn(
        dim_wit_ansatz,
        np.array(
            [
                [1, 1, 1, -1, -1, 0],
                [-1, -1, -1, 1, 1, 0],
            ]
        ),
        qnode_kwargs={"diff_method": diff_method},
    )

    return optimize(
        lambda settings: dim_wit_cost(*settings),
        np.array(
            dim_wit_ansatz.rand_network_settings(
                fixed_setting_ids=fixed_setting_ids,
                fixed_settings=fixed_settings,
            )
        ),
        step_size=step_size,
        num_steps=num_steps,
        verbose=verbose,
        filepath=data_path,
        optimizer=optimizer,
        filename="optimization_" + datetime_str,
        domain_periods=[hwp_period, qwp_period] * prep_num_in
        + [hwp_period, qwp_period] * meas_num_in,
    )
