import numpy as np
import qnetvo

import pennylane as qml
import pennylane.numpy as qnp
from .file_utilities import datetime_now_string
from .file_utilities import make_dir
from .kinesis_cage_rotator_device import KinesisMotors
from .operators import (
    HWPRotStage,
    QWPRotStage,
    VariableWPRotStage,
    WaveplateTypes,
    QWP_RETARDANCE_A,
    QWP_RETARDANCE_B,
    HWP_RETARDANCE_B,
)
from .optimization import optimize
import pennylane_kinesis as pk
from .elliptec_rotator_device import ElliptecMotors, DeviceIndex
import time
import pickle

# If fiber kraus ops are defined
# with open("./data/fiber_kraus_operations/kraus_ops_03_04_fixed.pkl", "rb") as f:
#    fiber_ops = pickle.load(f)

HWP_PERIOD = np.pi / 2
QWP_PERIOD = np.pi


def two_state_discrimination(
    filepath,
    shots=10000,
    shot_rate=[300000, 400000],
    background_counts=[300, 100],
    two_wp=False,
    numerics=False,
    step_size=0.08,
    num_steps=20,
    optimizer="adagrad",
    verbose=True,
    fixed_meas_settings=[0, 0],
    hwp_period=HWP_PERIOD,
    qwp_period=QWP_PERIOD,
    test_mode=False,
    diff_method="parameter-shift",
    efficiencies=[1, 1],
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

    def arb_unitary_retardance(settings, wires):
        HWPRotStage(settings[0], wires=[wires[0]])
        VariableWPRotStage(settings[1], wires=wires[0], retardance=QWP_RETARDANCE_A, wp_type=WaveplateTypes.QWP)

    # Function to use for MeasureNode if we are not moving the measurement waveplates.
    def meas_fn(settings, wires):
        HWPRotStage(settings[0], wires=wires[0:1])
        QWPRotStage(settings[1], wires=wires[0:1])

    def meas_fn_swapped_order(settings, wires):
        QWPRotStage(settings[1], wires=wires[0:1])
        HWPRotStage(settings[0], wires=wires[0:1])

    def meas_fn_retardance(settings, wires):
        VariableWPRotStage(settings[0], wires=[wires[0]], retardance=HWP_RETARDANCE_B, wp_type=WaveplateTypes.HWP)
        VariableWPRotStage(settings[1], wires=wires[0], retardance=QWP_RETARDANCE_B, wp_type=WaveplateTypes.QWP)

    def meas_fn_swapped_order_retardance(settings, wires):
        VariableWPRotStage(settings[1], wires=wires[0], retardance=QWP_RETARDANCE_B, wp_type=WaveplateTypes.QWP)
        VariableWPRotStage(settings[0], wires=[wires[0]], retardance=HWP_RETARDANCE_B, wp_type=WaveplateTypes.HWP)

    def noise_fn(settings, wires):
        kraus_ops = fiber_ops
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
            "h_efficiency": efficiencies[0],
            "v_efficiency": efficiencies[1],
        }
    else:
        motors = ElliptecMotors(test_mode=test_mode)
        data_path += "hardware/"
        dev_name = "elliptec.rotator"
        additional_dev_kwargs = {
            "motors": motors,
            "test_mode": test_mode,
            "filepath": filepath,
            "waveplate_order": [DeviceIndex.aHWP, DeviceIndex.aQWP, DeviceIndex.bQWP, DeviceIndex.bHWP],
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
            ansatz_fn=lambda *args: meas_fn_swapped_order_retardance(fixed_meas_settings, args[1]),
            num_settings=0,
        )

    else:
        data_path += "four_wp/"
        meas_node = qnetvo.MeasureNode(num_in=1, wires=[0], ansatz_fn=meas_fn_swapped_order_retardance, num_settings=2)

    make_dir(data_path)

    datetime_str = datetime_now_string()

    # network ansatz class manages circuit construction and settings

    # network ansatz class manages circuit construction and settings
    state_disc_ansatz = qnetvo.NetworkAnsatz(
        [qnetvo.PrepareNode(num_in=2, wires=[0], ansatz_fn=arb_unitary_retardance, num_settings=2)],
        # [qnetvo.NoiseNode(wires=[0], ansatz_fn=noise_fn)],
        [meas_node],
        dev_kwargs={
            "name": dev_name,
            "shot_rate": shot_rate,
            "shots": shots,
            "debug": verbose,
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
        state_disc_ansatz,
        np.eye(2),
        qnode_kwargs=qnode_kwargs,
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
    shot_rate=[300000, 400000],
    background_counts=[300, 100],
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
        domain_periods=[hwp_period, qwp_period] * prep_num_in + [hwp_period, qwp_period] * meas_num_in,
    )


def rac(
    filepath,
    shots=10000,
    shot_rate=[300000, 400000],
    background_counts=[300, 100],
    rac_3=False,
    numerics=False,
    step_size=0.08,
    num_steps=20,
    optimizer="adagrad",
    verbose=True,
    fixed_meas_settings=[0, 0],
    hwp_period=HWP_PERIOD,
    qwp_period=QWP_PERIOD,
    test_mode=False,
    diff_method="parameter-shift",
    efficiencies=[1, 1],
):
    data_path = make_dir(filepath)

    def arb_unitary_retardance(settings, wires):
        HWPRotStage(settings[0], wires=[wires[0]])
        VariableWPRotStage(settings[1], wires=wires[0], retardance=QWP_RETARDANCE_A, wp_type=WaveplateTypes.QWP)

    # Function to use for MeasureNode if we are not moving the measurement waveplates.
    def meas_fn(settings, wires):
        HWPRotStage(settings[0], wires=wires[0:1])
        QWPRotStage(settings[1], wires=wires[0:1])

    def meas_fn_swapped_order(settings, wires):
        QWPRotStage(settings[1], wires=wires[0:1])
        HWPRotStage(settings[0], wires=wires[0:1])

    def meas_fn_retardance(settings, wires):
        VariableWPRotStage(settings[0], wires=[wires[0]], retardance=HWP_RETARDANCE_B, wp_type=WaveplateTypes.HWP)
        VariableWPRotStage(settings[1], wires=wires[0], retardance=QWP_RETARDANCE_B, wp_type=WaveplateTypes.QWP)

    def meas_fn_swapped_order_retardance(settings, wires):
        VariableWPRotStage(settings[1], wires=wires[0], retardance=QWP_RETARDANCE_B, wp_type=WaveplateTypes.QWP)
        VariableWPRotStage(settings[0], wires=[wires[0]], retardance=HWP_RETARDANCE_B, wp_type=WaveplateTypes.HWP)

    def hwp_only_meas_fn_retardance(settings, wires):
        VariableWPRotStage(0, wires=wires[0], retardance=QWP_RETARDANCE_B, wp_type=WaveplateTypes.QWP)
        VariableWPRotStage(settings[0], wires=[wires[0]], retardance=HWP_RETARDANCE_B, wp_type=WaveplateTypes.HWP)

    if numerics:
        data_path += "numerics/"
        dev_name = "kinesis.simulator"
        additional_dev_kwargs = {
            "shot_rate": shot_rate,
            "shots": shots,
            "background_counts": background_counts,
            "qwp_domain_period": qwp_period,
            "hwp_domain_period": hwp_period,
            "h_efficiency": efficiencies[0],
            "v_efficiency": efficiencies[1],
        }
    else:
        motors = ElliptecMotors(test_mode=test_mode)
        data_path += "hardware/"
        dev_name = "elliptec.rotator"
        additional_dev_kwargs = {
            "motors": motors,
            "test_mode": test_mode,
            "filepath": filepath,
            "waveplate_order": [DeviceIndex.aHWP, DeviceIndex.aQWP, DeviceIndex.bQWP, DeviceIndex.bHWP],
        }

    pk.make_dir(data_path)

    datetime_str = pk.datetime_now_string()

    # network ansatz class manages circuit construction and settings
    if rac_3:
        rac_ansatz = qnetvo.NetworkAnsatz(
            [qnetvo.PrepareNode(num_in=8, wires=[0], ansatz_fn=arb_unitary_retardance, num_settings=2)],
            # [qnetvo.NoiseNode(wires=[0], ansatz_fn=noise_fn)],
            [
                qnetvo.MeasureNode(
                    num_in=3,
                    num_out=2,
                    wires=[0],
                    ansatz_fn=meas_fn_swapped_order_retardance,
                    num_settings=2,
                )
            ],
            dev_kwargs={
                "name": dev_name,
                "shot_rate": shot_rate,
                "shots": shots,
                **additional_dev_kwargs,
            },
        )
    else:
        rac_ansatz = qnetvo.NetworkAnsatz(
            [qnetvo.PrepareNode(num_in=4, wires=[0], ansatz_fn=arb_unitary_retardance, num_settings=2)],
            # [qnetvo.NoiseNode(wires=[0], ansatz_fn=noise_fn)],
            [qnetvo.MeasureNode(num_in=2, wires=[0], ansatz_fn=meas_fn_swapped_order_retardance, num_settings=2)],
            dev_kwargs={
                "name": dev_name,
                "shot_rate": shot_rate,
                "shots": shots,
                **additional_dev_kwargs,
            },
        )

    # Take background and shot_rate data
    device = qml.device(**rac_ansatz.dev_kwargs)
    shot_rate = device.shot_rate.tolist()
    background_rate = device.take_background()

    _qnode_kwargs = {"diff_method": diff_method}
    if diff_method == "finite-diff":
        _qnode_kwargs["h"] = 1e-3
        _qnode_kwargs["strategy"] = "center"
        _qnode_kwargs["approx_order"] = 2
    # construct a linear inequality cost function that
    # corresponds to the minimal qubit random access code.
    if rac_3:
        rac_cost = qnetvo.linear_probs_cost_fn(
            rac_ansatz,
            qnp.array(
                [
                    [1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1],
                ]
            ),
            qnode_kwargs=_qnode_kwargs,
        )
    else:
        rac_cost = qnetvo.linear_probs_cost_fn(
            rac_ansatz,
            qnp.array(
                [
                    [1, 0, 0, 1, 1, 1, 0, 0],
                    [0, 1, 1, 0, 0, 0, 1, 1],
                ]
            ),
            qnode_kwargs=_qnode_kwargs,
        )

    behavior_fn = qnetvo.behavior_fn(rac_ansatz, qnode_kwargs=_qnode_kwargs)

    if rac_3:
        domain_periods = [hwp_period, qwp_period] * 11
    else:
        domain_periods = [hwp_period, qwp_period] * 4 + [hwp_period, qwp_period] * 2

    rac_opt_dict = optimize(
        lambda settings: rac_cost(*settings),
        qnp.array(rac_ansatz.rand_network_settings()),
        background_rate=background_rate,
        shot_rate=shot_rate,
        verbose=verbose,
        behavior_fn=behavior_fn,
        shots=int(shots),
        numerics=numerics,
        step_size=float(step_size),
        num_steps=int(num_steps),
        diff_method=diff_method,
        background_counts=background_counts,
        optimizer=optimizer,
        filepath=data_path,
        filename="optimization_" + datetime_str,
        domain_periods=domain_periods,
    )
    if not numerics:
        motors.close()

    return rac_opt_dict
