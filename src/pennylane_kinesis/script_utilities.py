from .file_utilities import datetime_now_string, make_dir
from .optimization import optimize
from .operators import HWPRotStage, QWPRotStage

import qnetvo
from pennylane import numpy as qnp


def two_state_discrimination(
    filepath,
    shots=10000,
    shot_rate=3000000,
    two_wp=False,
    numerics=False,
    step_size=0.08,
    num_steps=20,
    verbose=True,
    fixed_settings=[0, 0],
):
    """
    Performs a hardware-compatible optimization of minimal error discrimination of two encoded states.
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
    else:
        data_path += "hardware/"
        dev_name = "kinesis.cage.rotator"

    make_dir(data_path)

    if two_wp:
        data_path += "two_wp/"
        fixed_setting_ids = [4, 5]
        fixed_settings = fixed_settings[0:2]  # these settings will be held at constant value
    else:
        data_path += "four_wp/"
        fixed_setting_ids = []
        fixed_settings = []

    make_dir(data_path)

    datetime_str = datetime_now_string()

    # network ansatz class manages circuit construction and settings
    state_disc_ansatz = qnetvo.NetworkAnsatz(
        [qnetvo.PrepareNode(num_in=2, wires=[0], ansatz_fn=arb_unitary, num_settings=2)],
        [qnetvo.MeasureNode(num_in=1, wires=[0], ansatz_fn=arb_unitary, num_settings=2)],
        dev_kwargs={
            "name": dev_name,
            "shot_rate": shot_rate,
            "shots": shots,
        },
    )

    # construct a linear inequality cost function that
    # corresponds to the minimal qubit dimensionality witness.
    state_disc_cost = qnetvo.linear_probs_cost_fn(state_disc_ansatz, qnp.eye(2))

    return optimize(
        lambda settings: state_disc_cost(*settings),
        qnp.array(
            state_disc_ansatz.rand_network_settings(
                fixed_setting_ids=fixed_setting_ids,
                fixed_settings=fixed_settings,
            )
        ),
        step_size=step_size,
        num_steps=num_steps,
        verbose=verbose,
        filepath=data_path,
        filename="optimization_" + datetime_str,
    )
