from pennylane import numpy as qnp
from matplotlib import pyplot as plt
import qnetvo
import sys

import pennylane_kinesis as pk

"""This script is intended to run on hardware. Its goal is to perform optimizations over a
range of step sizes and to save the corresponding results.
After many runs of this script, data can be aggregated to analyze the effect of shots on
optimization precision.


NOTE: Before running script make sure that the "./data/state_discrimination_hardware_step_size_dependence"
      directory exists with subdirectories "/step_size_0_N" where `_` replaces `.` in the stepsize float.
"""

theta_period = qnp.pi / 2
phi_period = qnp.pi
period_array = qnp.array([theta_period, theta_period, phi_period, phi_period])
filepath = "./data/state_discrimination_hardware_step_size_dependence/two_wp/"
shot_rate = 100000
shots = 10000


if __name__ == "__main__":
    if sys.argv[1] == "numerics":
        data_path = "./data/two-state_discrimination/numerics/"
        dev_name = "kinesis.simulator"
    else:
        data_path = "./data/two-state_discrimination/hardware/"
        dev_name = "kinesis.cage.rotator"

    # step_size_list = [0.01, 0.5, 0.2, 0.1, 0.05] * 5
    step_size_list = [0.12, 0.17, 0.1325, 0.135, 0.15, 0.2] * 5

    opt_dicts = []

    def arb_unitary(settings, wires):
        pk.HWPRotStage(settings[0], wires=[wires[0]])
        pk.QWPRotStage(settings[1], wires=[wires[0]])

    def fixed_unitary(settings, wires):
        pk.HWPRotStage(178.8, wires=[wires[0]])
        pk.QWPRotStage(99.1, wires=[wires[0]])

    motors = pk.KinesisMotors()

    dev_kwargs = {
        "name": "kinesis.cage.rotator",
        "motors": motors,
        "shots": shots,
        "shot_rate": shot_rate,
    }
    eval_kwargs = {
        "name": "kinesis.cage.rotator",
        "motors": motors,
        "shots": 100000,
        "shot_rate": shot_rate,
    }
    sim_dev_kwargs = {
        "name": "kinesis.simulator",
        "shots": shots,
        "qwp_domain_period": phi_period,
        "hwp_domain_period": theta_period,
    }

    # NOTE: uncomment sim_dev_kwargs if running on simulator
    ansatz = qnetvo.NetworkAnsatz(
        [qnetvo.PrepareNode(num_in=2, wires=[0], ansatz_fn=arb_unitary, num_settings=2)],
        [qnetvo.MeasureNode(num_out=2, wires=[0], ansatz_fn=fixed_unitary)],
        # dev_kwargs=sim_dev_kwargs,
        dev_kwargs=dev_kwargs,
    )

    eval_ansatz = qnetvo.NetworkAnsatz(
        [qnetvo.PrepareNode(num_in=2, wires=[0], ansatz_fn=arb_unitary, num_settings=2)],
        [qnetvo.MeasureNode(num_out=2, wires=[0], ansatz_fn=fixed_unitary)],
        # dev_kwargs=sim_dev_kwargs,
        dev_kwargs=eval_kwargs,
    )

    cost = qnetvo.linear_probs_cost_fn(
        ansatz, qnp.eye(2), qnode_kwargs={"diff_method": "parameter-shift"}
    )

    eval_cost = qnetvo.linear_probs_cost_fn(
        eval_ansatz, qnp.eye(2), qnode_kwargs={"diff_method": "parameter-shift"}
    )

    for step_size in step_size_list:
        datetime_str = pk.datetime_now_string()

        step_size_filepath = filepath + "step_size_" + str(step_size).replace(".", "_") + "/"

        print("Running optimization for step_size ", step_size)
        opt_dict = pk.optimize(
            lambda settings: cost(*settings),
            qnp.array(ansatz.rand_network_settings()),
            step_size=step_size,
            num_steps=50,
            filepath=step_size_filepath,
            filename="optimization_" + datetime_str,
            domain_periods=[theta_period, phi_period] * 2,
            eval_cost=lambda settings: eval_cost(*settings),
            shots=shots,
            shot_rate=shot_rate,
        )

        plt.loglog(
            range(len(opt_dict["cost_vals"])),
            qnp.array(opt_dict["cost_vals"]) + 2,
            label="step_size = " + str(step_size).replace(".", "_"),
        )

    plt.legend()
    plt.show()
