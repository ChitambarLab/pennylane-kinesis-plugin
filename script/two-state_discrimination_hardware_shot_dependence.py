import pennylane as qml
from pennylane import numpy as qnp
from matplotlib import pyplot as plt
import qnetvo

import pennylane_kinesis as pk

"""This script is intended to run on hardware. Its goal is to perform optimizations over a
range of shots and to save the corresponding results.
After many runs of this script, data can be aggregated to analyze the effect of shots on
optimization precision.


NOTE: Before running script make sure that the "./data/state_discrimination_hardware_shot_dependence"
      directory exists with subdirectories "/shots_N" where N in [10,100,1000,10000, 100000].
"""

theta_period = qnp.pi / 2
phi_period = qnp.pi
filepath = "./data/state_discrimination_hardware_shot_dependence/numerics/"
shot_rate = 50000

shots_list = [10, 100, 1000, 10000, 100000] * 5
# shots_list = [10000, 10, 1000, 100]
# shots_list = [10, 35, 100, 350, 1000, 3500, 10000]

opt_dicts = []


def arb_unitary(settings, wires):
    pk.HWPRotStage(settings[0], wires=[wires[0]])
    pk.QWPRotStage(settings[1], wires=[wires[0]])


def fixed_unitary(settings, wires):
    pk.HWPRotStage(0, wires=[wires[0]])
    pk.QWPRotStage(0, wires=[wires[0]])


# Set test_mode=True to run on simulator
motors = pk.KinesisMotors(test_mode=False)

for shots in shots_list:
    datetime_str = pk.datetime_now_string()

    dev_kwargs = {
        "name": "kinesis.cage.rotator",
        "motors": motors,
        "shots": shots,
        "shot_rate": shot_rate,
    }
    sim_dev_kwargs = {
        "name": "kinesis.simulator",
        "shots": shots,
        "qwp_domain_period": phi_period,
        "hwp_domain_period": theta_period,
        "background_counts": 4,
    }
    eval_kwargs = {
        "name": "kinesis.cage.rotator",
        "motors": motors,
        "shots": 100_000,
        "shot_rate": shot_rate,
    }
    sim_eval_kwargs = {
        "name": "kinesis.simulator",
        "shots": 100_000,
        "qwp_domain_period": phi_period,
        "hwp_domain_period": theta_period,
        "background_counts": 4,
    }

    # NOTE: uncomment sim_dev_kwargs if running on simulator
    ansatz = qnetvo.NetworkAnsatz(
        [
            qnetvo.PrepareNode(
                num_in=2, wires=[0], ansatz_fn=arb_unitary, num_settings=2
            )
        ],
        [qnetvo.MeasureNode(num_out=2, wires=[0], ansatz_fn=fixed_unitary)],
        # dev_kwargs=sim_dev_kwargs,
        dev_kwargs=dev_kwargs,
    )

    cost = qnetvo.linear_probs_cost_fn(
        ansatz, qnp.eye(2), qnode_kwargs={"diff_method": "parameter-shift"}
    )

    ansatz_eval = qnetvo.NetworkAnsatz(
        [
            qnetvo.PrepareNode(
                num_in=2, wires=[0], ansatz_fn=arb_unitary, num_settings=2
            )
        ],
        [qnetvo.MeasureNode(num_out=2, wires=[0], ansatz_fn=fixed_unitary)],
        # dev_kwargs=sim_eval_kwargs,
        dev_kwargs=eval_kwargs,
    )

    eval_cost = qnetvo.linear_probs_cost_fn(
        ansatz_eval, qnp.eye(2), qnode_kwargs={"diff_method": "parameter-shift"}
    )

    shots_filepath = filepath + "shots_" + str(shots) + "/"

    print("Running optimization for shots ", shots)
    opt_dict = pk.optimize(
        lambda settings: cost(*settings),
        qnp.array(ansatz.rand_network_settings()),
        step_size=0.1325,
        num_steps=50,
        filepath=shots_filepath,
        filename="optimization_" + datetime_str,
        domain_periods=[theta_period, phi_period] * 2,
        eval_cost=lambda settings: eval_cost(*settings),
        shots=shots,
        shot_rate=shot_rate,
    )

    plt.semilogy(
        range(len(opt_dict["cost_vals"])),
        qnp.array(opt_dict["cost_vals"]) + 2,
        label="shots = " + str(shots),
    )


plt.legend()
plt.show()
