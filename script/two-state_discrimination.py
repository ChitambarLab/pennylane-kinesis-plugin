import pennylane as qml
from pennylane import numpy as qnp
import qnetvo
from matplotlib import pyplot as plt
import sys

import pennylane_kinesis as pk

"""
Optimizes the state discrimination success probability for two states.

Arguments:
* `numerics` - runs on simulator device rather than on hardware 
"""


def arb_unitary(settings, wires):
    pk.HWPRotStage(settings[0], wires=[wires[0]])
    pk.QWPRotStage(settings[1], wires=[wires[0]])


if __name__ == "__main__":
    if sys.argv[1] == "numerics":
        data_path = "./data/two-state_discrimination/numerics/"
        shots = 1000
        dev_name = "kinesis.simulator"
    else:
        data_path = "./data/two-state_discrimination/hardware/"
        shots = 3000000
        dev_name = "kinesis.cage.rotator"

    pk.make_dir(data_path)

    shot_rate = 3000000

    datetime_str = pk.datetime_now_string()

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
    state_disc_cost = qnetvo.linear_probs_cost_fn(
        state_disc_ansatz,
        qnp.eye(2),
    )

    state_disc_opt_dict = pk.optimize(
        lambda settings: state_disc_cost(*settings),
        qnp.array(state_disc_ansatz.rand_network_settings()),
        step_size=0.1,
        num_steps=20,
        verbose=True,
        filepath=data_path,
        filename="optimization_" + datetime_str,
    )

    """
    Plotting
    """

    fig, (ax1, ax2) = plt.subplots(ncols=2)

    fig.suptitle("VQO of Two-State Discrimination")

    ax1.plot(
        range(len(state_disc_opt_dict["cost_vals"])),
        -1 * qnp.array(state_disc_opt_dict["cost_vals"]),
        label="Optimization Trajectory",
    )
    ax1.plot(
        range(len(state_disc_opt_dict["cost_vals"])),
        [2] * len(state_disc_opt_dict["cost_vals"]),
        label="Bound",
    )

    ax1.legend()

    ax2.semilogy(
        range(len(state_disc_opt_dict["cost_vals"])),
        [2] + qnp.array(state_disc_opt_dict["cost_vals"]),
        label="optimization trajectory",
    )

    ax1.set_ylabel(r"Two-State Discrimination")
    ax1.set_xlabel("Optimization Iteration")

    ax2.set_ylabel("Distance to Optimal State Discrimination Score")
    ax2.set_xlabel("Optimization Iteration")

    plt.tight_layout()

    plt.show()
