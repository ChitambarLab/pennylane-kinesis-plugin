import pennylane as qml
from pennylane import numpy as qnp
import qnetvo
from matplotlib import pyplot as plt
import sys

import pennylane_kinesis as pk


"""
Optimizes the violation of a qubit dimensionality witness.

Arguments:
* `numerics` - runs on simulator device rather than on hardware 
"""


def arb_unitary(settings, wires):
    pk.HWPRotStage(settings[0], wires=[wires[0]])
    pk.QWPRotStage(settings[1], wires=[wires[0]])


if __name__ == "__main__":
    if sys.argv[1] == "numerics":
        data_path = "./data/dimensionality_witnessing/numerics/"
        shots = 1000
        dev_name = "kinesis.simulator"
    else:
        data_path = "./data/dimensionality_witnessing/hardware/"
        shots = 120000
        dev_name = "kinesis.cage.rotator"

    pk.make_dir(data_path)

    shot_rate = 120000

    datetime_str = pk.datetime_now_string()

    # network ansatz class manages circuit construction and settings
    dim_wit_ansatz = qnetvo.NetworkAnsatz(
        [qnetvo.PrepareNode(num_in=3, wires=[0], ansatz_fn=arb_unitary, num_settings=2)],
        [qnetvo.MeasureNode(num_in=2, wires=[0], ansatz_fn=arb_unitary, num_settings=2)],
        dev_kwargs={
            "name": dev_name,
            "shot_rate": shot_rate,
            "shots": shots,
        },
    )

    # construct a linear inequality cost function that
    # corresponds to the minimal qubit dimensionality witness.
    dim_wit_cost = qnetvo.linear_probs_cost_fn(
        dim_wit_ansatz,
        qnp.array(
            [
                [1, 1, 1, -1, -1, 0],
                [-1, -1, -1, 1, 1, 0],
            ]
        ),
    )

    dim_wit_opt_dict = pk.optimize(
        lambda settings: dim_wit_cost(*settings),
        qnp.array(dim_wit_ansatz.rand_network_settings()),
        step_size=0.04,
        num_steps=30,
        verbose=True,
        filepath=data_path,
        filename="optimization_" + datetime_str,
    )

    """
    Plotting
    """

    fig, (ax1, ax2) = plt.subplots(ncols=2)

    fig.suptitle("VQO of Dimensionality Witness")

    ax1.plot(
        range(len(dim_wit_opt_dict["cost_vals"])),
        -1 * qnp.array(dim_wit_opt_dict["cost_vals"]),
        label="Optimization Trajectory",
    )
    ax1.plot(
        range(len(dim_wit_opt_dict["cost_vals"])),
        [3] * len(dim_wit_opt_dict["cost_vals"]),
        label="Classical Bound",
    )

    ax1.plot(
        range(len(dim_wit_opt_dict["cost_vals"])),
        [1 + 2 * qnp.sqrt(2)] * len(dim_wit_opt_dict["cost_vals"]),
        label="Quantum Bound",
    )

    ax1.legend()

    ax2.semilogy(
        range(len(dim_wit_opt_dict["cost_vals"])),
        [1 + 2 * qnp.sqrt(2)] + qnp.array(dim_wit_opt_dict["cost_vals"]),
        label="optimization trajectory",
    )
    ax2.semilogy(
        range(len(dim_wit_opt_dict["cost_vals"])),
        [1 + 2 * qnp.sqrt(2) - 3] * len(dim_wit_opt_dict["cost_vals"]),
        label="classical bound",
    )

    ax1.set_ylabel(r"Dimensionality Witness Score")
    ax1.set_xlabel("Optimization Iteration")

    ax2.set_ylabel("Distance to Optimal Dimensionality Witness Score")
    ax2.set_xlabel("Optimization Iteration")

    plt.tight_layout()

    plt.show()
