import pennylane as qml
from pennylane import numpy as qnp
import qnetvo
from matplotlib import pyplot as plt
import sys

import pennylane_kinesis as pk

"""
Optimizes the random access coding advantage.

Arguments:
* `numerics` - runs on simulator device rather than on hardware 
"""

hwp_period = qnp.pi / 2
qwp_period = qnp.pi


def arb_unitary(settings, wires):
    pk.HWPRotStage(settings[0], wires=[wires[0]])
    pk.QWPRotStage(settings[1], wires=[wires[0]])


def hwp_unitary(settings, wires):
    zero_angle = 0  # set according to hardware

    pk.HWPRotStage(settings[0], wires=[wires[0]])
    pk.QWPRotStage(zero_angle, wires=[wires[0]])


shot_rate = 3000000


if __name__ == "__main__":
    if sys.argv[1] == "numerics":
        data_path = pk.make_dir("./data/random_access_coding/numerics/")
        shots = 1000
        dev_name = "kinesis.simulator"
        dev_kwargs = {
            "name": dev_name,
            "shots": shots,
            "shot_rate": shot_rate,
            "qwp_domain_period": qwp_period,
            "hwp_domain_period": hwp_period,
            "background_counts": 10,
        }
    else:
        data_path = pk.make_dir("./data/random_access_coding/hardware/")
        shots = 3000000
        dev_name = "kinesis.cage.rotator"
        dev_kwargs = {
            "name": dev_name,
            "shots": shots,
            "shot_rate": shot_rate,
        }

    pk.make_dir(data_path)

    datetime_str = pk.datetime_now_string()

    # network ansatz class manages circuit construction and settings
    rac_ansatz = qnetvo.NetworkAnsatz(
        [qnetvo.PrepareNode(num_in=4, wires=[0], ansatz_fn=arb_unitary, num_settings=2)],
        [qnetvo.MeasureNode(num_in=2, wires=[0], ansatz_fn=hwp_unitary, num_settings=1)],
        dev_kwargs=dev_kwargs,
    )

    # construct a linear inequality cost function that
    # corresponds to the minimal qubit random access code.
    rac_cost = qnetvo.linear_probs_cost_fn(
        rac_ansatz,
        qnp.array(
            [
                [1, 0, 0, 1, 1, 1, 0, 0],
                [0, 1, 1, 0, 0, 0, 1, 1],
            ]
        ),
    )

    rac_opt_dict = pk.optimize(
        lambda settings: rac_cost(*settings),
        qnp.array(rac_ansatz.rand_network_settings()),
        step_size=0.06,
        num_steps=30,
        verbose=True,
        filepath=data_path,
        filename="optimization_" + datetime_str,
        domain_periods=[hwp_period, qwp_period] * 4 + [hwp_period] * 2,
    )

    """
    Plotting
    """

    fig, (ax1, ax2) = plt.subplots(ncols=2)

    fig.suptitle("VQO of Random Access Code")

    ax1.plot(
        range(len(rac_opt_dict["cost_vals"])),
        -1 * qnp.array(rac_opt_dict["cost_vals"]),
        label="Optimization Trajectory",
    )
    ax1.plot(
        range(len(rac_opt_dict["cost_vals"])),
        [6] * len(rac_opt_dict["cost_vals"]),
        label="Classical Bound",
    )

    ax1.plot(
        range(len(rac_opt_dict["cost_vals"])),
        [4 + 2 * qnp.sqrt(2)] * len(rac_opt_dict["cost_vals"]),
        label="Quantum Bound",
    )

    ax1.legend()

    ax2.semilogy(
        range(len(rac_opt_dict["cost_vals"])),
        [4 + 2 * qnp.sqrt(2)] + qnp.array(rac_opt_dict["cost_vals"]),
        label="optimization trajectory",
    )
    ax2.semilogy(
        range(len(rac_opt_dict["cost_vals"])),
        [4 + 2 * qnp.sqrt(2) - 6] * len(rac_opt_dict["cost_vals"]),
        label="classical bound",
    )

    ax1.set_ylabel(r"Random Access Code Score")
    ax1.set_xlabel("Optimization Iteration")

    ax2.set_ylabel("Distance to Optimal Random Access Code Score")
    ax2.set_xlabel("Optimization Iteration")

    plt.tight_layout()

    plt.show()
