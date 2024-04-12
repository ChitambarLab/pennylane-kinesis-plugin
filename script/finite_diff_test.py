import pennylane as qml
from pennylane import numpy as qnp
import qnetvo
from matplotlib import pyplot as plt
import sys
import argparse
import time
import pennylane_kinesis as pk

"""
Optimizes the state discrimination success probability for two states.

By default the script will run on hardware using 4 wave plates.

Use the `numerics` argument to run the script using the simulator backend.
Use the `two_wp` argument to run the script with fixed mneasurement waveplates.

```
$ python script/two-state_discrimination.py numerics
```
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Two state discrimination",
        description="Optimizes the state discrimination success probability for two states.",
    )

    parser.add_argument(
        "-2",
        "--two_wp",
        default=False,
        action="store_true",
        help="If only using two waveplates (fixed measurement waveplates).",
    )
    parser.add_argument(
        "-n",
        "--numerics",
        default=False,
        action="store_true",
        help="If using the simulator backend.",
    )

    parser.add_argument(
        "-o",
        "--optimizer",
        default="gradient-descent",
        help="Optimizer to use.\n Can be: 'gradient-descent' 'adam', 'momentum' or 'adagrad'"
    )
    parser.add_argument(
        '-d',
        '--diff',
        default='parameter-shift',
        help="Gradient method to use. Can be: 'parameter-shift' or 'finite-diff'",
    )

    parser.add_argument(
        "-v", "--verbose", default=False, action="store_true", help="More verbose output."
    )

    parser.add_argument(
        "-s",
        "--step_size",
        default=0.08,
        type=float,
        help="Step size to use for optimizer.",
    )

    parser.add_argument(
        "--num_steps",
        default=20,
        type=int,
        help="Number of optimization steps.",
    )
    args = parser.parse_args()

    start_time = time.time()
    state_disc_opt_dict = pk.two_state_discrimination(
        filepath="./data/two-state_discrimination/",
        shots=10000,
        shot_rate=3000000,
        two_wp=args.two_wp,
        numerics=args.numerics,
        step_size=args.step_size,
        num_steps=args.num_steps,
        verbose=args.verbose,
        optimizer=args.optimizer,
        diff_method=args.diff
    )

    end_time = time.time()

    print("took", end_time - start_time, "seconds to run VQO on Two-State Discrimination with", args.optimizer, "and", args.diff)

    """
    Plotting
    """

    fig, (ax1, ax2) = plt.subplots(ncols=2)

    fig.suptitle("VQO of Two-State Discrimination with " + args.optimizer + " and " + args.diff)

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

    filename="two-state-discrimination"
    filename+="-"+args.diff
    filename+="-"+args.optimizer
    filename+="-"+str(args.num_steps)
    filename+="-"+str(args.step_size)
    if args.numerics:
        filename+="-numerics"
    if args.two_wp:
        filename+="-twowp"
    

    plt.savefig("./data/finite-diff/"+filename+".png")