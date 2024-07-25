import pennylane as qml
from pennylane import numpy as qnp
import qnetvo
from matplotlib import pyplot as plt
import sys
import argparse

import pennylane_kinesis as pk
import cProfile
from pstats import SortKey

"""
Optimizes the state discrimination success probability for two states.

By default the script will run on hardware using 4 wave plates.

Use the `numerics` argument to run the script using the simulator backend.
Use the `two_wp` argument to run the script with fixed mneasurement waveplates.

```
$ python script/two-state_discrimination.py numerics
```
"""


def plot_state_discrimination(state_disc_opt_dict, args, datetime=None):
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

    filepath = "./data/two-state_discrimination/"
    if args.numerics:
        filepath += "numerics/"
    else:
        filepath += "hardware/"

    if args.two_wp:
        filepath += "two_wp/"
    else:
        filepath += "four_wp/"

    filename = "two-state-discrimination"
    filename += "-" + str(args.step_size)
    if args.numerics:
        filename += "-numerics"
    if args.two_wp:
        filename += "-twowp"

    if datetime is None:
        datetime_str = pk.datetime_now_string()
    else:
        datetime_str = datetime

    output_file = pk.winapi_path(filepath + datetime_str + filename + ".png")
    plt.savefig(output_file)


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
        "-v", "--verbose", default=False, action="store_true", help="More verbose output."
    )

    parser.add_argument("-s", "--step_size", default=0.08, help="Step size to use.")

    parser.add_argument("-p", "--plot", default=None, help="Only plot given filename, do not run")

    parser.add_argument("--num_steps", default=20, help="Num steps to use per optimization.")

    parser.add_argument("--background", default=300, help="Number of background counts.")

    args = parser.parse_args()

    if not args.plot:
        prof = cProfile.Profile()
        print("Starting Profiler")
        prof.enable()
        state_disc_opt_dict = pk.two_state_discrimination(
            filepath="./data/two-state_discrimination/",
            shots=10000,
            shot_rate=3000000,
            background_counts=int(args.background),
            two_wp=args.two_wp,
            numerics=args.numerics,
            step_size=float(args.step_size),
            num_steps=int(args.num_steps),
            verbose=args.verbose,
        )
        prof.disable()
        prof.print_stats()
        plot_state_discrimination(state_disc_opt_dict, args)
    else:
        print(args.plot)
        datetime_str = args.plot.split("\\")[-1].split("_")[1].split(".")[0]
        print(datetime_str)
        opt_dict = pk.read_json(pk.winapi_path(args.plot))
        plot_state_discrimination(opt_dict, args, datetime_str)

    """
    Plotting
    """
