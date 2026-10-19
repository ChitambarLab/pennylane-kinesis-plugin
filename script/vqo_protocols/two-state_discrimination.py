import pennylane as qml
from pennylane import numpy as qnp
import qnetvo
from matplotlib import pyplot as plt
import sys
import argparse
import time
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

def print_bar(msg):
    bar_length = len(msg) + 15  # Set the bar length to match the message length
    bar = "#" * bar_length  # Generate the bar dynamically
    padding = (len(bar) - len(msg)) // 2

    formatted_msg = f"{'#' * padding}{msg}{'#' * (len(bar) - len(msg) - padding)}"

    print(bar)
    print(formatted_msg)
    print(bar)

def plot_state_discrimination(state_disc_opt_dict, args, folder, datetime, suffix):
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

    filename = datetime+"two-state-discrimination"
    filename += "-" + str(args.step_size)
    if args.numerics:
        filename += "-numerics"
    if args.two_wp:
        filename += "-twowp"

    output_file = pk.winapi_path(folder + filename + suffix + ".png")
    plt.savefig(output_file)
    plt.close()


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
    parser.add_argument("-v", "--verbose", default=False, action="store_true", help="More verbose output.")

    parser.add_argument("-s", "--step_size", default=0.08, help="Step size to use.")

    parser.add_argument("--num_steps", default=20, help="Num steps to use per optimization.")

    parser.add_argument(
        "--background",
        default=[300, 100],
        nargs="+",
        type=int,
        help="Number of background counts, one per channel.",
    )

    parser.add_argument("--profile", default=False, action="store_true", help="Profile the code.")

    parser.add_argument(
        "-d",
        "--diff",
        default="parameter-shift",
        help="Gradient method to use. Can be: 'parameter-shift' or 'finite-diff'",
    )

    parser.add_argument(
        "-o",
        "--optimizer",
        default="adagrad",
        help="Optimizer to use.\n Can be: 'gradient-descent' 'adam', 'momentum' or 'adagrad'",
    )

    parser.add_argument(
        "--shots",
        default=10000,
        type=float,
        help="Number of shots to take.",
    )

    parser.add_argument("--num_trials", default=40, help="number of trials to run.")

    args = parser.parse_args()

    filepath = "./data/two-state_discrimination/"

    if args.numerics:
        filepath += "numerics/"
    else:
        filepath += "hardware/"

    if args.two_wp:
        filepath += "two_wp/"
    else:
        filepath += "four_wp/"

    datetime = pk.datetime_now_string()
    filepath += datetime + "/"

    if args.profile:
        prof = cProfile.Profile()
        print("Starting Profiler")
        prof.enable()

    for i in range(args.num_trials):
        print_bar(f"Starting trial {i} out of {args.num_trials}")

        datetime = pk.datetime_now_string()

        start_time = time.time()
        state_disc_opt_dict = pk.two_state_discrimination(
            filepath=filepath,
            shots=10000,
            shot_rate=[300000, 300000],
            background_counts=args.background,
            two_wp=args.two_wp,
            numerics=args.numerics,
            step_size=float(args.step_size),
            num_steps=int(args.num_steps),
            verbose=args.verbose,
        )

        end_time = time.time()
        time_taken = end_time - start_time

        print_bar(
            f"Took {time_taken} seconds to run VQO on RAC with {args.optimizer} and {args.diff} with step size {str(args.step_size)}."
        )
        plot_state_discrimination(state_disc_opt_dict, args, filepath, datetime, str(i))


    if args.profile:
        prof.disable()
        prof.print_stats()
