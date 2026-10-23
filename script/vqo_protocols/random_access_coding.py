"""Run VQO on random access coding (2 bit and 3 bit)."""

from pennylane import numpy as qnp
from matplotlib import pyplot as plt
import argparse
import os
import pennylane_kinesis as pk
import time
import numpy as np

MAXIMUM_BOUND_3_BIT = 24
QUANTUM_COST_BOUND_3_BIT = 12 + 12 / np.sqrt(3)
CLASSICAL_COST_BOUND_3_BIT = 18

QUANTUM_BOUND_3_BIT = QUANTUM_COST_BOUND_3_BIT / MAXIMUM_BOUND_3_BIT
CLASSICAL_BOUND_3_BIT = CLASSICAL_COST_BOUND_3_BIT / MAXIMUM_BOUND_3_BIT

MAXIMUM_BOUND_2_BIT = 8
QUANTUM_COST_BOUND_2_BIT = 4 + 2 * np.sqrt(2)
CLASSICAL_COST_BOUND_2_BIT = 6

QUANTUM_BOUND_2_BIT = QUANTUM_COST_BOUND_2_BIT / MAXIMUM_BOUND_2_BIT
CLASSICAL_BOUND_2_BIT = CLASSICAL_COST_BOUND_2_BIT / MAXIMUM_BOUND_2_BIT

BACKGROUND_2_BIT_MULTIPLIER = 1 / 2 + QUANTUM_COST_BOUND_2_BIT / MAXIMUM_BOUND_2_BIT
BACKGROUND_3_BIT_MULTIPLIER = 1 / 2 + QUANTUM_COST_BOUND_3_BIT / MAXIMUM_BOUND_3_BIT


def plot_data(opt_dict, rac_3, filepath, plot_filename, datetime=None, prefix=None):
    if rac_3:
        QUANTUM_BOUND = QUANTUM_BOUND_3_BIT
        QUANTUM_COST_BOUND = QUANTUM_COST_BOUND_3_BIT
        CLASSICAL_BOUND = CLASSICAL_BOUND_3_BIT
        BACKGROUND_MULTIPLIER = BACKGROUND_3_BIT_MULTIPLIER
        MAXIMUM_BOUND = MAXIMUM_BOUND_3_BIT
        title = "RAC Optimization (3 bits)"
    else:
        QUANTUM_BOUND = QUANTUM_BOUND_2_BIT
        QUANTUM_COST_BOUND = QUANTUM_COST_BOUND_2_BIT
        CLASSICAL_BOUND = CLASSICAL_BOUND_2_BIT
        BACKGROUND_MULTIPLIER = BACKGROUND_2_BIT_MULTIPLIER
        MAXIMUM_BOUND = MAXIMUM_BOUND_2_BIT
        title = "RAC Optimization (2 bits)"

    fig, (ax1, ax2) = plt.subplots(ncols=2)

    fig.suptitle(
        title,
        fontweight="bold",
        fontsize=14,
    )

    prob_success = (qnp.array(opt_dict["cost_vals"]) * -1) / MAXIMUM_BOUND
    num_steps = len(opt_dict["cost_vals"])
    background_rate = qnp.sum(opt_dict["background_rate"])
    shot_rate = qnp.sum(qnp.array(opt_dict["shot_rate"]))
    ax1.plot(
        range(num_steps),
        prob_success,
    )
    ax1.plot(
        range(num_steps),
        [CLASSICAL_BOUND] * num_steps,
        label="Classical Bound",
    )

    ax1.plot(
        range(num_steps),
        [QUANTUM_BOUND] * num_steps,
        label="Quantum Bound",
    )

    ax2.semilogy(
        range(num_steps),
        [QUANTUM_BOUND] * num_steps - prob_success,
        label=f"Optimization Trajectory",
    )
    ax2.semilogy(
        range(num_steps),
        [QUANTUM_BOUND - CLASSICAL_BOUND] * num_steps,
        color="orange",
    )
    ax2.semilogy(
        range(num_steps),
        [(background_rate / shot_rate) * BACKGROUND_MULTIPLIER] * num_steps,
        "k--",
        label=f"Background Limit \n(Average {background_rate} cps)",
    )
    ax2.set_yticks([1e-3, 1e-2, 1e-1, 1e-0])

    ax1.set_ylabel("Success Probability")
    ax1.set_xlabel("Optimization Iteration")

    ax2.set_ylabel("Distance to Quantum Bound")
    ax2.set_xlabel("Optimization Iteration")

    fig.legend(loc="lower center", bbox_to_anchor=(0.32, 0.15))
    fig.tight_layout()

    plt.grid()
    if datetime is None:
        datetime_str = pk.datetime_now_string()
    else:
        datetime_str = datetime
    output_file = pk.winapi_path(filepath + prefix + datetime_str + plot_filename + ".png")

    plt.savefig(output_file)
    plt.close()


"""
Optimizes the random access coding advantage.

Arguments:
* `numerics` - runs on simulator device rather than on hardware 
"""

hwp_period = qnp.pi / 2
qwp_period = qnp.pi


def print_bar(msg):
    bar_length = len(msg) + 15  # Set the bar length to match the message length
    bar = "#" * bar_length  # Generate the bar dynamically
    padding = (len(bar) - len(msg)) // 2

    formatted_msg = f"{'#' * padding}{msg}{'#' * (len(bar) - len(msg) - padding)}"

    print(bar)
    print(formatted_msg)
    print(bar)


def arb_unitary(settings, wires):
    pk.HWPRotStage(settings[0], wires=[wires[0]])
    pk.QWPRotStage(settings[1], wires=[wires[0]])


def hwp_unitary(settings, wires):
    zero_angle = 0  # set according to hardware

    pk.HWPRotStage(settings[0], wires=[wires[0]])
    pk.QWPRotStage(zero_angle, wires=[wires[0]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Two state discrimination",
        description="Optimizes the state discrimination success probability for two states.",
    )

    parser.add_argument(
        "-n",
        "--numerics",
        default=False,
        action="store_true",
        help="If using the simulator backend.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        help="More verbose output.",
    )

    parser.add_argument("--num_trials", default=40, type=int, help="number of trials to run.")

    parser.add_argument("-s", "--step_size", default=0.2, type=float, help="Step size to use.")

    parser.add_argument("-p", "--plot", default=None, help="Only plot given filename, do not run")

    parser.add_argument("--num_steps", type=int, default=60, help="Num steps to use per optimization.")

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
        type=int,
        help="Number of shots to take.",
    )

    parser.add_argument(
        "--test",
        default=False,
    )

    parser.add_argument(
        "--rac_3",
        default=False,
        action="store_true",
        help="If using the 3 qubit random access code.",
    )

    args = parser.parse_args()

    filepath = "./data/random_access_coding/"

    if args.rac_3:
        filepath += "rac_3/"
    else:
        filepath += "rac_2/"

    if args.numerics:
        filepath += "numerics/"
    else:
        filepath += "hardware/"

    num_trials = int(args.num_trials)
    datetime_str = pk.datetime_now_string()

    filepath = filepath + args.diff + "/" + args.optimizer + "/"
    script_start_time = pk.datetime_now_string()
    filepath += script_start_time + "/"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    filename = "rac"
    if args.rac_3:
        filename += "_3"
    else:
        filename += "_2"
    filename += "-" + args.diff
    filename += "-" + args.optimizer
    filename += "-" + str(args.num_steps)
    filename += "-" + str(args.step_size)
    if args.numerics:
        filename += "-numerics"

    print(f"Starting random access coding trials. Files will be written to {filepath}")
    for i in range(num_trials):
        print_bar(f"Starting trial {i} out of {num_trials}")
        start_time = time.time()

        rac_opt_dict = pk.rac(
            filepath=filepath,
            shots=args.shots,
            shot_rate=[300000, 300000],
            background_counts=args.background,
            rac_3=args.rac_3,
            numerics=args.numerics,
            step_size=args.step_size,
            num_steps=args.num_steps,
            optimizer=args.optimizer,
            diff_method=args.diff,
            verbose=args.verbose,
            filename_prefix=str(i),
        )

        end_time = time.time()
        time_taken = end_time - start_time

        print_bar(
            f"Took {time_taken} seconds to run VQO on RAC with {args.optimizer} and {args.diff} with {str(args.step_size)}."
        )
        f = open(filepath + datetime_str + "runtime.txt", "a")
        f.write(str(time_taken) + "\n")
        f.close()
        time_remaining = time_taken * (num_trials - i)
        print_bar(
            f"At this rate, it will take {time_remaining:.3f} seconds or {time_remaining / 60:.3f} minutes or {time_remaining / 3600:.3f} hours."
        )

        plot_data(rac_opt_dict, args.rac_3, filepath, filename + ".png", prefix=str(i) + "_")
        time.sleep(0.1)

    print(
        f"Finished random access coding trials. Files written to \n{filepath}\nYou can use this path for the plotting script."
    )
