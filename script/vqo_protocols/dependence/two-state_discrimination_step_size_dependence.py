import pennylane as qml
from pennylane import numpy as qnp
import qnetvo
from matplotlib import pyplot as plt
import sys
import os
import argparse
import time
import pennylane_kinesis as pk
from pennylane_kinesis.file_utilities import datetime_now_string
import datetime
from pennylane_kinesis.tomography_utilities import take_tomography_data

"""
Runs multiple two-state discrimination optimizations with different step-sizes for each
to determine the dependence of the optimization on the step-size.

By default the script will run on hardware using 4 wave plates.
"""


def print_bar(msg):
    bar_length = len(msg) + 15  # Set the bar length to match the message length
    bar = "#" * bar_length  # Generate the bar dynamically
    padding = (len(bar) - len(msg)) // 2

    formatted_msg = f"{'#' * padding}{msg}{'#' * (len(bar) - len(msg) - padding)}"

    print(bar)
    print(formatted_msg)
    print(bar)


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
        help="Optimizer to use.\n Can be: 'gradient-descent' 'adam', 'momentum' or 'adagrad'",
    )

    parser.add_argument(
        "--shots",
        default=10000,
        type=float,
        help="Number of shots to take.",
    )

    parser.add_argument(
        "--shot_rate",
        default=qnp.array([418000, 400000]),
        type=list,
        help="The shot rate of the experiment (for numerics).",
    )

    parser.add_argument(
        "--step_size_list",
        nargs="+",
        default=[0.01, 0.05, 0.1, 0.2, 0.5],
    )

    parser.add_argument(
        "--num_trials",
        default=40,
        type=int,
        help="Number of trials to run for each step size in the step-size list.",
    )

    parser.add_argument(
        "-d",
        "--diff",
        default="parameter-shift",
        help="Gradient method to use. Can be: 'parameter-shift' or 'finite-diff'",
    )

    parser.add_argument("-v", "--verbose", default=False, action="store_true", help="More verbose output.")

    parser.add_argument(
        "--background_counts",
        default=400,
        type=int,
        help="Background count rate for the experiment (for numerics)",
    )

    parser.add_argument(
        "--num_steps",
        default=20,
        type=int,
        help="Number of optimization steps.",
    )
    parser.add_argument(
        "--meas_only",
        default=False,
        action="store_true",
        help="For two waveplates, use only measurement waveplates instead of prepare.",
    )
    parser.add_argument("--qpt", default=False, action="store_true", help="Run process tomography each trial.")
    args = parser.parse_args()

    filepath = "./data/state_discrimination"
    if args.numerics:
        filepath += "/numerics_step_size_dependence/"
    else:
        filepath += "/new_hardware_step_size_dependence/"

    datetime_str = pk.datetime_now_string()

    filepath = filepath + args.diff + "/" + args.optimizer + "/"
    script_start_time = datetime_now_string()
    filepath += script_start_time + "/"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    step_size_list = args.step_size_list * args.num_trials

    print_bar(
        f"Taking {'two_wp' if args.two_wp else 'four_wp'} data for two-state discrimination with {args.optimizer} and {args.diff}."
    )
    print("Step size list: ", step_size_list)
    print("Number of trials: ", args.num_trials)
    for i, step_size in enumerate(step_size_list):
        print_bar(f"Starting trial {i} out of {len(step_size_list)}")
        cst_time = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-6)))
        print(f"Started trial at {cst_time.strftime('%Y-%m-%d %H:%M:%S %Z')}.")

        step_size_filepath = filepath + "step_size_" + str(step_size).replace(".", "_") + "/"

        if args.qpt:
            print("Taking tomography data...")
            start_time = time.time()
            take_tomography_data(step_size_filepath, None, 1, 5, False)
            end_time = time.time()
            time_taken = end_time - start_time
            print(f"Took {time_taken} seconds to run process tomography.")

        print("Running optimization for step_size ", step_size)

        start_time = time.time()
        state_disc_opt_dict = pk.two_state_discrimination(
            filepath=step_size_filepath,
            shots=args.shots,
            shot_rate=args.shot_rate,
            two_wp=args.two_wp,
            numerics=args.numerics,
            step_size=float(step_size),
            num_steps=args.num_steps,
            verbose=args.verbose,
            optimizer=args.optimizer,
            diff_method=args.diff,
            background_counts=args.background_counts,
            meas_only=args.meas_only,
        )

        end_time = time.time()

        print(
            "took",
            end_time - start_time,
            "seconds to run VQO on Two-State Discrimination with",
            args.optimizer,
            "and",
            args.diff,
            "with ",
            str(step_size),
        )
        end_time = time.time()
        time_taken = end_time - start_time

        datetime_str = datetime_now_string()
        f = open(step_size_filepath + datetime_str + "runtime.txt", "a")
        f.write(str(end_time - start_time))

        time_remaining = time_taken * (len(step_size_list) - i)
        print_bar(
            f"At this rate, it will take {time_remaining:.3f} seconds or {time_remaining / 60:.3f} minutes or {time_remaining / 3600:.3f} hours."
        )

        """
        Plotting
        """

        fig, (ax1, ax2) = plt.subplots(ncols=2)

        fig.suptitle(
            "VQO of Two-State Discrimination with \n"
            + args.optimizer
            + " and "
            + args.diff
            + "\n with step_size "
            + str(step_size)
        )

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

        filename = "two-state-discrimination"
        filename += "-" + args.diff
        filename += "-" + args.optimizer
        filename += "-" + str(args.num_steps)
        filename += "-" + str(step_size)
        if args.numerics:
            filename += "-numerics"
        if args.two_wp:
            filename += "-twowp"

        output_file = pk.winapi_path(step_size_filepath + datetime_str + filename + ".png")
        plt.savefig(output_file)
        plt.close()
        time.sleep(0.5)
