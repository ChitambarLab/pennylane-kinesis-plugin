"""
Runs multiple two-state discrimination optimizations with different numbers of shots for each
to determine the dependence of the optimization on the step-size.

By default the script will run on hardware using 4 wave plates.
"""

from pennylane import numpy as qnp
from matplotlib import pyplot as plt
import os
import argparse
import time
import pennylane_kinesis as pk
from pennylane_kinesis.file_utilities import datetime_now_string


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Two state discrimination shot dependence.",
        description="Runs multiple two-state discrimination optimizations with different numbers of shots for each to determine the dependence of the optimization on the step-size.",
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
        default="adagrad",
        help="Optimizer to use.\n Can be: 'gradient-descent' 'adam', 'momentum' or 'adagrad'",
    )

    parser.add_argument(
        "-d",
        "--diff",
        default="parameter-shift",
        help="Gradient method to use. Can be: 'parameter-shift' or 'finite-diff'",
    )

    parser.add_argument("-v", "--verbose", default=False, action="store_true", help="More verbose output.")

    parser.add_argument(
        "-s",
        "--step_size",
        default=0.08,
        type=float,
        help="Step size to use for optimizer.",
    )

    parser.add_argument(
        "--shots_list",
        nargs="+",
        type=int,
        default=[100, 1000, 10000, 100000],
    )

    parser.add_argument(
        "--num_trials",
        default=40,
        type=int,
    )

    parser.add_argument(
        "--shot_rate",
        default=100000,
        type=int,
        help="Shot rate for the experiment.",
    )

    parser.add_argument(
        "--num_steps",
        default=20,
        type=int,
        help="Number of optimization steps.",
    )

    parser.add_argument(
        "-b",
        "--background_counts",
        default=[300, 100],
        nargs="+",
        type=int,
        help="Background counts for the experiment, one per channel.",
    )
    args = parser.parse_args()

    filepath = "./data/two-state_discrimination"
    if args.numerics:
        filepath += "/numerics_shot_dependence/"
    else:
        filepath += "/hardware_shot_dependence/"

    filepath = filepath + args.diff + "/" + args.optimizer + "/"
    script_start_time = datetime_now_string()
    filepath += script_start_time + "/"
    pk.make_dir(filepath)

    shots_list = args.shots_list * args.num_trials

    for i, shot in enumerate(shots_list):
        datetime_str = pk.datetime_now_string()

        shots_filepath = filepath + "shots_" + str(shot).replace(".", "_") + "/"

        print("Running optimization for shot ", shot)

        start_time = time.time()
        state_disc_opt_dict = pk.two_state_discrimination(
            filepath=shots_filepath,
            shots=shot,
            shot_rate=args.shot_rate,
            two_wp=args.two_wp,
            numerics=args.numerics,
            step_size=args.step_size,
            num_steps=args.num_steps,
            verbose=args.verbose,
            optimizer=args.optimizer,
            diff_method=args.diff,
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
            str(shot),
        )

        datetime_str = datetime_now_string()
        f = open(shots_filepath + datetime_str + "runtime.txt", "a")
        f.write(str(end_time - start_time))

        """
        Plotting
        """

        fig, (ax1, ax2) = plt.subplots(ncols=2)

        fig.suptitle(
            "VQO of Two-State Discrimination with \n"
            + args.optimizer
            + " and "
            + args.diff
            + "\n with shots "
            + str(shot)
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
        filename += "-" + str(shot)
        filename += "-" + str(args.step_size).replace(".", "-")
        if args.numerics:
            filename += "-numerics"
        if args.two_wp:
            filename += "-twowp"

        import os

        output_file = pk.winapi_path(shots_filepath + datetime_str + filename + ".png")
        plt.savefig(output_file)
        plt.close()

        time.sleep(0.5)
