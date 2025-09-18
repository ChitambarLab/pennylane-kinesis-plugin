import pennylane as qml
from pennylane import numpy as qnp
import qnetvo
from matplotlib import pyplot as plt
import sys
import os
import argparse
import time
import numpy as np
import pennylane_kinesis as pk
from pennylane_kinesis.file_utilities import datetime_now_string
try:
    # If you're running this on hardware that doesn't have the swabian timetagger library
    # such as when you're doing numerics work, this import will break
    from pennylane_kinesis.tomography_utilities import take_tomography_data
except:
    pass
import datetime

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

    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        help="More verbose output.",
    )

    parser.add_argument(
        "--background_counts",
        default=[300,100],
        type=list,
        help="Background count rate for the experiment (for numerics)",
    )

    parser.add_argument(
        "--num_steps",
        default=20,
        type=int,
        help="Number of optimization steps.",
    )
    parser.add_argument(
        "--qpt",
        default=False,
        action="store_true",
        help="Run process tomography each trial.",
    )
    args = parser.parse_args()

    filepath = "./data/state_discrimination"
    if args.numerics:
        filepath += "/numerics_meas_setting_dependence/"
    else:
        filepath += "/new_hardware_meas_setting_dependence/"

    datetime_str = pk.datetime_now_string()

    filepath = filepath + args.diff + "/" + args.optimizer + "/"
    script_start_time = datetime_now_string()
    filepath += script_start_time + "/"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]

    meas_settings_list = []
    for i in range(5):
        for j in range(5):
            meas_settings_list.append([angles[i], angles[j]])

    step_size = 0.2
    print_bar(
        f"Taking {'two_wp' if args.two_wp else 'four_wp'} data for two-state discrimination with {args.optimizer} and {args.diff}."
    )
    print("Meas_settings_list", meas_settings_list)
    print("Number of trials: ", args.num_trials)
    meas_settings_list = meas_settings_list * args.num_trials
    for i, meas_setting in enumerate(meas_settings_list):
        print_bar(f"Starting trial {i} out of {len(meas_settings_list)}")
        cst_time = datetime.datetime.now(
            datetime.timezone(datetime.timedelta(hours=-6))
        )
        print(f"Started trial at {cst_time.strftime('%Y-%m-%d %H:%M:%S %Z')}.")

        meas_setting_filepath = (
            filepath
            + "meas_setting"
            + str(meas_setting).replace("[", "").replace("]", "").replace(",", "_")
            + "/"
        )
        print(meas_setting_filepath)

        if args.qpt:
            print("Taking tomography data...")
            start_time = time.time()
            take_tomography_data(meas_setting_filepath, None, 1, 5, False)
            end_time = time.time()
            time_taken = end_time - start_time
            print(f"Took {time_taken} seconds to run process tomography.")

        print("Running optimization for meas_setting ", meas_setting)

        start_time = time.time()
        state_disc_opt_dict = pk.two_state_discrimination(
            filepath=meas_setting_filepath,
            shots=int(args.shots),
            shot_rate=args.shot_rate,
            two_wp=args.two_wp,
            numerics=args.numerics,
            step_size=float(step_size),
            num_steps=args.num_steps,
            verbose=args.verbose,
            optimizer=args.optimizer,
            diff_method=args.diff,
            fixed_meas_settings=meas_setting,
            background_counts=args.background_counts,
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
            str(meas_setting),
        )
        end_time = time.time()
        time_taken = end_time - start_time

        datetime_str = datetime_now_string()
        f = open(meas_setting_filepath + datetime_str + "runtime.txt", "a")
        f.write(str(end_time - start_time))

        time_remaining = time_taken * (len(meas_settings_list) - i)
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
            + "\n with meas_setting "
            + str(meas_setting)
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
        filename += "-" + str(meas_setting).replace("[", "").replace("]", "").replace(
            ",", "_"
        )
        if args.numerics:
            filename += "-numerics"
        if args.two_wp:
            filename += "-twowp"

        output_file = pk.winapi_path(
            meas_setting_filepath + datetime_str + filename + ".png"
        )
        plt.savefig(output_file)
        plt.close()
        time.sleep(0.5)
