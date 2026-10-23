"""This script generates a plot showing the optimization precision as
dependent upon the number of measured samples. Data for this plot is obtained
using the `./script/vqo_protocols/dependence/two-state_discrimination_shot_dependence.py` script.
"""

import pennylane_kinesis as pk
import argparse

filepath = "./data/state_discrimination/hardware_shot_dependence/parameter-shift/gradient-descent/2024-04-30T23-30-05Z/"
shots_list = [10, 100, 1000, 10000, 100000]
plot_filename = "shots_dependence_parameter_shift_grad_descent_hardware"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Shot dependence",
        description="Generates a plot showing the optimization precision as dependent upon the number of shots taken.",
    )

    parser.add_argument(
        "-f",
        "--filepath",
        default=None,
        help="Path to the data folder.",
    )

    parser.add_argument(
        "-p",
        "--plot_filename",
        default="shot_dependence",
        help="Filename for the plot.",
    )

    parser.add_argument(
        "-s",
        "--shots_list",
        nargs="+",
        default=[10, 100, 1000, 10000, 100000],
        help="List of shots to plot. default = [10, 100, 1000, 10000, 100000]",
    )

    parser.add_argument(
        "-n",
        "--numerics",
        default=False,
        action="store_true",
        help="If using the simulator backend.",
    )

    parser.add_argument(
        "-2",
        "--two_wp",
        default=False,
        action="store_true",
        help="If only using two waveplates (fixed measurement waveplates).",
    )

    args = parser.parse_args()

    if args.filepath is None:
        print("Please provide a filepath to the data folder. (-f)")
        exit()

    pk.plot_shot_dependence(
        args.filepath,
        shots_list=args.shots_list,
        plot_filename=args.plot_filename,
        numerics=args.numerics,
        two_wp=args.two_wp,
        add_background=False,
        background=300,
        shot_rate=10000,
        title=f"Number of Shots vs. Optimization Performance \n (adagrad, parameter-shift, {'two_wp' if args.two_wp else 'four_wp'})",
    )
