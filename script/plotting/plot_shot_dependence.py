import pennylane_kinesis as pk
from datetime import datetime
import argparse

"""This script generates a plot showing the optimization precision as
dependent upon the optimization step size. Data for this plot is obtained numerically
using the `./script/fiber_inversion_numeric_step_size_dependence.py` script.
"""

filepath = "./data/state_discrimination/hardware_shot_dependence/parameter-shift/gradient-descent/2024-04-30T23-30-05Z/"
# step_size_list = [1.0, 0.2, 0.15, 0.1, 0.05, 0.01, 0.001]
# step_size_list = [0.2, 0.15, 0.1, 0.05]
# shots_list = [10, 100, 1000, 10000, 100000]
shots_list = [10, 100, 1000, 10000, 100000]
plot_filename = "shots_dependence_parameter_shift_grad_descent_hardware"

date_after = datetime.strptime("2024-04-25T00-17-20Z", "%Y-%m-%dT%H-%M-%SZ")

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

    date_after = datetime.strptime("2024-06-27T00-35-01Z", "%Y-%m-%dT%H-%M-%SZ")

    pk.plot_shot_dependence(
        args.filepath,
        shots_list=args.shots_list,
        plot_filename=args.plot_filename,
        numerics=args.numerics,
        two_wp=args.two_wp,
        add_background=True,
        background=300,
        shot_rate=10000,
        title="Number of Shots Vs.\nOptimization Performance \n(gradient-descent, \n parameter-shift, hardware)",
    )
