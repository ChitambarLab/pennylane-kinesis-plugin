import pennylane_kinesis as pk
from datetime import datetime, timezone
import argparse

"""This script generates a plot showing the optimization precision as
dependent upon the optimization step size. Data for this plot is obtained
using the `./script/vqo_protocols/dependence/two-state_discrimination_step_size_dependence.py` script.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Step size dependence",
        description="Generates a plot showing the optimization precision as dependent upon the optimization step size.",
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
        default="step_size_dependence",
        help="Filename for the plot.",
    )

    parser.add_argument(
        "-s",
        "--step_size_list",
        nargs="+",
        default=[0.01, 0.05, 0.1, 0.2, 0.5],
        help="List of step sizes to plot. default = [0.01, 0.05, 0.1, 0.2, 0.5]",
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

    parser.add_argument("-b", "--background", default=None, help="Plot background count line.")

    args = parser.parse_args()

    if args.filepath is None:
        print("Please provide a filepath to the data folder. (-f)")
        exit()

    date_after = datetime.strptime("2024-06-27T00-35-01Z", "%Y-%m-%dT%H-%M-%SZ").replace(
        tzinfo=timezone.utc
    )
    if not args.two_wp:
        args.subtitle.replace("two_wp", "four_wp")

    print(args.title)
    pk.plot_step_size_dependence(
        args.filepath,
        args.step_size_list,
        args.plot_filename,
        date_after=date_after,
        numerics=args.numerics,
        two_wp=args.two_wp,
        # add_background=True,
        background= args.background,
        shot_rate=250000,
        title=f"Step Size Dependence \n(adagrad, parameter-shift, {'two_wp' if args.two_wp else 'four_wp'})",
    )
