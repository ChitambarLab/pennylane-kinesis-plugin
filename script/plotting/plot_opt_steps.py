"""Plot optimization step-times."""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pennylane_kinesis import datetime_now_string, get_files, read_json
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Optimization Time Plotter",
        description="Generates a plot for optimization time based on opt_step_time.",
    )

    parser.add_argument(
        "-f",
        "--filepath_param_shift",
        default=None,
        help="Path to the data folder.",
    )

    parser.add_argument(
        "-g",
        "--filepath_finite_diff",
        default=None,
        help="Path to the data folder.",
    )

    parser.add_argument("--rac", default=False, action="store_true")

    parser.add_argument("-2", "--two", default=False, action="store_true")
    parser.add_argument(
        "-p",
        "--plot_filename",
        default="time_bar_chart",
        help="Filename for the plot.",
    )

    parser.add_argument(
        "--title",
        default="Optimization Step Time Comparison",
        help="Title for the plot.",
    )

    parser.add_argument(
        "-s",
        "--save_path",
        default=None,
    )

    args = parser.parse_args()

    if args.filepath_param_shift is None or args.filepath_finite_diff is None or args.save_path is None:
        print("Please provide a filepath to the data folder or save folder. (-f, -g, -s)")
        exit()

    title = args.title

    filepath_type = "/hardware/"

    if args.two and not args.rac:
        filepath_type += "two_wp/"
        title += "\n State-discrimination (two waveplate)"
    elif not args.rac:
        filepath_type += "four_wp/"
        title += "\n State-discrimination (four waveplate)"

    if args.rac and args.two:
        title += "\n Random Access Coding (2 bits)"
    if args.rac and not args.two:
        title += "\n Random Access Coding (3 bits)"

    datetime_str = datetime_now_string()

    fig, ax1 = plt.subplots()

    fig.suptitle(
        title,
        fontweight="bold",
        fontsize=14,
    )

    opt_param_shift_files = get_files(Path(args.filepath_param_shift + filepath_type), r"[0-9]*_optimization_.*\.json")
    opt_finite_diff_files = get_files(Path(args.filepath_finite_diff + filepath_type), r"[0-9]*_optimization_.*\.json")

    opt_param_shift_dicts = [read_json(file) for file in opt_param_shift_files]
    opt_finite_diff_dicts = [read_json(file) for file in opt_finite_diff_files]

    aggregate_param_shift = []
    aggregate_finite_diff = []
    for j, opt_dict in enumerate(opt_param_shift_dicts):
        for time in opt_dict["opt_step_time"]:
            aggregate_param_shift.append(time)

    for j, opt_dict in enumerate(opt_finite_diff_dicts):
        for time in opt_dict["opt_step_time"]:
            aggregate_finite_diff.append(time)

    average_param_shift_time = np.mean(aggregate_param_shift)
    average_finite_diff_time = np.mean(aggregate_finite_diff)

    std_param_shift = np.std(aggregate_param_shift)
    std_finite_diff = np.std(aggregate_finite_diff)

    ax1.bar(
        ["Parameter Shift", "Finite Difference"],
        [average_param_shift_time, average_finite_diff_time],
        yerr=[std_param_shift, std_finite_diff],
        capsize=8,
    )
    ax1.set_ylabel("Average time per step (s)")

    average_fraction = (average_finite_diff_time) / average_param_shift_time

    print(f"Average finite_diff: {average_finite_diff_time}")
    print(f"Average finite_diff: {average_param_shift_time}")

    fig.tight_layout()
    ax1.text(
        0.6,
        0.6,
        f"Finite-difference time is {(average_fraction) * 100:.1f}%\n of Parameter-shift time.",
        transform=ax1.transAxes,
    )

    Path.mkdir(Path(args.save_path + "/plots/"), exist_ok=True)
    plot_filepath = args.save_path + f"/plots/{args.plot_filename}_{datetime_str}"
    plt.savefig(Path(plot_filepath))
    print(f"Plot saved at {plot_filepath}")
