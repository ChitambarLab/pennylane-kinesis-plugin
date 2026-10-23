"""Plot optimization performance for different settings of measurement waveplates for two-state discrimination."""

import pennylane_kinesis as pk
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import argparse
from pathlib import Path
from pennylane_kinesis.file_utilities import (
    get_files,
    read_json,
    datetime_now_string,
)
import numpy as np
import matplotlib.pyplot as plt
import os

_COLORS = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"] * 3
PI = 3.141592653589793
PI_2 = 1.5707963267948966
PI_4 = 0.7853981633974483
PI_3_4 = 2.356194490192345

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Meas Setting Plotter",
        description="Generates a plot showing the optimization performance for different settings of measurement waveplates for 2-state discrimination.",
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
        default="meas_dependence",
        help="Filename for the plot.",
    )

    parser.add_argument(
        "-n",
        "--numerics",
        default=False,
        action="store_true",
        help="If using the simulator backend.",
    )

    parser.add_argument(
        "-b",
        "--background",
        action="store_true",
        default=False,
        help="Plot background count line.",
    )

    parser.add_argument(
        "--no_plot",
        default=False,
        action="store_true",
        help="If you don't want to plot the data and only want the statistics at the end.",
    )

    parser.add_argument(
        "--title",
        default="Local Optima in State Discrimination\n",
        help="Title for the plot.",
    )

    parser.add_argument("--subtitle", default="\n(adagrad, parameter-shift)")

    parser.add_argument(
        "--best",
        default=False,
        action="store_true",
        help="Plot the best cost achieved out of all runs for a particular measurement setting.",
    )

    args = parser.parse_args()

    if args.filepath is None:
        print("Please provide a filepath to the data folder. (-f)")
        exit()

    filepath_type = "/numerics/" if args.numerics else "/hardware/"

    plt.rc("font", family="sans-serif", size=12)
    title = args.title
    title += " parameter-shift"
    title += ", adagrad, $\eta_0 = 0.2$"
    datetime_str = datetime_now_string()

    date_path = Path(args.filepath).name
    date_of_run = (
        datetime.strptime(date_path, "%Y-%m-%dT%H-%M-%SZ")
        .replace(tzinfo=timezone.utc)
        .astimezone(ZoneInfo("America/Chicago"))
        .strftime("%Y-%m-%d %H:%M:%S")
    )

    meas_settings_HWP = [str(PI_4)]
    meas_names_HWP = [r"$\frac{\pi}{4}$"]

    meas_settings_QWP = [str(0), str(PI_4), str(PI_2), str(PI_3_4)]
    meas_names_QWP = [
        str(0),
        r"$\frac{\pi}{4}$",
        r"$\frac{\pi}{2}$",
        r"$\frac{3\pi}{4}$",
    ]
    meas_settings_HWP = [meas_settings_QWP[0]]
    meas_names_HWP = [meas_names_QWP[0]]
    all_meas_names = []
    folder_names = []

    for i in range(len(meas_settings_HWP)):
        for j in range(len(meas_settings_QWP)):
            folder_names.append(f"meas_setting{meas_settings_HWP[i]}_ {meas_settings_QWP[j]}")
            all_meas_names.append(
                f"${float(meas_settings_QWP[j]) * 180 / np.pi:.0f}^\circ$ x ${float(meas_settings_HWP[i]) * 180 / np.pi:.0f}^\circ$ "
            )

    two_wp = "/two_wp/"

    N = len(meas_names_HWP)

    fig, ax = plt.subplots(figsize=(7, 5))

    fig.suptitle(
        title,
        fontweight="bold",
        fontsize=14,
    )

    c_id = 0
    for k, meas_setting in enumerate(folder_names):
        opt_files = get_files(
            Path(pk.winapi_path(args.filepath + meas_setting + filepath_type + two_wp)),
            r"[0-9]*_optimization_.*\.json",
        )

        date_path = Path(args.filepath).name
        date_of_run = (
            datetime.strptime(date_path, "%Y-%m-%dT%H-%M-%SZ")
            .replace(tzinfo=timezone.utc)
            .astimezone(ZoneInfo("America/Chicago"))
            .strftime("%Y-%m-%d %H:%M:%S")
        )

        for file in opt_files:
            print(f"Using {file}")

        opt_dicts = [read_json(file) for file in opt_files]
        num_trials = len(opt_dicts)

        background_rates = []
        aggregate_data = []
        shot_rates = []
        for j, opt_dict in enumerate(opt_dicts):
            for i in range(len(opt_dict["cost_vals"])):
                if len(aggregate_data) <= i:
                    aggregate_data += [[]]

                aggregate_data[i] += [opt_dict["cost_vals"][i] / 2 + 1]
            if args.background:
                background_rates += [np.sum(opt_dict["background_rate"])]

            shot_rates += [np.sum(opt_dict["shot_rate"])]

        std_mean_cost_vals = []
        std_std_err_cost_vals = []
        std_min_cost_vals = []
        for i in range(len(aggregate_data)):
            std_mean_cost_vals += [np.mean(aggregate_data[i])]
            std_std_err_cost_vals += [np.std(aggregate_data[i]) / np.sqrt(len(aggregate_data[i]))]
            std_min_cost_vals += [np.min(aggregate_data[i])]

        std_np_mean = np.array(std_mean_cost_vals)
        std_np_std_err = np.array(std_std_err_cost_vals)

        if args.best:
            best_cost_val = np.min(aggregate_data)
            ax.semilogy(
                range(len(std_mean_cost_vals)),
                best_cost_val * np.ones(len(std_mean_cost_vals)),
                alpha=3 / 4,
                linestyle="--",
                label="Best Cost",
                color=_COLORS[c_id],
            )

        ax.semilogy(
            range(len(std_mean_cost_vals)),
            std_mean_cost_vals,
            alpha=3 / 4,
            label=f"{all_meas_names[k]}",
            color=_COLORS[c_id],
            linestyle="--",
        )

        ax.fill_between(
            range(len(std_mean_cost_vals)),
            std_np_mean - std_np_std_err,
            std_np_mean + std_np_std_err,
            alpha=0.1,
        )

        c_id += 1

    if args.background:
        shot_rates = np.array(shot_rates)
        print(f"Average shot rates: {np.mean(shot_rates)}")
        shot_rate_std_err = np.std(shot_rates) / np.sqrt(len(shot_rates))
        print(f"Shot rate std err: {shot_rate_std_err}")
        background_rate = np.array(background_rates)
        background_std_dev = np.std(background_rate)
        background_std_err = np.std(background_rate) / np.sqrt(len(background_rate))
        print(f"max background rate: {np.max(background_rate)}")
        print(f"min background rate: {np.min(background_rate)}")
        print(f"Background std dev: {background_std_dev}")
        print(f"background mean: {np.mean(background_rate)}")
        print(f"sqrt(n):", np.sqrt(len(background_rate)))
        print(f"Background std err: {background_std_err}")
        background_mean = np.mean(background_rate)
        ax.plot(
            range(len(std_np_mean)),
            [background_mean / np.mean(shot_rates)] * len(std_np_mean),
            "k--",
            label="Background (" + str(np.mean(background_rate).round(1)) + " cps)",
        )
        ax.fill_between(
            range(len(std_np_mean)),
            ([(background_mean - background_std_err) / np.mean(shot_rates)] * len(std_np_mean)),
            ([(background_mean + background_std_err) / np.mean(shot_rates)] * len(std_np_mean)),
            alpha=1 / 4,
            color="black",
            linestyle="--",
        )

    ax.set_xlabel("Optimization Iteration")
    ax.set_ylabel("Probability of Undiscriminated States")
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter

    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))

    fig.legend(
        loc="upper right",
        bbox_to_anchor=(0.97, 0.84),
        ncol=2,
        title="bQWP x bHWP Measurement Settings",
    )
    fig.tight_layout()

    plt.grid()

    face_id = 1
    while os.path.exists(
        Path(args.filepath + f"/plots/{args.plot_filename}_{date_of_run.split(' ')[0]}_{face_id}.png")
    ):
        face_id += 1

    if not args.no_plot:
        Path.mkdir(Path(args.filepath + "/plots/"), exist_ok=True)
        plot_filepath = args.filepath + f"/plots/{args.plot_filename}_{date_of_run.split(' ')[0]}_{face_id}"
        plt.savefig(Path(plot_filepath))
        print(f"Plot saved at {plot_filepath}")
