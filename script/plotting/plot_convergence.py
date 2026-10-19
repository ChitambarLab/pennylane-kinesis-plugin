import pennylane_kinesis as pk
from datetime import datetime, timezone
import argparse
import matplotlib.pyplot as plt
from pennylane_kinesis.file_utilities import get_files, get_date_from_filename, datetime_now_string, read_json
from pathlib import Path
import numpy as np

_COLORS = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

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
        default="convergence_histogram",
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
    parser.add_argument(
        "--min_cost", default=False, action="store_true", help="Plot min cost value instead of the last cost value."
    )
    parser.add_argument("--diff", default="parameter-shift", help="Differentiation method used in the optimization.")

    parser.add_argument(
        "--total", default=False, action="store_true", help="Plot total convergence instead of divided by step-size."
    )

    args = parser.parse_args()

    if args.filepath is None:
        print("Please provide a filepath to the data folder. (-f)")
        exit()

    title = f"Min cost values" if args.min_cost else "Final cost values"
    title += f"\n(adagrad, {args.diff}, {'two_wp' if args.two_wp else 'four_wp'}, fiber)"

    datetime_str = datetime_now_string()
    step_size_list = args.step_size_list
    filepath = args.filepath
    plot_filename = args.plot_filename

    if not args.total:
        fig, ax = plt.subplots(nrows=2, ncols=2)
    else:
        fig, ax = plt.subplots()
    fig.suptitle(
        title,
        fontweight="bold",
        fontsize=14,
    )

    filepath_type = "/numerics/" if args.numerics else "/hardware/"
    filepath_wp = "two_wp/" if args.two_wp else "four_wp/"

    background_rate = []
    shot_rates = []
    all_final_cost_vals = []
    all_max_cost_vals = []
    c_id = 0
    plot_num = 0
    max_binned = 0
    for step_size in args.step_size_list:
        step_size_filepath = Path(
            filepath + "step_size_" + str(step_size).replace(".", "_") + filepath_type + filepath_wp
        )

        std_filenames = get_files(step_size_filepath, r"optimization_.*\.json")

        for f in std_filenames:
            print(f"Using {f}")
        std_opt_dicts = [read_json(filename) for filename in std_filenames]

        std_aggregate_data = []
        final_cost_vals = []
        max_cost_vals = []
        for j, std_opt_dict in enumerate(std_opt_dicts):
            cost_vals = np.array(std_opt_dict["cost_vals"]) + np.ones(len(std_opt_dict["cost_vals"])) * 2
            max_cost_vals.append(np.min(cost_vals))
            final_cost_vals.append(cost_vals[-1])
            all_final_cost_vals.append(cost_vals[-1])
            all_max_cost_vals.append(np.min(cost_vals))
            for i in range(len(std_opt_dict["cost_vals"])):
                if len(std_aggregate_data) <= i:
                    std_aggregate_data += [[]]

                std_aggregate_data[i] += [std_opt_dict["cost_vals"][i] + 2]
            if args.background:
                if np.sum(std_opt_dict["background_rate"]) > 1000:
                    print("Background rate too high!")
                    print(f"Background rate: {np.sum(std_opt_dict['background_rate'])}")
                    print(f"std_opt_dict['background_rate']: {std_opt_dict['background_rate']}")
                    print(f"filename: {std_filenames[j]}")
                    input()
                else:
                    background_rate += [np.sum(std_opt_dict["background_rate"])]

            shot_rates += [np.sum(std_opt_dict["shot_rate"])]

        std_mean_cost_vals = []
        std_std_err_cost_vals = []
        std_min_cost_vals = []
        for i in range(len(std_aggregate_data)):
            std_mean_cost_vals += [np.mean(std_aggregate_data[i])]
            std_std_err_cost_vals += [np.std(std_aggregate_data[i]) / np.sqrt(len(std_aggregate_data[i]))]
            std_min_cost_vals += [np.min(std_aggregate_data[i])]

        std_np_mean = np.array(std_mean_cost_vals)
        std_np_std_err = np.array(std_std_err_cost_vals)

        if not args.total:
            curr_ax = ax.flat[plot_num]
            counts, bins = np.histogram(
                max_cost_vals if args.min_cost else final_cost_vals, bins=np.logspace(-3, 0, 30)
            )
            print(counts)
            max_binned = max(max_binned, np.max(counts))
            curr_ax.hist(
                max_cost_vals if args.min_cost else final_cost_vals,
                bins=np.logspace(-3, 0, 30),
                alpha=3 / 4,
                label=str(step_size),
                color=_COLORS[c_id],
                stacked=True,
                histtype="barstacked",
            )
            c_id += 1
            curr_ax.set_xscale("log")
            curr_ax.grid()
            plot_num += 1

    if args.background:
        shot_rates = np.array(shot_rates)
        print(f"Average shot rates: {np.mean(shot_rates)}")
        shot_rate_std_err = np.std(shot_rates) / np.sqrt(len(shot_rates))
        print(f"Shot rate std err: {shot_rate_std_err}")
        background_rate = np.array(background_rate)
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

    if args.total:
        ax.hist(
            all_max_cost_vals if args.min_cost else all_final_cost_vals,
            bins=np.logspace(-3, 0, 30),
            alpha=3 / 4,
            color=_COLORS[c_id],
            stacked=False,
            histtype="bar",
        )
        ax.axvline(
            x=background_mean / np.mean(shot_rates),
            color="black",
            linestyle="--",
            label="Background \n(" + str(np.mean(background_rate).round(1)) + " cps)",
        )
        plt.xscale("log")
    else:
        for a in ax.flat:
            a.set_ylim(0, max_binned + 0.5)
            a.axvline(
                x=background_mean / np.mean(shot_rates),
                color="black",
                linestyle="--",
                label="Background \n(" + str(np.mean(background_rate).round(1)) + " cps)",
            )
        bigplot = fig.add_subplot(111, frameon=False)
        bigplot.set_ylabel("Number of trials")
        bigplot.spines["top"].set_color("none")
        bigplot.spines["bottom"].set_color("none")
        bigplot.spines["left"].set_color("none")
        bigplot.spines["right"].set_color("none")
        bigplot.set_xticks([])
        bigplot.set_yticks([])
        bigplot.yaxis.set_label_coords(-0.1, 0.5)
        bigplot.xaxis.set_label_coords(0.5, -0.1)
        if args.min_cost:
            bigplot.set_xlabel("Min cost value")
        else:
            bigplot.set_xlabel("Final cost value")

        plt.draw()

    fig.legend(loc="center right", bbox_to_anchor=(1, 0.8))

    Path.mkdir(Path(filepath + "plots/"), exist_ok=True)
    plt.savefig(filepath + "plots/" + plot_filename + "_" + datetime_str)
