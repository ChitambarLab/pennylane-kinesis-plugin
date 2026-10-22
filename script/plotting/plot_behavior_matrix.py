import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from pathlib import Path
from pennylane_kinesis.file_utilities import (
    get_files,
    read_json,
    datetime_now_string,
    get_date_from_filename,
)
import argparse

_COLORS = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--filepath",
        default=None,
        help="Path to the data folder.",
    )
    parser.add_argument(
        "-s",
        "--step_size_list",
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.5],
        help="List of step sizes to plot. default = [0.01, 0.05, 0.1, 0.2, 0.5]",
    )
    parser.add_argument(
        "-2",
        "--two_wp",
        default=False,
        action="store_true",
        help="If only using two waveplates (fixed measurement waveplates).",
    )
    parser.add_argument("-b", "--background", default=False, action="store_true", help="Plot background count line.")

    parser.add_argument(
        "--numerics",
        default=False,
        action="store_true",
        help="If using the simulator backend.",
    )
    parser.add_argument(
        "-p",
        "--plot_filename",
        default="behavior_matrix_dependence",
        help="Filename for the plot.",
    )
    parser.add_argument(
        "--no_plot",
        default=False,
        action="store_true",
    )
    args = parser.parse_args()
    datetime_str = datetime_now_string()

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.suptitle(
        f"Best Run Behavior Matrix \n(adagrad, parameter-shift, {'two_wp' if args.two_wp else 'four_wp'}, fiber)",  # \n (Dashed) Mean Adam GD vs. (Solid) Mean Standard GD",
        fontweight="bold",
        fontsize=14,
    )
    ax.set_xlabel("Gradient Descent Step")
    ax.set_ylabel("Probability of undiscriminated photons (Cost)")

    filepath = args.filepath
    filepath_type = "/numerics/" if args.numerics else "/hardware/"
    filepath_wp = "two_wp/" if args.two_wp else "four_wp/"
    background = args.background
    plot_filename = args.plot_filename

    background_rate = []
    shot_rates = []
    c_id = 0
    for step_size in args.step_size_list:
        step_size_filepath = Path(
            filepath + "step_size_" + str(step_size).replace(".", "_") + filepath_type + filepath_wp
        )

        filenames = get_files(step_size_filepath, r"[0-9]*_optimization_.*\.json")

        for f in filenames:
            print(f"Using {f}")
        opt_dicts = [read_json(filename) for filename in filenames]

        aggregate_data = []
        aggregate_behavior_mats = []
        for j, opt_dict in enumerate(opt_dicts):
            for i in range(len(opt_dict["cost_vals"])):
                if len(aggregate_data) <= i:
                    aggregate_data += [[]]
                    aggregate_behavior_mats += [[]]

                aggregate_data[i] += [opt_dict["cost_vals"][i] + 2]
                if i == len(opt_dict["cost_vals"]) - 1:
                    aggregate_behavior_mats[i] += [[[0, 0], [0, 0]]]
                else:
                    aggregate_behavior_mats[i] += [opt_dict["behavior_mat"][i]]
            if background:
                if np.sum(opt_dict["background_rate"]) > 1000:
                    print("Background rate too high!")
                    print(f"Background rate: {np.sum(opt_dict['background_rate'])}")
                    print(f"std_opt_dict['background_rate']: {opt_dict['background_rate']}")
                    print(f"filename: {filenames[j]}")
                    input()
                else:
                    background_rate += [np.sum(opt_dict["background_rate"])]

            shot_rates += [np.sum(opt_dict["shot_rate"])]

        mean_cost_vals = []
        std_err_cost_vals = []
        min_cost_vals = []
        worst_cost_vals = []
        for i in range(len(aggregate_data)):
            mean_cost_vals += [np.mean(aggregate_data[i])]
            std_err_cost_vals += [np.std(aggregate_data[i]) / np.sqrt(len(aggregate_data[i]))]
            min_cost_vals += [np.min(aggregate_data[i])]
            worst_cost_vals += [np.max(aggregate_data[i])]

        np_mean = np.array(mean_cost_vals)
        np_std_err = np.array(std_err_cost_vals)
        best_cost_val = np.min(aggregate_data)
        aggregate_data = np.array(aggregate_data)
        aggregate_behavior_mats = np.array(aggregate_behavior_mats)
        best_final_cost_val = np.min(aggregate_data[:, -1])
        run_with_best_final_cost = np.argmin(aggregate_data[-1, :])
        behavior_mats_best_final = aggregate_behavior_mats[:, run_with_best_final_cost]
        runs_with_worst_cost = np.argmax(aggregate_data[-1, :])
        behavior_mats_worst = aggregate_behavior_mats[:, runs_with_worst_cost]

        behavior_mats_best_final = np.array(behavior_mats_best_final)
        print(behavior_mats_best_final[:, 0, 0])
        ax.semilogy(
            range(len(behavior_mats_best_final) - 1),
            1 - behavior_mats_best_final[:-1, 0, 0],
            label="1-P(0,0)",
            color=_COLORS[c_id],
        )
        c_id += 1
        c_id += 1
        c_id += 1
        ax.semilogy(
            range(len(behavior_mats_best_final) - 1),
            1 - behavior_mats_best_final[:-1, 1, 1],
            label="1-P(1,1)",
            color=_COLORS[c_id],
        )

    if background:
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
            range(len(np_mean)),
            [background_mean / np.mean(shot_rates)] * len(np_mean),
            "k--",
            label="Background (" + str(np.mean(background_rate).round(1)) + " cps)",
        )
        ax.fill_between(
            range(len(np_mean)),
            ([(background_mean - background_std_err) / np.mean(shot_rates)] * len(np_mean)),
            ([(background_mean + background_std_err) / np.mean(shot_rates)] * len(np_mean)),
            alpha=1 / 4,
            color="black",
            linestyle="--",
        )

    ax.set_yticks([1e-3, 1e-2, 1e-1, 1e-0])

    plt.grid()
    plt.legend(ncols=2)
    if not args.no_plot:
        Path.mkdir(Path(filepath + "plots/"), exist_ok=True)
        plot_filepath = args.filepath + f"/plots/{args.plot_filename}_{datetime_str}"
        plt.savefig(Path(plot_filepath))
        print(f"Plot saved at {plot_filepath}")
