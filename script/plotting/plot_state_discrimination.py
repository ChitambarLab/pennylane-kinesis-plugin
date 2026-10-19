import pennylane_kinesis as pk
from datetime import datetime, timezone
import argparse
from pathlib import Path
from pennylane_kinesis.file_utilities import (
    get_files,
    read_json,
    datetime_now_string,
    get_date_from_filename,
)
import numpy as np
import matplotlib.pyplot as plt

_COLORS = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="State Discrimination",
        description="Generates a plot showing the optimization performance for state discrimination averaged over several trials.",
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
        default="two_state_discrimination",
        help="Filename for the plot.",
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
        default="Two-State Discrimination Optimization",
        help="Title for the plot.",
    )

    args = parser.parse_args()

    if args.filepath is None:
        print("Please provide a filepath to the data folder. (-f)")
        exit()

    import matplotlib
    matplotlib.rcParams['mathtext.fontset'] = 'cm'

    filepath_type = "/numerics/" if "numerics" in args.filepath else "/hardware/"
    plt.rc("font", family="sans-serif",size=12)
    title = args.title
    if "numerics" in args.filepath:
        title += " (numerics)"


    if "two_wp" in args.filepath:
        filepath_type += "two_wp/"
    else:
        filepath_type += "four_wp/"
    
    if "parameter-shift" in args.filepath:
        title += '\nparameter-shift'
    else:
        title += "\nfinite-diff"
    title += ', adagrad, $\eta_0 = 0.2$'
    datetime_str = datetime_now_string()

    fig = plt.figure(figsize=(7,5))
    ax1 = fig.subplots()
    
    fig.suptitle(
        title,
    )

    opt_files = get_files(Path(args.filepath + filepath_type), r"optimization_.*\.json")
    for file in opt_files:
        print(f"Using {file}")

    opt_dicts = [read_json(file) for file in opt_files]

    background_rates = []
    aggregate_data = []
    shot_rates = []
    for j, opt_dict in enumerate(opt_dicts):
        for i in range(len(opt_dict["cost_vals"])):
            if len(aggregate_data) <= i:
                aggregate_data += [[]]

            aggregate_data[i] += [opt_dict["cost_vals"][i]/2 + 1]
        if args.background:
            if np.sum(opt_dict["background_rate"]) > 1000:
                print("Background rate too high!")
                print(f"Background rate: {np.sum(opt_dict['background_rate'])}")
                print(f"std_opt_dict['background_rate']: {opt_dict['background_rate']}")
                print(f"filename: {opt_files[j]}")
                input()
            else:
                background_rates += [np.sum(opt_dict["background_rate"])]

        shot_rates += [np.sum(opt_dict["shot_rate"])]

    mean_cost_vals = []
    std_err_cost = []
    std_min_cost = []

    for i in range(len(aggregate_data)):
        mean_cost_vals += [np.mean(aggregate_data[i])]
        std_err_cost += [np.std(aggregate_data[i]) / np.sqrt(len(aggregate_data[i]))]
        std_min_cost += [np.min(aggregate_data[i])]

    mean_cost_vals = np.array(mean_cost_vals)
    print(mean_cost_vals)
    std_err_cost = np.array(std_err_cost)
    std_min_cost = np.array(std_min_cost)
    best_cost_val = np.min(aggregate_data)
    print(np.array(aggregate_data).shape)
    best_cost_vals = np.min(aggregate_data, axis=1)

    print(np.mean(background_rates))
    print(np.mean(shot_rates))
    num_steps = len(mean_cost_vals)

    print("Max val: ", best_cost_val)
    print("Final Average value:", mean_cost_vals[len(aggregate_data) - 1])
    print(f"Mean background:{np.mean(background_rates)/np.mean(shot_rates)}")

    aggregate_data= np.array(aggregate_data)
    best_single_opt_cost_vals = aggregate_data[:,np.unravel_index(np.argmin(aggregate_data),aggregate_data.shape)[1]]
    print(np.unravel_index(np.argmin(aggregate_data),aggregate_data.shape))
    print(best_single_opt_cost_vals)

    ax1.semilogy(
        range(num_steps),
        mean_cost_vals,
        label=f"Average Cost (over {len(opt_dicts)} trials)",
        color=_COLORS[0],
    )


    ax1.semilogy(
        range(num_steps),
        best_single_opt_cost_vals,
        label=f"Lowest Cost Optimization",
        color=_COLORS[3],
    )

    ax1.fill_between(
        range(num_steps),
        (mean_cost_vals - std_err_cost),
        (mean_cost_vals + std_err_cost),
        alpha=1 / 4,
        color=_COLORS[0],
    )

    if args.background:
        ax1.plot(
            range(num_steps),
            [(np.mean(background_rates) / np.mean(shot_rates))]
            * num_steps,
            "k--",
            label=f"Background Limit (average {int(np.mean(background_rates))} cps)"
        )

    from matplotlib.ticker import MaxNLocator, FormatStrFormatter
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    ax1.set_ylabel("Probability of Undiscriminated States")
    ax1.set_xlabel("Optimization Iteration")

    fig.legend(loc="upper right", bbox_to_anchor=(0.975, 0.85))
    fig.tight_layout()

    plt.grid()

    if not args.no_plot:
        Path.mkdir(Path(args.filepath + "/plots/"), exist_ok=True)
        plt.savefig(Path(args.filepath + f"/plots/{args.plot_filename}_{datetime_str}"))