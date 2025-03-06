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


MAXIMUM_BOUND_3_BIT = 24
QUANTUM_COST_BOUND_3_BIT = 12 + 12 / np.sqrt(3)
CLASSICAL_COST_BOUND_3_BIT = 18

QUANTUM_BOUND_3_BIT = QUANTUM_COST_BOUND_3_BIT / MAXIMUM_BOUND_3_BIT
CLASSICAL_BOUND_3_BIT = CLASSICAL_COST_BOUND_3_BIT / MAXIMUM_BOUND_3_BIT

MAXIMUM_BOUND_2_BIT = 8
QUANTUM_COST_BOUND_2_BIT = 4 + 2 * np.sqrt(2)
CLASSICAL_COST_BOUND_2_BIT = 6

QUANTUM_BOUND_2_BIT = QUANTUM_COST_BOUND_2_BIT / MAXIMUM_BOUND_2_BIT
CLASSICAL_BOUND_2_BIT = CLASSICAL_COST_BOUND_2_BIT / MAXIMUM_BOUND_2_BIT

BACKGROUND_2_BIT_MULTIPLIER = 1 / 2 + QUANTUM_COST_BOUND_2_BIT / MAXIMUM_BOUND_2_BIT
BACKGROUND_3_BIT_MULTIPLIER = 1 / 2 + QUANTUM_COST_BOUND_3_BIT / MAXIMUM_BOUND_3_BIT

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="RAC plotter",
        description="Generates a plot showing the optimization performance for RAC averaged over several trials.",
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
        default="rac_optimization",
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
        "-b", "--background", action="store_true", default=False, help="Plot background count line."
    )

    parser.add_argument(
        "-r", "--rac_3", default=False, action="store_true", help="If using the 3-bit RAC bound."
    )

    parser.add_argument(
        "--no_plot",
        default=False,
        action="store_true",
        help="If you don't want to plot the data and only want the statistics at the end.",
    )

    parser.add_argument(
        "--title",
        default="RAC Optimization (2 bits)",
        help="Title for the plot.",
    )

    args = parser.parse_args()

    if args.filepath is None:
        print("Please provide a filepath to the data folder. (-f)")
        exit()

    if args.rac_3:
        QUANTUM_BOUND = QUANTUM_BOUND_3_BIT
        QUANTUM_COST_BOUND = QUANTUM_COST_BOUND_3_BIT
        CLASSICAL_BOUND = CLASSICAL_BOUND_3_BIT
        BACKGROUND_MULTIPLIER = BACKGROUND_3_BIT_MULTIPLIER
        MAXIMUM_BOUND = MAXIMUM_BOUND_3_BIT
        args.title = "RAC Optimization (3 bits)"
    else:
        QUANTUM_BOUND = QUANTUM_BOUND_2_BIT
        QUANTUM_COST_BOUND = QUANTUM_COST_BOUND_2_BIT
        CLASSICAL_BOUND = CLASSICAL_BOUND_2_BIT
        BACKGROUND_MULTIPLIER = BACKGROUND_2_BIT_MULTIPLIER
        MAXIMUM_BOUND = MAXIMUM_BOUND_2_BIT

    filepath_type = "/numerics/" if args.numerics else "/hardware/"

    datetime_str = datetime_now_string()

    fig, (ax1, ax2) = plt.subplots(ncols=2)
    # fig, ax2 = plt.subplots()

    fig.suptitle(
        args.title,  # \n (Dashed) Mean Adam GD vs. (Solid) Mean Standard GD",
        fontweight="bold",
        fontsize=14,
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

            aggregate_data[i] += [(opt_dict["cost_vals"][i] * -1) / MAXIMUM_BOUND]
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
    std_err_cost = np.array(std_err_cost)
    std_min_cost = np.array(std_min_cost)
    best_cost_val = np.max(aggregate_data)

    num_steps = len(mean_cost_vals)

    print("Max val: ", best_cost_val)
    print("Final Average value:", mean_cost_vals[len(aggregate_data) - 1])

    ax1.plot(
        range(num_steps),
        mean_cost_vals,
    )
    ax1.plot(
        range(num_steps),
        [CLASSICAL_BOUND] * num_steps,
        label="Classical Bound",
    )

    ax1.plot(
        range(num_steps),
        [QUANTUM_BOUND] * num_steps,
        label="Quantum Bound",
    )

    ax1.fill_between(
        range(num_steps),
        (mean_cost_vals - std_err_cost),
        (mean_cost_vals + std_err_cost),
        alpha=1 / 4,
        color="C1",
    )

    ax2.semilogy(
        range(num_steps),
        [QUANTUM_BOUND - best_cost_val] * num_steps,
        alpha=3 / 4,
        label="Best Cost",
        color="blue",
    )

    ax2.semilogy(
        range(num_steps),
        [QUANTUM_BOUND] * num_steps - mean_cost_vals,
        label=f"Average Trajectory \n({len(opt_dicts)} trials)",
    )
    ax2.semilogy(
        range(num_steps),
        [QUANTUM_BOUND - CLASSICAL_BOUND] * num_steps,
        color="orange",
    )
    ax2.semilogy(
        range(num_steps),
        [(np.mean(background_rates) / np.mean(shot_rates)) * BACKGROUND_MULTIPLIER] * num_steps,
        "k--",
        label=f"Background Limit \n(Average {np.mean(background_rates):.2f} cps)",
    )
    ax2.fill_between(
        range(num_steps),
        [QUANTUM_BOUND] * num_steps - (mean_cost_vals - std_err_cost),
        [QUANTUM_BOUND] * num_steps - (mean_cost_vals + std_err_cost),
        alpha=1 / 4,
        color="C1",
    )

    ax2.set_yticks([1e-3, 1e-2, 1e-1, 1e-0])

    ax1.set_ylabel("Success Probability")
    ax1.set_xlabel("Optimization Iteration")

    ax2.set_ylabel("Distance to Quantum Bound")
    ax2.set_xlabel("Optimization Iteration")

    fig.legend(loc="lower center", bbox_to_anchor=(0.32, 0.15))
    fig.tight_layout()

    plt.grid()

    if not args.no_plot:
        Path.mkdir(Path(args.filepath + "/plots/"), exist_ok=True)
        plt.savefig(Path(args.filepath + f"/plots/{args.plot_filename}_{datetime_str}"))
