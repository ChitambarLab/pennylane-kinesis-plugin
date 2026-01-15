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
        "-2",
        "--two_wp",
        default=False,
        action="store_true",
        help="If using only 2 waveplates.",
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
    # matplotlib.rcParams['mathtext.fontset'] = 'custom'
    # matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    # matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    # matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
    matplotlib.rcParams['mathtext.fontset'] = 'cm'

    filepath_type = "/numerics/" if args.numerics else "/hardware/"
    plt.rc("font", family="sans-serif",size=12)
    plt.rc('text', usetex=True)
    plt.rcParams['text.latex.preamble'] = r'\centering'
    title = r'{' + args.title + r'}'
    if args.numerics:
        title += " (numerics)"

    
        # title += ""

    if args.two_wp:
        filepath_type += "two_wp/"
        # title += '\n\\selectfont{}{bQWP: $0^\circ$, bHWP: $45^\circ$}'
    else:
        filepath_type += "four_wp/"
    
    if Path(args.filepath).stem == "parameter-shift":
        title += '\n{parameter-shift'
    else:
        title += "\n{finite-diff"
    title += ', adagrad, $\eta_0 = 0.2$}'
    datetime_str = datetime_now_string()

    fig = plt.figure(figsize=(7,5))
    ax1 = fig.subplots()
    # fig, ax2 = plt.subplots()
    
    fig.suptitle(
        title,  # \n (Dashed) Mean Adam GD vs. (Solid) Mean Standard GD",
        # fontweight="bold",
        # fontsize=12,
    )

        # title += "\n(measurement WPs fixed at $(0^\circ,0^\circ)$)"

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

    # print(aggregate_data)
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

    # print(mean_cost_vals)
    print("Max val: ", best_cost_val)
    print("Final Average value:", mean_cost_vals[len(aggregate_data) - 1])
    print(f"Mean background:{np.mean(background_rates)/np.mean(shot_rates)}")

    aggregate_data= np.array(aggregate_data)
    best_single_opt_cost_vals = aggregate_data[:,np.unravel_index(np.argmin(aggregate_data),aggregate_data.shape)[1]]
    print(np.unravel_index(np.argmin(aggregate_data),aggregate_data.shape))
    print(best_single_opt_cost_vals)
    # exit()
    
    # exit()
    # print(std_err_cost)
    # print(background_rates)

    ax1.semilogy(
        range(num_steps),
        mean_cost_vals,
        label=f"Average Cost (over {len(opt_dicts)} trials)",
        color=_COLORS[0],
    )

    # ax1.semilogy(
    #     range(num_steps),
    #     best_cost_vals,
    #     label=f"Best Cost at Step (out of {len(opt_dicts)} trials)",
    #     color=_COLORS[1],
    # )

    ax1.semilogy(
        range(num_steps),
        best_single_opt_cost_vals,
        label=f"Lowest Cost Optimization",
        color=_COLORS[3],
    )
    # ax1.plot(
    #     range(num_steps),
    #     [CLASSICAL_BOUND] * num_steps,
    #     label="Classical Bound",
    # )

    # ax1.plot(
    #     range(num_steps),
    #     [QUANTUM_BOUND] * num_steps,
    #     label="Quantum Bound",
    # )

    ax1.fill_between(
        range(num_steps),
        (mean_cost_vals - std_err_cost),
        (mean_cost_vals + std_err_cost),
        alpha=1 / 4,
        color=_COLORS[0],
        # label="Standard Error",
    )

    ax1.plot(
        range(num_steps),
        [(np.mean(background_rates) / np.mean(shot_rates))]
        * num_steps,
        "k--",
        label=f"Background Limit (average {int(np.mean(background_rates))} cps)"
    )

    from matplotlib.ticker import MaxNLocator, FormatStrFormatter
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
    ax1.set_xlim(0,60)
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    # ax1.set_yticks([0, 5, 10, 15, 20] if args.rac_3 else [0, 1, 2, 3, 4, 5, 6,7])

    # print(np.mean(background_rates) / np.mean(shot_rates) * BACKGROUND_MULTIPLIER)
    # print(np.mean(background_rates)/np.mean(shot_rates)*(1+np.sqrt(2)/4))
    # print(np.mean(background_rates)/np.mean(shot_rates)*(np.sqrt(2)/4))

    # ax2.semilogy(
    #     range(num_steps),
    #     [QUANTUM_BOUND - best_cost_val] * num_steps,
    #     alpha=3 / 4,
    #     label="Best Cost",
    #     color="blue",
    # )

    # ax2.semilogy(
    #     range(num_steps),
    #     [QUANTUM_BOUND] * num_steps - mean_cost_vals,
    #     label=f"Average Trajectory \n({len(opt_dicts)} trials)",
    # )
    # ax2.semilogy(
    #     range(num_steps),
    #     [QUANTUM_BOUND - CLASSICAL_BOUND] * num_steps,
    #     color="orange",
    #     # label="Classical Limit",
    # )
    # ax2.semilogy(
    #     range(num_steps),
    #     [(np.mean(background_rates)/np.mean(shot_rates))]*num_steps,
    #     "k--",
    #     label=f"Background Limit \n(Average {np.mean(background_rates):.2f} cps)",
    # )
    # ax2.semilogy(
    #     range(num_steps),
    #     [(np.mean(background_rates) / np.mean(shot_rates)) * BACKGROUND_MULTIPLIER] * num_steps,
    #     "k--",
    #     label=f"Background Limit \n(Average {np.mean(background_rates):.2f} cps)",
    # )
    # ax2.fill_between(
    #     range(num_steps),
    #     [QUANTUM_BOUND] * num_steps - (mean_cost_vals - std_err_cost),
    #     [QUANTUM_BOUND] * num_steps - (mean_cost_vals + std_err_cost),
    #     alpha=1 / 4,
    #     label="Standard Error",
    #     color="C1",
    # )

    # ax2.set_yticks([1e-3, 1e-2, 1e-1, 1e-0])

    ax1.set_ylabel("Probability of Undiscriminated States")
    ax1.set_xlabel("Optimization Iteration")

    # ax2.set_ylabel("Distance to Quantum Bound")
    # ax2.set_xlabel("Optimization Iteration")

    fig.legend(loc="upper right", bbox_to_anchor=(0.975, 0.85))
    fig.tight_layout()

    plt.grid()
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if not args.no_plot:
        Path.mkdir(Path(args.filepath + "/plots/"), exist_ok=True)
        plt.savefig(Path(args.filepath + f"/plots/{args.plot_filename}_{datetime_str}"))

    # ax.semilogy(
    #     range(len(mean_cost_vals)),
    #     best_cost_val*np.ones(len(mean_cost_vals)),
    #     alpha=3 / 4,
    #     label="Best Cost",
    # )
    # ax.semilogy(
    #     range(len(std_mean_cost_vals)),
    #     std_mean_cost_vals,
    #     alpha=3 / 4,
    #     linestyle="--",
    # )
    #     # ax.plot(
    #     #     range(len(std_mean_cost_vals)),
    #     #     std_mean_cost_vals,
    #     #     alpha=3 / 4,
    #     #     label="step_size = " + str(step_size),
    #     #     color=_COLORS[c_id],
    #     #     linestyle="--",
    #     # )
    # ax.fill_between(
    #     range(len(std_mean_cost_vals)),
    #     std_np_mean - std_np_std_err,
    #     std_np_mean + std_np_std_err,
    #     alpha=1 / 4,
    #     color=_COLORS[c_id],
    # )
