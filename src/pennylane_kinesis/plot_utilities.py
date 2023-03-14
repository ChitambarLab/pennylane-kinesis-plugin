from matplotlib import pyplot as plt
import numpy as np

from .file_utilities import get_files, read_json, datetime_now_string


def plot_trajectory_on_cost_contour(filename, scan_dict, opt_dict):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.grid()

    fig.suptitle(
        "Variational Quantum Optimization of\nFiber Transformation Inversion",
        fontweight="bold",
        fontsize=20,
    )

    label_size = 14

    # make contour plot of cost landscape
    CS = ax1.contour(scan_dict["x_grid"], scan_dict["y_grid"], scan_dict["z_grid"])
    ax1.clabel(CS, inline=True, fontsize=10)
    ax1.set_xlabel(r"HWP Angle (Radians)", fontsize=label_size)
    ax1.set_ylabel(r"QWP Angle (Radians)", fontsize=label_size)

    ax1.set_xticks(
        np.arange(0, np.pi / 2 + 0.1, np.pi / 6),
        [r"$0$", r"$\frac{\pi}{6}$", r"$\frac{\pi}{3}$", r"$\frac{\pi}{2}$"],
    )
    ax1.set_yticks(
        np.arange(0, np.pi + 0.1, np.pi / 6),
        [
            r"$0$",
            r"$\frac{\pi}{6}$",
            r"$\frac{\pi}{3}$",
            r"$\frac{\pi}{2}$",
            r"$\frac{2\pi}{3}$",
            r"$\frac{5\pi}{6}$",
            r"$\pi$",
        ],
    )

    # plot trajectory of optimization
    theta_settings = []
    phi_settings = []
    for settings in opt_dict["settings_history"]:
        theta_settings += [settings[0]]
        phi_settings += [settings[1]]

    ax1.plot(theta_settings, phi_settings, marker="o", label="Gradient Descent")
    ax1.plot(theta_settings[0], phi_settings[0], "rx", markersize=8, label="Starting Point")
    ax1.plot(
        theta_settings[-1],
        phi_settings[-1],
        color="lime",
        marker="v",
        linestyle="",
        markersize=8,
        label="Ending Point",
    )

    # plot log scale of optimization on right
    ax2.semilogy(
        range(len(opt_dict["cost_vals"])),
        opt_dict["cost_vals"],
        label="Gradient Descent",
        marker="o",
    )
    ax2.plot(0, opt_dict["cost_vals"][0], "rx", markersize=8, label="Starting Point")
    ax2.plot(
        len(opt_dict["cost_vals"]) - 1,
        opt_dict["cost_vals"][-1],
        color="lime",
        marker="v",
        linestyle="",
        markersize=8,
        label="Ending Point",
    )

    ax2.grid()
    ax2.set_ylabel("Detector Extinction Rate (Cost)", fontsize=label_size)
    ax2.set_xlabel("Gradient Descent Iteration", fontsize=label_size)

    plt.legend()

    plt.tight_layout()

    plt.savefig(filename)


def plot_shot_dependence(filepath, shots_list, plot_filename):
    datetime_str = datetime_now_string()

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.suptitle(
        "Number of Shots Vs.\nGradient Descent Performance", fontweight="bold", fontsize=20
    )
    ax.set_xlabel("Gradient Descent Iteration")
    ax.set_ylabel("Photon Count Extinction Rate (Cost)")

    for shots in shots_list:
        shots_filepath = filepath + "shots_" + str(shots) + "/"
        filenames = get_files(shots_filepath, r"optimization_.*\.json")
        opt_dicts = [read_json(filename) for filename in filenames]

        aggregate_data = []
        for opt_dict in opt_dicts:
            for i in range(len(opt_dict["cost_vals"])):
                if len(aggregate_data) <= i:
                    aggregate_data += [[]]

                aggregate_data[i] += [opt_dict["cost_vals"][i]]

        mean_cost_vals = []
        std_err_cost_vals = []
        for i in range(len(aggregate_data)):
            mean_cost_vals += [np.mean(aggregate_data[i])]
            std_err_cost_vals += [np.std(aggregate_data[i]) / np.sqrt(len(aggregate_data[i]))]

        np_mean = np.array(mean_cost_vals)
        np_std_err = np.array(std_err_cost_vals)

        ax.semilogy(
            range(len(mean_cost_vals)), mean_cost_vals, alpha=3 / 4, label="shots = " + str(shots)
        )
        ax.fill_between(
            range(len(mean_cost_vals)), np_mean - np_std_err, np_mean + np_std_err, alpha=1 / 4
        )

    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath + "plots/" + plot_filename + "_" + datetime_str)


def plot_step_size_dependence(filepath, step_size_list, plot_filename):
    datetime_str = datetime_now_string()

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.suptitle(
        "Step Size Vs. Gradient Descent Performance",  # \n (Dashed) Mean Adam GD vs. (Solid) Mean Standard GD",
        fontweight="bold",
        fontsize=14,
    )
    ax.set_xlabel("Gradient Descent Step")
    ax.set_ylabel("Photon Count Extinction Rate (Cost)")

    colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
    c_id = 0
    for step_size in step_size_list:
        step_size_filepath = filepath + "step_size_" + str(step_size).replace(".", "_") + "/"

        std_filenames = get_files(step_size_filepath, r"optimization_.*\.json")
        # std_filenames = get_files(step_size_filepath, r"standard_optimization_.*\.json")
        adam_filenames = get_files(step_size_filepath, r"adam_optimization_.*\.json")
        std_opt_dicts = [read_json(filename) for filename in std_filenames]
        adam_opt_dicts = [read_json(filename) for filename in adam_filenames]

        std_aggregate_data = []
        for std_opt_dict in std_opt_dicts:
            for i in range(len(std_opt_dict["cost_vals"])):
                if len(std_aggregate_data) <= i:
                    std_aggregate_data += [[]]

                std_aggregate_data[i] += [std_opt_dict["cost_vals"][i]]

        adam_aggregate_data = []
        for adam_opt_dict in adam_opt_dicts:
            for i in range(len(adam_opt_dict["cost_vals"])):
                if len(adam_aggregate_data) <= i:
                    adam_aggregate_data += [[]]

                adam_aggregate_data[i] += [adam_opt_dict["cost_vals"][i]]

        std_mean_cost_vals = []
        std_std_err_cost_vals = []
        for i in range(len(std_aggregate_data)):
            std_mean_cost_vals += [np.mean(std_aggregate_data[i])]
            std_std_err_cost_vals += [
                np.std(std_aggregate_data[i]) / np.sqrt(len(std_aggregate_data[i]))
            ]

        adam_mean_cost_vals = []
        adam_std_err_cost_vals = []
        for i in range(len(adam_aggregate_data)):
            adam_mean_cost_vals += [np.mean(adam_aggregate_data[i])]
            adam_std_err_cost_vals += [
                np.std(adam_aggregate_data[i]) / np.sqrt(len(adam_aggregate_data[i]))
            ]

        std_np_mean = np.array(std_mean_cost_vals)
        std_np_std_err = np.array(std_std_err_cost_vals)

        adam_np_mean = np.array(adam_mean_cost_vals)
        adam_np_std_err = np.array(adam_std_err_cost_vals)

        ax.semilogy(
            range(len(std_mean_cost_vals)),
            std_mean_cost_vals,
            alpha=3 / 4,
            label="step_size = " + str(step_size),
            color=colors[c_id],
        )
        ax.fill_between(
            range(len(std_mean_cost_vals)),
            std_np_mean - std_np_std_err,
            std_np_mean + std_np_std_err,
            alpha=1 / 4,
            color=colors[c_id],
        )
        # ax.semilogy(
        #     range(len(adam_mean_cost_vals)),
        #     adam_mean_cost_vals,
        #     alpha=3 / 4,
        #     color=colors[c_id],
        #     linestyle="--",
        # )
        # ax.fill_between(
        #     range(len(adam_mean_cost_vals)),
        #     adam_np_mean - adam_np_std_err,
        #     adam_np_mean + adam_np_std_err,
        #     alpha=1 / 4,
        #     color=colors[c_id],
        #     linestyle="--",
        # )

        c_id += 1

    # plt.tight_layout(pad=2)

    plt.grid(linewidth=2)
    fig.subplots_adjust(bottom=0.3)
    plt.figlegend(loc="lower center", ncols=3)

    plt.savefig(filepath + "plots/" + plot_filename + "_" + datetime_str)
