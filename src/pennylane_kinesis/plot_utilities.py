from matplotlib import pyplot as plt
import numpy as np

from .file_utilities import get_files, read_json, datetime_now_string


def plot_trajectory_on_cost_contour(filename, scan_dict, opt_dict, background_shots=None):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.grid()

    fig.suptitle(
        "Variational Quantum Optimization of\nFiber Transformation Inversion",
        fontweight="bold",
        fontsize=20,
    )

    label_size = 14

    scan_shots = scan_dict["shots"]
    opt_shots = opt_dict["shots"]

    # make contour plot of cost landscape
    CS = ax1.contour(np.array(scan_dict["x_grid"])*180/np.pi, np.array(scan_dict["y_grid"])*180/np.pi, np.array(scan_dict["z_grid"])*scan_shots/1000, linewidth=4)
    
    
    ax1.clabel(CS, inline=True, fontsize=10)

    cbar = fig.colorbar(CS)
    cbar.ax.set_ylabel('Photon Counts (x1000)')
    
    ax1.set_xlabel(r"HWP Angle (Degrees)", fontsize=label_size)
    ax1.set_ylabel(r"QWP Angle (Degrees)", fontsize=label_size)
    ax1.set_title("Photon Counts Vs.\nWave Plate Settings", fontsize=label_size)

    ax1.set_xticks([0,15,30,45,60,75,90])
    ax1.set_yticks([0,30,60,90,120,150,180])

    # plot trajectory of optimization
    theta_settings = []
    phi_settings = []
    for settings in opt_dict["settings_history"]:
        theta_settings += [settings[0]]
        phi_settings += [settings[1]]

    ax1.plot(np.array(theta_settings)*180/np.pi, np.array(phi_settings)*180/np.pi, marker="o", label="Gradient Descent")

    # plot log scale of optimization on right
    ax2.semilogy(
        range(len(opt_dict["cost_vals"])),
        np.array(opt_dict["cost_vals"])*opt_shots,
        label="Gradient Descent",
        marker="o",
    )

    ax1.plot(theta_settings[0]*180/np.pi, phi_settings[0]*180/np.pi, "rx", markersize=8)
    ax1.plot(
        theta_settings[-1]*180/np.pi,
        phi_settings[-1]*180/np.pi,
        color="lime",
        marker="v",
        linestyle="",
        markersize=8,
    )
    ax2.plot(0, opt_dict["cost_vals"][0]*opt_shots, "rx", markersize=8, label="Starting Point")
    ax2.plot(
        len(opt_dict["cost_vals"]) - 1,
        opt_dict["cost_vals"][-1]*opt_shots,
        color="lime",
        marker="v",
        linestyle="",
        markersize=8,
        label="Ending Point",
    )

    if background_shots:
        num_iter = len(opt_dict["cost_vals"])
        ax2.plot(range(num_iter), [background_shots] * num_iter, label="Background", linestyle="--")

    ax2.grid()
    ax2.set_ylabel("Photon Counts (Cost)", fontsize=label_size)
    ax2.set_xlabel("Iteration", fontsize=label_size)
    ax2.set_title("Photon Counts Vs.\nGradient Descent Iteration", fontsize=label_size)

    plt.legend()

    plt.tight_layout()

    plt.savefig(filename)


def plot_shot_dependence_comparison(
    hardware_filepath, numerics_filepath, shots_list, plot_filename
):
    datetime_str = datetime_now_string()

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.suptitle(
        "Number of Shots Vs.\nGradient Descent Performance", fontweight="bold", fontsize=20
    )
    ax.set_xlabel("Gradient Descent Iteration")
    ax.set_ylabel("Photon Count Extinction Rate (Cost)")

    for shots in shots_list:
        hardware_shots_filepath = hardware_filepath + "shots_" + str(shots) + "/"
        numerics_shots_filepath = numerics_filepath + "shots_" + str(shots) + "/"

        hardware_filenames = get_files(hardware_shots_filepath, r"optimization_.*\.json")
        numeric_filenames = get_files(
            numerics_shots_filepath, r"background_noise_optimization_.*\.json"
        )

        hardware_opt_dicts = [read_json(filename) for filename in hardware_filenames]
        numeric_opt_dicts = [read_json(filename) for filename in numeric_filenames]

        hardware_aggregate_data = []
        for opt_dict in hardware_opt_dicts:
            for i in range(len(opt_dict["cost_vals"])):
                if len(hardware_aggregate_data) <= i:
                    hardware_aggregate_data += [[]]

                hardware_aggregate_data[i] += [opt_dict["cost_vals"][i]]

        numerics_aggregate_data = []
        for opt_dict in numeric_opt_dicts:
            for i in range(len(opt_dict["cost_vals"])):
                if len(numerics_aggregate_data) <= i:
                    numerics_aggregate_data += [[]]

                numerics_aggregate_data[i] += [opt_dict["cost_vals"][i]]

        hardware_mean_cost_vals = []
        hardware_std_err_cost_vals = []
        for i in range(len(hardware_aggregate_data)):
            hardware_mean_cost_vals += [np.mean(hardware_aggregate_data[i])]
            hardware_std_err_cost_vals += [
                np.std(hardware_aggregate_data[i]) / np.sqrt(len(hardware_aggregate_data[i]))
            ]

        theory_np_mean = np.array(hardware_mean_cost_vals)
        theory_np_std_err = np.array(hardware_std_err_cost_vals)

        numerics_mean_cost_vals = []
        numerics_std_err_cost_vals = []
        for i in range(len(numerics_aggregate_data)):
            numerics_mean_cost_vals += [np.mean(numerics_aggregate_data[i])]
            numerics_std_err_cost_vals += [
                np.std(numerics_aggregate_data[i]) / np.sqrt(len(numerics_aggregate_data[i]))
            ]

        numerics_np_mean = np.array(numerics_mean_cost_vals)
        numerics_np_std_err = np.array(numerics_std_err_cost_vals)

        ax.semilogy(
            range(len(hardware_mean_cost_vals)),
            hardware_mean_cost_vals,
            alpha=3 / 4,
            label="Hardware, shots = " + str(shots),
        )
        ax.fill_between(
            range(len(hardware_mean_cost_vals)),
            theory_np_mean - theory_np_std_err,
            theory_np_mean + theory_np_std_err,
            alpha=1 / 4,
        )

        ax.semilogy(
            range(len(numerics_mean_cost_vals)),
            numerics_mean_cost_vals,
            alpha=3 / 4,
            label="Numerics, shots = " + str(shots),
        )
        ax.fill_between(
            range(len(numerics_mean_cost_vals)),
            numerics_np_mean - numerics_np_std_err,
            numerics_np_mean + numerics_np_std_err,
            alpha=1 / 4,
        )

    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_filename + "_" + datetime_str)


def plot_shot_dependence(filepath, shots_list, plot_filename, background=None, shot_rate=None, add_background=False):
    datetime_str = datetime_now_string()

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.suptitle(
        "Number of Shots Vs.\nOptimization Performance", fontweight="bold", fontsize=20
    )
    ax.set_xlabel("Gradient Descent Iteration")
    ax.set_ylabel("Percent of Incident Photons Detected (Cost)")


    for shots in shots_list:
        shots_filepath = filepath + "shots_" + str(shots) + "/"
        filenames = get_files(shots_filepath, r"optimization_.*\.json")
        opt_dicts = [read_json(filename) for filename in filenames]

        aggregate_data = []
        for opt_dict in opt_dicts:
            for i in range(len(opt_dict["cost_vals"])):
                if len(aggregate_data) <= i:
                    aggregate_data += [[]]

                if add_background:
                    dark_count_rate = background /shot_rate
                    corrected_probs = opt_dict["cost_vals"][i] + dark_count_rate
                    aggregate_data[i] += [corrected_probs]

                else:
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

    
    if background:
        ax.plot(range(len(mean_cost_vals)), [background/shot_rate] * len(mean_cost_vals), "k--", label="Background (" + str(background) + " cps)")


    ax.set_yticks([1e-2,1e-1,1e-0])
    plt.grid()
    plt.legend(ncols=2)
    plt.tight_layout()
    plt.savefig(filepath + "plots/" + plot_filename + "_" + datetime_str)


def plot_step_size_dependence(filepath, step_size_list, plot_filename, add_background=False, background=None, shot_rate=None, plot_adam=False):
    datetime_str = datetime_now_string()

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.suptitle(
        "Step Size Vs. Gradient Descent Performance",  # \n (Dashed) Mean Adam GD vs. (Solid) Mean Standard GD",
        fontweight="bold",
        fontsize=14,
    )
    ax.set_xlabel("Gradient Descent Step")
    ax.set_ylabel("Percent Incident Photons Detected (Cost)")

    colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
    c_id = 0
    for step_size in step_size_list:
        step_size_filepath = filepath + "step_size_" + str(step_size).replace(".", "_") + "/"

        # data naming is inconsistent between numerics and hardware.
        std_filenames = get_files(step_size_filepath, r"standard_optimization_.*\.json")
        if len(std_filenames) == 0:
            std_filenames = get_files(step_size_filepath, r"optimization_.*\.json")

        adam_filenames = get_files(step_size_filepath, r"adam_optimization_.*\.json")
        std_opt_dicts = [read_json(filename) for filename in std_filenames]
        adam_opt_dicts = [read_json(filename) for filename in adam_filenames]

        std_aggregate_data = []
        for std_opt_dict in std_opt_dicts:
            for i in range(len(std_opt_dict["cost_vals"])):
                if len(std_aggregate_data) <= i:
                    std_aggregate_data += [[]]

                if add_background:
                    corrected_val = std_opt_dict["cost_vals"][i] + background/shot_rate
                    std_aggregate_data[i] += [corrected_val]
                else:
                    std_aggregate_data[i] += [std_opt_dict["cost_vals"][i]]

        adam_aggregate_data = []
        for adam_opt_dict in adam_opt_dicts:
            for i in range(len(adam_opt_dict["cost_vals"])):
                if len(adam_aggregate_data) <= i:
                    adam_aggregate_data += [[]]

                if add_background:
                    corrected_val = adam_opt_dict["cost_vals"][i] + background/shot_rate
                    adam_aggregate_data[i] += [corrected_val]
                else:
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


        if plot_adam:
            adam_np_mean = np.array(adam_mean_cost_vals)
            adam_np_std_err = np.array(adam_std_err_cost_vals)
            ax.semilogy(
                range(len(adam_mean_cost_vals)),
                adam_mean_cost_vals,
                alpha=3 / 4,
                color=colors[c_id],
                linestyle="--",
            )
            ax.fill_between(
                range(len(adam_mean_cost_vals)),
                adam_np_mean - adam_np_std_err,
                adam_np_mean + adam_np_std_err,
                alpha=1 / 4,
                color=colors[c_id],
                linestyle="--",
            )

        c_id += 1

    if background:
        ax.plot(range(len(std_np_mean)), [background/shot_rate] * len(std_np_mean), "k--", label="Background (" + str(background) + " cps)")

    # plt.tight_layout(pad=2)
    ax.set_yticks([1e-2,1e-1,1e-0])

    plt.grid()
    # fig.subplots_adjust(bottom=0.3)
    plt.legend(ncols=2)

    plt.savefig(filepath + "plots/" + plot_filename + "_" + datetime_str)
