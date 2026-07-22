import argparse
from pennylane_kinesis import read_json, get_files
import pennylane_kinesis as pk
from pathlib import Path
import numpy as np
import pickle
import pytz
from datetime import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt

PI = 3.141592653589793
PI_2 = 1.5707963267948966
PI_4 = 0.7853981633974483
PI_3_4 = 2.356194490192345

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--folder", help="path to sweep data folder.")
    parser.add_argument("-d", "--data", help="path to optimization data folder.")
    parser.add_argument("-o", "--optimization", help="path to specific optimization to plot.")
    parser.add_argument("-m", "--measurement", help="which measurement to use.")

    args = parser.parse_args()

    sweep_folder = Path(args.folder)
    data_folder = Path(args.data)

    meas_angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]

    meas_angles_deg = [angle * 180 / np.pi for angle in meas_angles]

    filenames = [Path(str(sweep_folder) + "/" + f"0.0_{meas_angles_deg1}") for meas_angles_deg1 in meas_angles_deg]
    pkl_files = ["./" + str(filename) + ".pkl" for filename in filenames]
    json_files = [str(filename) + ".json" for filename in filenames]

    meas_settings_HWP = [str(0)]
    meas_names_HWP = [
        str(0),
        r"$\frac{\pi}{4}$",
        r"$\frac{\pi}{2}$",
        r"$\frac{3\pi}{4}$",
        r"$\pi$",
    ]
    meas_settings_QWP = [str(0), str(PI_4), str(PI_2), str(PI_3_4), str(PI)]
    meas_names_QWP = [
        str(0),
        r"$\frac{\pi}{4}$",
        r"$\frac{\pi}{2}$",
        r"$\frac{3\pi}{4}$",
        r"$\pi$",
    ]
    all_meas_names = []
    folder_names = []
    filepath_type = "/hardware/"
    two_wp = "/two_wp/"

    for i in range(len(meas_settings_HWP)):
        for j in range(len(meas_settings_QWP)):
            folder_names.append(f"meas_setting{meas_settings_HWP[i]}_ {meas_settings_QWP[j]}")
            all_meas_names.append(f"{meas_names_HWP[i]} x {meas_names_QWP[j]}")

    best_all_settings = []
    best_all_cost_vals = []
    worst_all_settings = []
    worst_all_cost_vals = []
    for k, meas_setting in enumerate(folder_names):
        print(meas_setting)
        opt_files = get_files(
            Path(pk.winapi_path(str(data_folder) + "/" + meas_setting + filepath_type + two_wp)),
            r"optimization_.*\.json",
        )

        date_path = Path(data_folder).name
        date_of_run = (
            pytz.timezone("UTC")
            .localize(datetime.strptime(date_path, "%Y-%m-%dT%H-%M-%SZ"))
            .astimezone(pytz.timezone("US/Central"))
            .strftime("%Y-%m-%d %H:%M:%S")
        )
        opt_dicts = [read_json(file) for file in opt_files]
        num_trials = len(opt_dicts)

        background_rates = []
        aggregate_data = []
        aggregate_data_settings = []
        shot_rates = []
        for j, opt_dict in enumerate(opt_dicts):
            aggregate_data.append(np.array(opt_dict["cost_vals"]) / 2 + 1)
            aggregate_data_settings.append(np.array(opt_dict["settings_history"]))
            shot_rates += [np.sum(opt_dict["shot_rate"])]

        std_mean_cost_vals = []
        std_std_err_cost_vals = []
        std_min_cost_vals = []
        for i in range(len(aggregate_data)):
            std_mean_cost_vals += [np.mean(aggregate_data[i])]
            std_std_err_cost_vals += [np.std(aggregate_data[i]) / np.sqrt(len(aggregate_data[i]))]
            std_min_cost_vals += [np.min(aggregate_data[i])]

        # print(np.array(aggregate_data).shape)
        # exit()
        aggregate_data = np.array(aggregate_data)
        aggregate_data_settings = np.array(aggregate_data_settings)
        print(aggregate_data.shape)

        best_opts = np.argmin(np.array(aggregate_data), axis=1)
        worst_opts = np.argmax(np.array(aggregate_data), axis=1)
        all_best_costs = []
        all_worst_costs = []
        for opt_num, opt in enumerate(aggregate_data):
            all_best_costs.append(opt[best_opts[opt_num]])
            all_worst_costs.append(opt[worst_opts[opt_num]])

        print("worst costs", all_worst_costs)
        all_best_costs = np.array(all_best_costs)
        all_worst_costs = np.array(all_worst_costs)

        print(all_best_costs.shape)
        print(all_worst_costs.shape)
        best_opt = np.argmin(all_best_costs)
        print(best_opt)
        worst_opt = np.argmax(all_worst_costs)

        best_settings = aggregate_data_settings[best_opt]
        best_all_settings.append(best_settings)
        worst_settings = aggregate_data_settings[worst_opt]
        worst_all_settings.append(worst_settings)

        best_all_cost_vals.append(all_best_costs[best_opt])
        worst_all_cost_vals.append(all_worst_costs[worst_opt])

        print(f"Best cost mean for {meas_setting}: {np.mean(all_best_costs)}")
        # print(best_opt)

        # all_best_costs = aggregate_data[:,best_opt]
        # print(aggregate_data.shape)
        # print(best_opt.shape)
        # print(all_best_costs.shape)
        # print(all_best_costs)
        # best_run = np.argmin(all_best_costs)
        # print(best_run)
        # all_best_settings = aggregate_data_settings[:,best_opt]
    #
    # all_worst_costs = aggregate_data[:,worst_opt]
    # print(all_worst_costs.shape)
    # worst_run = np.argmax(all_worst_costs)

    # print(worst_run)
    # all_worst_settings = aggregate_data_settings[:,worst_opt]
    # print(all_best_costs)
    # exit()
    # print(best_opt)
    # worst_opt = np.argmax(np.array(aggregate_data),axis=1)

    # print("worst", np.max(aggregate_data))
    # std_np_mean = np.array(std_mean_cost_vals)
    # std_np_std_err = np.array(std_std_err_cost_vals)

    # best_opt_cost_vals = aggregate_data[best_opt]
    # print(aggregate_data[best_opt][-1])
    # best_opt_settings = aggregate_data_settings[best_opt]

    # best_all_settings.append(best_opt_settings)
    # best_all_cost_vals.append(best_opt_cost_vals)

    # worst_all_settings.append(aggregate_data_settings[worst_opt])
    # worst_all_cost_vals.append(aggregate_data[worst_opt])
    # print(opt_files[best_opt])
    count_arrs = []
    hwp_sweep_angles = np.linspace(0, 180, 180)
    qwp_sweep_angles = np.linspace(0, 180, 180)

    for i, file in enumerate(pkl_files):
        with open(file, "rb") as f:
            counts = np.array(pickle.load(f))

        fig = plt.figure(dpi=300, figsize=(13, 5))
        ax = fig.subplots(1, 2)
        count_ax = ax[0]
        std_ax = ax[1]

        print(counts.shape)
        norm = counts[:, :, :, 0] + counts[:, :, :, 1]
        # print(norm)
        counts[:, :, :, 0] /= norm
        counts[:, :, :, 1] /= norm

        mean_counts = np.mean(counts, axis=2)[:, :, 0]

        print(f"Max for {file} is {np.max(mean_counts)}")
        print(f"Min for {file} is {np.min(mean_counts)}")
        print(f"Cost: {2 - (np.max(mean_counts) + (1 - np.min(mean_counts)))}")

        max_point = np.unravel_index(np.argmax(mean_counts), mean_counts.shape, order="C")
        min_point = np.unravel_index(np.argmin(mean_counts), mean_counts.shape, order="C")

        max_points = np.mean(counts[:, :, :, 0], axis=2) > 0.999
        min_points = np.mean(counts[:, :, :, 0], axis=2) < 0.001

        std_dev_counts = np.std(counts[:, :, :, 0], axis=2)
        mean_counts = np.mean(counts[:, :, :, 0], axis=2)
        print(std_dev_counts.shape)
        count_contour = count_ax.contourf(hwp_sweep_angles, qwp_sweep_angles, mean_counts.T, levels=50)
        std_contour = std_ax.contourf(hwp_sweep_angles, qwp_sweep_angles, std_dev_counts.T, levels=50)

        count_divider = make_axes_locatable(count_ax)
        std_divider = make_axes_locatable(std_ax)

        count_cax = count_divider.append_axes("right", size="5%", pad=0.05)
        std_cax = std_divider.append_axes("right", size="5%", pad=0.05)
        # plt.contourf(hwp_sweep_angles, qwp_sweep_angles, counts[:,:,0].T, levels=50)

        opt_settings = np.array(best_all_settings[i])
        opt_settings *= 180 / np.pi
        opt_settings[:, 0] = opt_settings[:, 0] % 180
        opt_settings[:, 1] = opt_settings[:, 1] % 180
        opt_settings[:, 2] = opt_settings[:, 2] % 180
        opt_settings[:, 3] = opt_settings[:, 3] % 180

        worst_opt_settings = np.array(worst_all_settings[i])
        worst_opt_settings *= 180 / np.pi
        worst_opt_settings[:, 0] = worst_opt_settings[:, 0] % 180
        worst_opt_settings[:, 1] = worst_opt_settings[:, 1] % 180
        worst_opt_settings[:, 2] = worst_opt_settings[:, 2] % 180
        worst_opt_settings[:, 3] = worst_opt_settings[:, 3] % 180

        # plt.text(opt_settings[-1,0], opt_settings[-1,1], f"{counts[int(opt_settings[-1,0]/2), int(opt_settings[-1,1]/2),0]}")

        # count_ax.plot(worst_opt_settings[:,0], worst_opt_settings[:,1], color="purple")
        # count_ax.scatter(worst_opt_settings[-1,0], worst_opt_settings[-1,1],label="Worst run final point (0 encoding)", color="purple")
        # count_ax.plot(worst_opt_settings[:,2], worst_opt_settings[:,3], color="red")
        # count_ax.scatter(worst_opt_settings[-1,2], worst_opt_settings[-1,3], label="Worst run final point (1 encoding)", color="red")

        # count_ax.text(worst_opt_settings[-1,0], worst_opt_settings[-1,1], f"{mean_counts[int(worst_opt_settings[-1,0]), int(worst_opt_settings[-1,1])]:.4f}")
        # count_ax.text(worst_opt_settings[-1,2], worst_opt_settings[-1,3], f"{mean_counts[int(worst_opt_settings[-1,2]), int(worst_opt_settings[-1,3])]:.4f}")

        count_ax.plot(opt_settings[:, 0], opt_settings[:, 1], color="purple")
        count_ax.scatter(
            opt_settings[-1, 0], opt_settings[-1, 1], label="Best run final point (0 encoding)", color="purple"
        )
        count_ax.plot(opt_settings[:, 2], opt_settings[:, 3], color="red")
        count_ax.scatter(
            opt_settings[-1, 2], opt_settings[-1, 3], label="Best run final point (1 encoding)", color="red"
        )

        count_ax.text(
            opt_settings[-1, 0],
            opt_settings[-1, 1],
            f"{mean_counts[int(opt_settings[-1, 0]), int(opt_settings[-1, 1])]:.4f}",
        )
        count_ax.text(
            opt_settings[-1, 2],
            opt_settings[-1, 3],
            f"{mean_counts[int(opt_settings[-1, 2]), int(opt_settings[-1, 3])]:.4f}",
        )

        # plt.plot(worst_opt_settings[:,0], worst_opt_settings[:,1],color="orange")
        # plt.scatter(worst_opt_settings[-1,0], worst_opt_settings[-1,1],label="Worst run final point (0 encoding)", color="orange")
        # plt.plot(worst_opt_settings[:,2], worst_opt_settings[:,3], color="purple")
        # plt.scatter(worst_opt_settings[-1,2], worst_opt_settings[-1,3], label="Worst run final point (1 encoding)", color="purple")

        # plt.text(opt_settings[-1,0], opt_settings[-1,1], f"{counts[int(opt_settings[-1,0]/2), int(opt_settings[-1,1]/2),0]}")
        # plt.text(opt_settings[-1,2], opt_settings[-1,3], f"{counts[int(opt_settings[-1,2]/2), int(opt_settings[-1,3]/2),0]}")

        # plt.text(worst_opt_settings[-1,0], worst_opt_settings[-1,1], f"{counts[int(worst_opt_settings[-1,0]/2), int(worst_opt_settings[-1,1]/2),0]}")
        # plt.text(worst_opt_settings[-1,2], worst_opt_settings[-1,3], f"{counts[int(worst_opt_settings[-1,2]/2), int(worst_opt_settings[-1,3]/2),0]}")
        # plt.plot()
        count_ax.scatter(hwp_sweep_angles[max_point[0]], qwp_sweep_angles[max_point[1]], label="Global maximum point")
        count_ax.text(
            hwp_sweep_angles[max_point[0]],
            qwp_sweep_angles[max_point[1]],
            f"{mean_counts[int(max_point[0]), int(max_point[1])]:.4f}",
        )
        count_ax.scatter(hwp_sweep_angles[min_point[0]], qwp_sweep_angles[min_point[1]], label="Global minimum point")
        count_ax.text(
            hwp_sweep_angles[min_point[0]],
            qwp_sweep_angles[min_point[1]],
            f"{mean_counts[int(min_point[0]), int(min_point[1])]:.4f}",
        )

        std_ax.scatter(hwp_sweep_angles[max_point[0]], qwp_sweep_angles[max_point[1]], label="_Global maximum point")
        std_ax.text(
            hwp_sweep_angles[max_point[0]],
            qwp_sweep_angles[max_point[1]],
            f"{mean_counts[int(max_point[0]), int(max_point[1])]:.4f}",
        )
        std_ax.scatter(hwp_sweep_angles[min_point[0]], qwp_sweep_angles[min_point[1]], label="_Global minimum point")
        std_ax.text(
            hwp_sweep_angles[min_point[0]],
            qwp_sweep_angles[min_point[1]],
            f"{mean_counts[int(min_point[0]), int(min_point[1])]:.4f}",
        )
        count_ax.set_title(
            f"Probability of Measurement on Transmitted Port\nMeasurement settings:{0.0},{meas_angles_deg[i]}"
        )
        std_ax.set_title(f"Standard Deviation of P(H) (5 trials)")
        std_ax.set_xlabel("Half-Waveplate angle (degrees)")
        count_ax.set_xlabel("Half-Waveplate angle (degrees)")
        count_ax.set_ylabel("Quarter-Waveplate angle (degrees)")
        # std_ax.set_ylabel("Quarter-Waveplate angle (degrees)")

        fig.colorbar(count_contour, cax=count_cax)
        fig.colorbar(std_contour, cax=std_cax)
        fig.legend()
        plt.tight_layout
        plt.savefig(str(filenames[i]) + ".png")
        plt.clf()
