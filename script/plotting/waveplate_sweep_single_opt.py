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
    parser.add_argument("-o", "--optimization", help="path to specific optimization to plot.")
    parser.add_argument("-m", "--measurement", help="which measurement to use.", type=int)

    args = parser.parse_args()

    sweep_folder = Path(args.folder)
    data_folder = Path(args.optimization)

    meas_angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
    meas_angles_deg = [angle * 180 / np.pi for angle in meas_angles]

    filenames = [Path(str(sweep_folder) + "/" + f"{meas_angles_deg1}_0.0") for meas_angles_deg1 in meas_angles_deg]

    filename = filenames[args.measurement]
    pkl_files = ["./" + str(filename) + ".pkl" for filename in filenames]
    pkl_file = pkl_files[args.measurement]
    # json_files = [str(filename) +".json" for filename in filenames]
    json_file = str(args.optimization)

    best_all_settings = []
    best_all_cost_vals = []
    worst_all_settings = []
    worst_all_cost_vals = []

    opt_dict = read_json(json_file)
    background_rates = []
    cost_vals = np.array(opt_dict["cost_vals"]) + 2
    settings = np.array(opt_dict["settings_history"])
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

    with open(pkl_file, "rb") as f:
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

    mean_counts_h = np.mean(counts, axis=2)[:, :, 0]
    mean_counts_v = np.mean(counts, axis=2)[:, :, 1]

    print(mean_counts_h.shape)
    print(mean_counts_v.shape)
    print(f"Max h for {pkl_file} is {np.max(mean_counts_h)}")
    print(f"Max v for {pkl_file} is {np.max(mean_counts_v)}")
    print(f"Cost: {2 - (np.max(mean_counts_h) + (np.max(mean_counts_v)))}")

    max_point = np.unravel_index(np.argmax(mean_counts_h), mean_counts_h.shape, order="C")
    min_point = np.unravel_index(np.argmax(mean_counts_v), mean_counts_v.shape, order="C")

    # max_points = np.mean(counts[:,:,:,0],axis=2) > 0.999
    # min_points = np.mean(counts[:,:,:,0],axis=2) < 0.001

    std_dev_counts_h = np.std(counts[:, :, :, 0], axis=2)
    std_dev_counts_v = np.std(counts[:, :, :, 0], axis=2)

    std_dev_counts_h /= mean_counts_h

    # mean_counts = np.mean(counts[:,:,:,0],axis=2)
    # print(std_dev_counts.shape)
    count_contour = count_ax.contourf(hwp_sweep_angles, qwp_sweep_angles, mean_counts_h.T, levels=50)
    std_contour = std_ax.contourf(hwp_sweep_angles, qwp_sweep_angles, std_dev_counts_h.T, levels=50)

    count_divider = make_axes_locatable(count_ax)
    std_divider = make_axes_locatable(std_ax)

    count_cax = count_divider.append_axes("right", size="5%", pad=0.05)
    std_cax = std_divider.append_axes("right", size="5%", pad=0.05)
    # plt.contourf(hwp_sweep_angles, qwp_sweep_angles, counts[:,:,0].T, levels=50)

    opt_settings = np.array(settings)
    print(opt_settings.shape)
    opt_settings *= 180 / np.pi
    opt_settings[:, 0] %= 180
    opt_settings[:, 1] %= 180
    opt_settings[:, 2] %= 180
    opt_settings[:, 3] %= 180

    # print(mean_counts_h[int(opt_settings[-1,0]), int(opt_settings[-1,1])])
    # worst_opt_settings = np.array(worst_all_settings[i])
    # worst_opt_settings *= 180/np.pi
    # worst_opt_settings[:,0] = worst_opt_settings[:,0]%180
    # worst_opt_settings[:,1] = worst_opt_settings[:,1]%180
    # worst_opt_settings[:,2] = worst_opt_settings[:,2]%180
    # worst_opt_settings[:,3] = worst_opt_settings[:,3]%180

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
    count_ax.scatter(opt_settings[-1, 2], opt_settings[-1, 3], label="Best run final point (1 encoding)", color="red")

    count_ax.text(
        opt_settings[-1, 0],
        opt_settings[-1, 1],
        f"{mean_counts_v[int(opt_settings[-1, 0]), int(opt_settings[-1, 1])]:.4f}$\pm${std_dev_counts_h[int(opt_settings[-1, 0]), int(opt_settings[-1, 1])]:.4f}",
    )
    count_ax.text(
        opt_settings[-1, 2],
        opt_settings[-1, 3],
        f"{mean_counts_h[int(opt_settings[-1, 2]), int(opt_settings[-1, 3])]:.4f}$\pm${std_dev_counts_v[int(opt_settings[-1, 2]), int(opt_settings[-1, 3])]:.4f}",
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
        qwp_sweep_angles[max_point[1]] + 4,
        f"{mean_counts_h[int(max_point[0]), int(max_point[1])]:.4f}$\pm${std_dev_counts_h[int(max_point[0]), int(max_point[1])]:.4f}",
    )
    count_ax.scatter(hwp_sweep_angles[min_point[0]], qwp_sweep_angles[min_point[1]], label="Global minimum point")
    count_ax.text(
        hwp_sweep_angles[min_point[0]] + 2,
        qwp_sweep_angles[min_point[1]],
        f"{mean_counts_v[int(min_point[0]), int(min_point[1])]:.4f}$\pm${std_dev_counts_v[int(min_point[0]), int(min_point[1])]:.4f}",
    )

    std_ax.scatter(hwp_sweep_angles[max_point[0]], qwp_sweep_angles[max_point[1]], label="_Global maximum point")
    std_ax.text(
        hwp_sweep_angles[max_point[0]],
        qwp_sweep_angles[max_point[1]] + 4,
        f"{mean_counts_h[int(max_point[0]), int(max_point[1])]:.4f}$\pm${std_dev_counts_h[int(max_point[0]), int(max_point[1])]:.4f}",
    )
    std_ax.scatter(hwp_sweep_angles[min_point[0]], qwp_sweep_angles[min_point[1]], label="_Global minimum point")
    std_ax.text(
        hwp_sweep_angles[min_point[0]] + 2,
        qwp_sweep_angles[min_point[1]],
        f"{mean_counts_v[int(min_point[0]), int(min_point[1])]:.4f}$\pm${std_dev_counts_v[int(min_point[0]), int(min_point[1])]:.4f}",
    )
    count_ax.set_title(
        f"Probability of Measurement on Transmitted Port\nMeasurement settings:{0.0},{meas_angles_deg[args.measurement]}\n Final cost: {cost_vals[-1]:.4f}$\pm${std_dev_counts_h[int(opt_settings[-1, 0]), int(opt_settings[-1, 1])] + std_dev_counts_v[int(opt_settings[-1, 2]), int(opt_settings[-1, 3])]:.4f}, Global min cost: {2 - (mean_counts_h[int(max_point[0]), int(max_point[1])] + (mean_counts_v[int(min_point[0]), int(min_point[1])])):.4f}$\pm${std_dev_counts_h[int(max_point[0]), int(max_point[1])] + std_dev_counts_v[int(min_point[0]), int(min_point[1])]:.4f}"
    )
    std_ax.set_title(f"Standard Deviation of P(H) (5 trials)")
    std_ax.set_xlabel("Half-Waveplate angle (degrees)")
    count_ax.set_xlabel("Half-Waveplate angle (degrees)")
    count_ax.set_ylabel("Quarter-Waveplate angle (degrees)")
    # std_ax.set_ylabel("Quarter-Waveplate angle (degrees)")

    fig.colorbar(count_contour, cax=count_cax)
    fig.colorbar(std_contour, cax=std_cax)
    fig.legend()
    plt.tight_layout()
    filename = Path(filename).stem
    plt.savefig(str(sweep_folder) + "/" + str(filename) + ".png")
    plt.clf()
