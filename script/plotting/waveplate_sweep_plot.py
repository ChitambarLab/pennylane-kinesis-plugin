import argparse
from pennylane_kinesis import read_json, get_files
import pennylane_kinesis as pk
from pathlib import Path
import numpy as np
import pickle 
import pytz
from datetime import datetime

import matplotlib.pyplot as plt

PI = 3.141592653589793
PI_2 = 1.5707963267948966
PI_4 = 0.7853981633974483
PI_3_4 = 2.356194490192345

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--folder", help="path to sweep data folder.")
    parser.add_argument("-d", "--data", help="path to optimization data folder.")

    args = parser.parse_args()

    sweep_folder = Path(args.folder)
    data_folder = Path(args.data)

    meas_angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]

    meas_angles_deg = [angle*180/np.pi for angle in meas_angles]

    filenames = [Path(str(sweep_folder)+"/"+f"0.0_{meas_angles_deg1}") for meas_angles_deg1 in meas_angles_deg]
    pkl_files = ["./"+str(filename)+".pkl" for filename in filenames]
    json_files = [str(filename) +".json" for filename in filenames]

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
            folder_names.append(
                f"meas_setting{meas_settings_HWP[i]}_ {meas_settings_QWP[j]}"
            )
            all_meas_names.append(f"{meas_names_HWP[i]} x {meas_names_QWP[j]}")

    best_all_settings = []
    best_all_cost_vals = []
    worst_all_settings = []
    worst_all_cost_vals = []
    for k, meas_setting in enumerate(folder_names):
        print(meas_setting)
        opt_files = get_files(
            Path(pk.winapi_path(str(data_folder)+"/" + meas_setting + filepath_type + two_wp)),
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
            aggregate_data.append(np.array(opt_dict["cost_vals"])/2+1)
            aggregate_data_settings.append(np.array(opt_dict["settings_history"]))
            shot_rates += [np.sum(opt_dict["shot_rate"])]

        
        std_mean_cost_vals = []
        std_std_err_cost_vals = []
        std_min_cost_vals = []
        for i in range(len(aggregate_data)):
            std_mean_cost_vals += [np.mean(aggregate_data[i])]
            std_std_err_cost_vals += [
                np.std(aggregate_data[i]) / np.sqrt(len(aggregate_data[i]))
            ]
            std_min_cost_vals += [np.min(aggregate_data[i])]

        best_opt = np.argmin(np.array(aggregate_data)[:,-1])
        worst_opt = np.argmax(np.array(aggregate_data)[:,-1])
        std_np_mean = np.array(std_mean_cost_vals)
        std_np_std_err = np.array(std_std_err_cost_vals)
        
        best_opt_cost_vals = aggregate_data[best_opt]
        print(aggregate_data[best_opt][-1])
        best_opt_settings = aggregate_data_settings[best_opt]

        best_all_settings.append(best_opt_settings)
        best_all_cost_vals.append(best_opt_cost_vals)

        worst_all_settings.append(aggregate_data_settings[worst_opt])
        worst_all_cost_vals.append(aggregate_data[worst_opt])
        print(opt_files[best_opt])
    count_arrs = []
    hwp_sweep_angles = np.linspace(0,180,90)
    qwp_sweep_angles = np.linspace(0,180,90)
    for i, file in enumerate(pkl_files):
        with open(file, 'rb') as f:
            counts = np.array(pickle.load(f))

        norm = (counts[:,:,0] + counts[:,:,1])
        counts[:,:,0] /= norm
        counts[:,:,1] /= norm

        max_point = np.unravel_index(np.argmax(counts[:,:,0]), counts[:,:,0].shape,order="C")
        min_point = np.unravel_index(np.argmin(counts[:,:,0]), counts[:,:,0].shape, order="C")

        max_points = counts[:,:,0] > 0.999
        min_points = counts[:,:,0] < 0.001

        
        plt.contourf(hwp_sweep_angles, qwp_sweep_angles, counts[:,:,0].T, levels=50)

        opt_settings = np.array(best_all_settings[i])
        opt_settings *= 180/np.pi
        opt_settings[:,0] = opt_settings[:,0]%180
        opt_settings[:,1] = opt_settings[:,1]%180
        opt_settings[:,2] = opt_settings[:,2]%180
        opt_settings[:,3] = opt_settings[:,3]%180

        worst_opt_settings = np.array(worst_all_settings[i])
        worst_opt_settings *= 180/np.pi
        worst_opt_settings[:,0] = worst_opt_settings[:,0]%180
        worst_opt_settings[:,1] = worst_opt_settings[:,1]%180
        worst_opt_settings[:,2] = worst_opt_settings[:,2]%180
        worst_opt_settings[:,3] = worst_opt_settings[:,3]%180
        print(f"Best settings for {0.0},{meas_angles_deg[i]} are {opt_settings}")

        # plt.plot(opt_settings[:,0], opt_settings[:,1], color="purple")
        # plt.scatter(opt_settings[-1,0], opt_settings[-1,1],label="Best run final point (0 encoding)", color="purple")
        # plt.plot(opt_settings[:,2], opt_settings[:,3], color="red")
        # plt.scatter(opt_settings[-1,2], opt_settings[-1,3], label="Best run final point (1 encoding)", color="red")

        # plt.text(opt_settings[-1,0], opt_settings[-1,1], f"{counts[int(opt_settings[-1,0]/2), int(opt_settings[-1,1]/2),0]}")

        plt.plot(worst_opt_settings[:,0], worst_opt_settings[:,1], color="purple")
        plt.scatter(worst_opt_settings[-1,0], worst_opt_settings[-1,1],label="Worst run final point (0 encoding)", color="purple")
        plt.plot(worst_opt_settings[:,2], worst_opt_settings[:,3], color="red")
        plt.scatter(worst_opt_settings[-1,2], worst_opt_settings[-1,3], label="Worst run final point (1 encoding)", color="red")

        plt.text(worst_opt_settings[-1,0], worst_opt_settings[-1,1], f"{counts[int(worst_opt_settings[-1,0]/2), int(worst_opt_settings[-1,1]/2),0]}")


        # plt.plot(worst_opt_settings[:,0], worst_opt_settings[:,1],color="orange")
        # plt.scatter(worst_opt_settings[-1,0], worst_opt_settings[-1,1],label="Worst run final point (0 encoding)", color="orange")
        # plt.plot(worst_opt_settings[:,2], worst_opt_settings[:,3], color="purple")
        # plt.scatter(worst_opt_settings[-1,2], worst_opt_settings[-1,3], label="Worst run final point (1 encoding)", color="purple")

        # plt.text(opt_settings[-1,0], opt_settings[-1,1], f"{counts[int(opt_settings[-1,0]/2), int(opt_settings[-1,1]/2),0]}")
        # plt.text(opt_settings[-1,2], opt_settings[-1,3], f"{counts[int(opt_settings[-1,2]/2), int(opt_settings[-1,3]/2),0]}")

        # plt.text(worst_opt_settings[-1,0], worst_opt_settings[-1,1], f"{counts[int(worst_opt_settings[-1,0]/2), int(worst_opt_settings[-1,1]/2),0]}")
        # plt.text(worst_opt_settings[-1,2], worst_opt_settings[-1,3], f"{counts[int(worst_opt_settings[-1,2]/2), int(worst_opt_settings[-1,3]/2),0]}")
        # plt.plot()
        plt.scatter(hwp_sweep_angles[max_point[0]],qwp_sweep_angles[max_point[1]],label="Global maximum point")
        plt.text(hwp_sweep_angles[max_point[0]],qwp_sweep_angles[max_point[1]], f"{counts[int(max_point[0]), int(max_point[1]),0]}")
        plt.scatter(hwp_sweep_angles[min_point[0]],qwp_sweep_angles[min_point[1]],label="Global minimum point")
        # plt.text(hwp_sweep_angles[max_point[0]],qwp_sweep_angles[max_point[1]], f"{counts[int(max_point[0]), int(max_point[1]),0]}")
        plt.title(f"Probability of Measurement on Transmitted Port\nMeasurement settings:{0.0},{meas_angles_deg[i]}")
        plt.xlabel("Half-Waveplate angle (degrees)")
        plt.ylabel("Quarter-Waveplate angle (degrees)")
        plt.colorbar()
        plt.legend()
        plt.savefig(str(filenames[i])+".png")
        plt.clf()



