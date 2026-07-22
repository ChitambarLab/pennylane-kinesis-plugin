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
    parser.add_argument("-m", "--measurement", help="which measurement to use.", type=int)
    parser.add_argument("-o", "--optimization", help="path to specific optimization to plot.")

    args = parser.parse_args()

    sweep_folder = Path(args.folder)
    data_folder = Path(args.optimization)

    meas_angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
    meas_angles_deg = [angle*180/np.pi for angle in meas_angles]


    filenames = [Path(str(sweep_folder)+"/"+f"0.0_{meas_angles_deg1}") for meas_angles_deg1 in meas_angles_deg]

    filename = filenames[args.measurement]
    pkl_files = ["./"+str(filename)+".pkl" for filename in filenames]
    pkl_file = pkl_files[args.measurement]
    # json_files = [str(filename) +".json" for filename in filenames]
    json_file = str(args.optimization)

    best_all_settings = []
    best_all_cost_vals = []
    worst_all_settings = []
    worst_all_cost_vals = []

    opt_dict = read_json(json_file)
    background_rates = []
    cost_vals = np.array(opt_dict["cost_vals"])+2
    settings = np.array(opt_dict["settings_history"])

    count_arrs = []
    hwp_sweep_angles = np.linspace(0,180,180)
    qwp_sweep_angles = np.linspace(0,180,180)
    
    with open(pkl_file, 'rb') as f:
        counts = np.array(pickle.load(f))



    print(counts.shape)
    norm = (counts[:,:,:,0] + counts[:,:,:,1])
    # print(norm)
    counts[:,:,:,0] /= norm
    counts[:,:,:,1] /= norm

    mean_counts_h = 1- np.mean(counts,axis=2)[:,:,0]
    mean_counts_v = 1- np.mean(counts,axis=2)[:,:,1]


    # print(f"Max for {pkl_file} is {np.max(mean_counts)}")
    # print(f"Min for {pkl_file} is {np.min(mean_counts)}")
    # print(f"Cost: {2- (np.max(mean_counts) + (1-np.min(mean_counts)))}")



    # max_point = np.unravel_index(np.argmax(mean_counts), mean_counts.shape,order="C")
    # min_point = np.unravel_index(np.argmin(mean_counts), mean_counts.shape, order="C")

    max_points = np.mean(counts[:,:,:,0],axis=2) > 0.999
    min_points = np.mean(counts[:,:,:,0],axis=2) < 0.001

    qwp_angle = 130

    max_point = np.argmax(mean_counts_h[:,qwp_angle])
    min_point = np.argmax(mean_counts_v[:,qwp_angle])
    std_dev_counts_h = np.std(1- counts[:,:,:,0],axis=2)
    std_dev_counts_v = np.std(1- counts[:,:,:,1],axis=2)

    opt_settings = np.array(settings)
    print(opt_settings.shape)
    opt_settings *= 180/np.pi
    opt_settings[:,0] %= 180
    opt_settings[:,1] %= 180
    opt_settings[:,2] %= 180
    opt_settings[:,3] %= 180

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })
    fig = plt.figure(dpi=300)
    ax = fig.subplots(1,1)
    # ax.scatter(hwp_sweep_angles, mean_counts_h[:,qwp_angle], label=f"Mean H counts at QWP = ${qwp_angle}^\circ$",s=0.2)
    # ax.scatter(hwp_sweep_angles, mean_counts_v[:,qwp_angle], label=f"Mean V counts at QWP = ${qwp_angle}^\circ$",s=0.2)
    ax.errorbar(hwp_sweep_angles, mean_counts_h[:,qwp_angle], std_dev_counts_h[:,qwp_angle], None,fmt="none",ecolor="red",color="red",markersize=2, label="H counts")
    ax.errorbar(hwp_sweep_angles, mean_counts_v[:,qwp_angle], std_dev_counts_v[:,qwp_angle], None,fmt="none",ecolor="blue",color="red",markersize=2,label="V Counts")
    
    ax.scatter(max_point, 1-mean_counts_h[max_point,qwp_angle], label="Minimum H")
    ax.scatter(min_point, 1-mean_counts_v[min_point,qwp_angle], label="Minimum V")

    ax.text(max_point, 1-mean_counts_h[max_point,qwp_angle], s=f"{1-mean_counts_h[max_point,qwp_angle]:.4f}")
    ax.text(min_point, 1-mean_counts_v[min_point,qwp_angle], s=f"{1-mean_counts_v[min_point,qwp_angle]:.4f}")

    ax.set_ylabel("$1 - $ Probability of Measurement")
    ax.set_xlabel("HWP angle ($^\circ$)")
    ax.set_yscale("log")
    # ax.fill_between(
    #     hwp_sweep_angles,
    #     mean_counts[:,0] - std_dev_counts[:,0],
    #     mean_counts[:,0] + std_dev_counts[:,0],
    #     alpha=0.5,
    #     color="red"
    #     # color=_COLORS[c_id],
    # )
    ax.set_title(f"Measurement Sweep Standard Deviation (QWP = ${qwp_angle}^\circ$)")
    ax.grid()
    # count_ax = ax[0]
    # std_ax = ax[1]

    # print(std_dev_counts.shape)
    # count_contour = count_ax.contourf(hwp_sweep_angles, qwp_sweep_angles, mean_counts.T, levels=50)
    # std_contour = std_ax.contourf(hwp_sweep_angles, qwp_sweep_angles, std_dev_counts.T, levels=50)

    # count_divider = make_axes_locatable(count_ax)
    # std_divider =  make_axes_locatable(std_ax)

    # count_cax = count_divider.append_axes('right', size='5%', pad=0.05)
    # std_cax = std_divider.append_axes('right', size='5%', pad=0.05)

    # count_ax.scatter(hwp_sweep_angles[max_point[0]],qwp_sweep_angles[max_point[1]],label="Global maximum point")
    # count_ax.text(hwp_sweep_angles[max_point[0]],qwp_sweep_angles[max_point[1]], f"{mean_counts[int(max_point[0]), int(max_point[1])]:.4f}")
    # count_ax.scatter(hwp_sweep_angles[min_point[0]],qwp_sweep_angles[min_point[1]],label="Global minimum point")
    # count_ax.text(hwp_sweep_angles[min_point[0]],qwp_sweep_angles[min_point[1]], f"{mean_counts[int(min_point[0]), int(min_point[1])]:.4f}")

    # std_ax.scatter(hwp_sweep_angles[max_point[0]],qwp_sweep_angles[max_point[1]],label="_Global maximum point")
    # std_ax.text(hwp_sweep_angles[max_point[0]],qwp_sweep_angles[max_point[1]], f"{mean_counts[int(max_point[0]), int(max_point[1])]:.4f}")
    # std_ax.scatter(hwp_sweep_angles[min_point[0]],qwp_sweep_angles[min_point[1]],label="_Global minimum point")
    # std_ax.text(hwp_sweep_angles[min_point[0]],qwp_sweep_angles[min_point[1]], f"{mean_counts[int(min_point[0]), int(min_point[1])]:.4f}")
    # count_ax.set_title(f"Probability of Measurement on Transmitted Port\nMeasurement settings:{0.0},{meas_angles_deg[args.measurement]}")
    # std_ax.set_title(f"Standard Deviation of P(H) (5 trials)")
    # std_ax.set_xlabel("Half-Waveplate angle (degrees)")
    # count_ax.set_xlabel("Half-Waveplate angle (degrees)")
    # count_ax.set_ylabel("Quarter-Waveplate angle (degrees)")
    # # std_ax.set_ylabel("Quarter-Waveplate angle (degrees)")

    # fig.colorbar(count_contour, cax=count_cax)
    # fig.colorbar(std_contour, cax=std_cax)
    # fig.legend()
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(filenames[args.measurement])+f"_wp_sweep_slice_{qwp_angle:n}"+".png")
    plt.clf()



