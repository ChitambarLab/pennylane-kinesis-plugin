import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import pennylane_kinesis as pk
from pathlib import Path
from datetime import datetime, timezone


def plot_movements(timing_file):
    with open(timing_file, "rb") as f:
        timing_data = pickle.load(f)

        start_times = np.array(timing_data)[:, :, 0]
        end_times = np.array(timing_data)[:, :, 1]
        set_angles = np.array(timing_data)[:, :, 2]
        start_positions = np.array(timing_data)[:, :, 3]
        end_positions = np.array(timing_data)[:, :, 4]
        h_counts_start = np.array(timing_data)[:, :, 5]
        v_counts_start = np.array(timing_data)[:, :, 6]
        h_counts_end = np.array(timing_data)[:, :, 7]
        v_counts_end = np.array(timing_data)[:, :, 8]

        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        ax[0][0].scatter(set_angles[0, :], h_counts_end[0, :], label="prep HWP")
        ax[0][1].scatter(set_angles[1, :], h_counts_end[1, :], label="prep QWP")
        ax[1][0].scatter(set_angles[2, :], h_counts_end[2, :], label="meas HWP")
        ax[1][1].scatter(set_angles[3, :], h_counts_end[3, :], label="meas QWP")
        fig.savefig("./data/two-state_discrimination/timing_files/h_counts_end.png")


def plot_timing(timing_file):
    total_time = 0
    duty_cycles = []
    timing_file = pk.winapi_path(timing_file)
    timing_data = []
    # Since we save in chunks, we need to load each file separately to get all of the chunks
    with open(timing_file, "rb") as f:
        while True:
            try:
                timing_data += pickle.load(f)
            except EOFError:
                break

        new_timing_data = [[] for i in range(4)]
        for i, wp_list in enumerate(timing_data):
            new_timing_data[i % 4] += wp_list

        timing_data = np.array(new_timing_data)

        off_times = [[] for i in range(len(timing_data))]
        min_time = np.min(np.array(timing_data)[:, 0, 0])
        max_time = np.max(np.array(timing_data))
        total_time += max_time - min_time
        print("Total time taken for" + timing_file + ":", max_time - min_time)
        timing_data = np.array(timing_data) - min_time
        run_times = [[] for _ in range(len(timing_data))]

        for i in range(len(timing_data)):
            for j in range(len(timing_data[0]) - 1):
                # Starting time of the next one minus the end time of the current
                off_times[i].append(timing_data[i][j + 1][0] - timing_data[i][j][1])
                # End time of the current minus the start time of the current
                run_times[i].append((timing_data[i][j][1] - timing_data[i][j][0]))
                if j == len(timing_data[0]) - 2:
                    run_times[i].append(timing_data[i][j + 1][1] - timing_data[i][j + 1][0])

        run_times = np.array(run_times)

        run_times_filtered = np.where(run_times > 0.5, run_times, 0)

        start_times = np.array(timing_data)[:, :, 0]
        start_times_filtered = np.where(run_times > 0.5, start_times, 0)

        if args.run_times:
            fig, ax = plt.subplots()

            # Use unfiltered data
            ax.broken_barh(
                xranges=[(start_times[0][i], run_times[0][i]) for i in range(len(run_times[0]))],
                yrange=(0, 1),
                facecolors="blue",
            )
            ax.broken_barh(
                xranges=[(start_times[1][i], run_times[1][i]) for i in range(len(run_times[0]))],
                yrange=(1, 1),
                facecolors="red",
            )
            ax.broken_barh(
                xranges=[(start_times[2][i], run_times[2][i]) for i in range(len(run_times[0]))],
                yrange=(2, 1),
                facecolors="green",
            )
            ax.broken_barh(
                xranges=[(start_times[3][i], run_times[3][i]) for i in range(len(run_times[0]))],
                yrange=(3, 1),
                facecolors="orange",
            )

            title = "Active time of each rotation stage\n" + str(timing_file.split("\\")[2].split(".")[0].split("Z")[0])
            ax.set_title(title)
            ax.set_yticks([0.5, 1.5, 2.5, 3.5], labels=["Prep HWP", "Prep QWP", "Meas HWP", "Meas QWP"])
            ax.set_xlabel("Time (s)")

            Path(timing_file.rsplit("\\", 1)[0] + "/timing_files/").mkdir(parents=True, exist_ok=True)
            fig_path = (
                timing_file.rsplit("\\", 1)[0] + "/timing_files/" + timing_file.split("\\")[-1].split(".")[0] + ".png"
            )
            print("writing to file:" + fig_path)
            plt.savefig(fig_path, bbox_inches="tight")
            plt.close()
        diff = np.sum(run_times, axis=1) - np.sum(run_times_filtered, axis=1)
        duty_cycle = np.sum(run_times, axis=1) / (np.sum(off_times, axis=1) + np.sum(run_times, axis=1))

        duty_cycles.append(duty_cycle)
        print("Duty cycle percentage:", np.sum(run_times, axis=1) / (np.sum(off_times, axis=1)))
        print("Off times:", np.sum(off_times, axis=1))

    return total_time, duty_cycles


"""
timing_data is a 2d array with shape (4, n) consisting of tuples of (start_time, end_time), denoting the running time of each rotation stage .
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Timing data reader",
        description="Reads timing data in a directory and plots the active time of each rotation stage.",
    )

    parser.add_argument(
        "-d",
        "--directory",
        default="./data/state_discrimination_hardware_step_size_dependence/",
        help="Directory to look for .pkl files in.",
    )

    parser.add_argument(
        "-r",
        "--run_times",
        default=False,
        action="store_true",
        help="If true, will produce a plot for the run times of each rotation stage.",
    )

    parser.add_argument(
        "-c",
        "--cycle",
        default=False,
        action="store_true",
        help="If true, will produce a plot for the duty cycle average.",
    )
    parser.add_argument(
        "-t",
        "--total_times",
        default=False,
        action="store_true",
        help="If true, will produce a plot for the total time taken for each step size.",
    )

    parser.add_argument(
        "--shot_dep",
        default=False,
        action="store_true",
        help="Run plotting over shot dependence data instead of step-size data.",
    )

    parser.add_argument(
        "--step_dep",
        default=False,
        action="store_true",
        help="Run plotting over step size dependence data instead of shot dependence data.",
    )

    parser.add_argument(
        "-s",
        "--step_size_list",
        nargs="+",
        default=[0.01, 0.05, 0.1, 0.2, 0.5],
        help="List of step sizes to plot. default = [0.01, 0.05, 0.1, 0.2, 0.5]",
    )

    parser.add_argument(
        "--shots_list",
        nargs="+",
        default=[100, 1000, 10000, 100000],
        help="List of shots to plot. default = [100, 1000, 10000, 100000]",
    )

    parser.add_argument("--file", "-f", default=None, help="File to plot instead of directory.")

    args = parser.parse_args()
    run_directory = args.directory + "\\"
    date_of_run = run_directory.split("\\")[-2]

    directories = []

    if args.file:
        plot_timing(args.file)
        exit()

    step_sizes = []
    shots_list = []
    for step_size in args.step_size_list:
        step_sizes.append(str(step_size).replace(".", "_"))

    for shot in args.shots_list:
        shots_list.append(str(shot))
    print("Running with step sizes:", step_sizes)

    if args.shot_dep:
        for shot in shots_list:
            directories.append(run_directory + "/shots_" + str(shot) + "/")
    elif args.step_dep:
        for step_size in step_sizes:
            directories.append(run_directory + "/step_size_" + str(step_size) + "/")
    else:
        directories.append(run_directory)

    pkl_files = []

    total_time_files = []

    for directory in directories:
        for file in os.listdir(directory):
            if "timing_data" in file:
                if file.endswith(".pkl"):
                    pkl_files.append(directory + file)
            if "runtime" in file:
                if file.endswith(".txt"):
                    total_time_files.append(directory + file)
    total_time_files = sorted(
        total_time_files,
        key=lambda x: datetime.strptime(x.split("/")[-1].split("Z")[0], "%Y-%m-%dT%H-%M-%S").timestamp(),
    )

    if args.total_times:
        total_times = []
        for total_time_file in total_time_files:
            with open(total_time_file, "r") as f:
                total_time = f.read()
                if total_time == "":
                    continue
                total_times.append(float(total_time))

        fig, ax = plt.subplots()

        if args.shot_dep:
            labels = num_shots
        else:
            labels = step_sizes

        colors = plt.get_cmap("Set1")(range(len(step_sizes)))
        ax.plot(np.arange(1, len(total_times) + 1), total_times, c="grey")
        for i in range(len(step_sizes)):
            ax.scatter(
                np.arange(1, len(total_times) + 1)[i :: len(step_sizes)],
                total_times[i :: len(step_sizes)],
                c=np.array([colors[i]]),
                label=labels[i],
            )

        if args.shot_dep:
            ax.legend(title="Number of Shots")
        else:
            ax.legend(title="Step Sizes")
        ax.set_title("Total time of each optimization (parameter-shift, adagrad)")
        ax.set_xlabel("Optimization Run Number (Sequential)")
        ax.set_ylabel("Time (s)")
        print("writing total_times to", run_directory + "total_times.png")

        plt.savefig(
            pk.winapi_path(run_directory + "/total_times_" + date_of_run + ".png"),
            bbox_inches="tight",
        )

    total_time = 0
    duty_cycles = []
    for timing_file in pkl_files:
        (time, duty_cycle) = plot_timing(timing_file)
        total_time += time
        duty_cycles.append(duty_cycle)

    print("Total time taken for all runs:", total_time)
    duty_cycles = np.array(duty_cycles)
    duty_cycles = np.squeeze(duty_cycles)
    if args.cycle:
        duty_cycle_average = np.mean(duty_cycles, axis=0)
        duty_std = np.std(duty_cycles, axis=0)
        fig, ax = plt.subplots()
        bars = ax.bar(
            [0.5, 1.5, 2.5, 3.5],
            height=duty_cycle_average,
            yerr=duty_std,
            color=["blue", "red", "green", "orange"],
        )
        ax.bar_label(bars, fmt="%.2f")
        ax.set_xticks([0.5, 1.5, 2.5, 3.5], labels=["Prep HWP", "Prep QWP", "Meas HWP", "Meas QWP"])
        ax.set_title("Duty Cycle Average \n (parameter-shift, adagrad)")
        ax.set_xlabel("Waveplate")
        ax.set_ylabel("Duty Cycle")
        path = pk.winapi_path(run_directory + "/duty_cycle" + date_of_run + ".png")
        print("writing total_times to", path)
        plt.savefig(path, bbox_inches="tight")

    print("Duty cycle average", np.mean(duty_cycles, axis=0))
print("Total total time:", total_time)
