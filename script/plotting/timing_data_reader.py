import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import pennylane_kinesis as pk
from pathlib import Path
from datetime import datetime

timing_file = './data/2024-04-16T20-31-20Ztiming_data.pkl'

directory = os.path.dirname('./data/state_discrimination_hardware_step_size_dependence/finite-diff/adagrad/')

step_sizes = ['0_01', '0_5', '0_2', '0_1', '0_05']

date_after = datetime.strptime("2024-04-25T20-16-20Z", "%Y-%m-%dT%H-%M-%SZ")
directories = []

# for step_size in step_sizes:
#     directories.append(directory + '/step_size_' + str(step_size) +"/")

# pkl_files = []

# total_time_files = []

# for directory in directories:
#     for file in os.listdir(directory):
#         if file.endswith(".txt"):
#             total_time_files.append(directory + file)
#         if file.endswith(".pkl"):
#             pkl_files.append(directory + file)

def get_date_from_timing_filename(filename):
    stripped_filename = filename.split("Z")[0]
    stripped_filename+= "Z"
    return datetime.strptime(stripped_filename, "%Y-%m-%dT%H-%M-%SZ")


'''
timing_data is a 2d array with shape (4, n) consisting of tuples of (start_time, end_time), denoting the running time of each rotation stage .
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Timing data reader",
        description="Reads timing data in a directory and plots the active time of each rotation stage.",
    )

    parser.add_argument(
        "-d",
        "--directory",
        default='./data/state_discrimination_hardware_step_size_dependence/',
        help="Directory to look for .pkl files in.",
    )
    # parser.add_argument(
    #     "-s",
    #     "--step_sizes",
    #     default=[],
    #     help="Step sizes to look for in the directory as an array of strings. e.g. ['0_01', '0_5']",
    # )

    parser.add_argument(
        "-r",
        "--run_times",
        default=False,
        action="store_true",
        help="If true, will produce a plot for the run times of each rotation stage."
    )
    
    parser.add_argument(
        "-c",
        "--cycle",
        default=False,
        action="store_true",
        help="If true, will produce a plot for the duty cycle average."
    )
    parser.add_argument(
        "-t",
        "--total_times",
        default=False,
        action="store_true",
        help="If true, will produce a plot for the total time taken for each step size."
    )

    args = parser.parse_args()

    directories = []

    for step_size in step_sizes:
        directories.append(directory + '/step_size_' + str(step_size) +"/")

    pkl_files = []

    total_time_files = []

    for step_size_directory in directories:
        for file in os.listdir(step_size_directory):
            if(("timing_data" in file or "runtime" in file) 
               and get_date_from_timing_filename(file) >= date_after):
                if file.endswith(".txt") and args.total_times:
                        total_time_files.append(step_size_directory + file)
                if file.endswith(".pkl"):
                        pkl_files.append(step_size_directory + file)
# datetime.utcnow().strftime()
    total_time_files = sorted(total_time_files, key=lambda x: datetime.strptime(x.split('/')[-1].split('Z')[0], "%Y-%m-%dT%H-%M-%S").timestamp())

    # print(total_time_files)
    # exit()
    if args.total_times:
        total_times = []
        for total_time_file in total_time_files:
            with open(total_time_file, 'r') as f:
                total_time = f.read()
                total_times.append(float(total_time))

        fig, ax = plt.subplots()

        # remove_outlier = np.array(total_times[0:5])
        # remove_outlier = np.append(remove_outlier, np.array(total_times[6:26]))
        ax.plot(np.arange(1,26), total_times)
        ax.scatter(np.arange(1,26)[::5], total_times[::5], c='red', label=step_sizes[0])
        ax.scatter(np.arange(1,26)[1::5], total_times[1::5], c='blue', label=step_sizes[1])
        ax.scatter(np.arange(1,26)[2::5], total_times[2::5], c='green', label=step_sizes[2])
        ax.scatter(np.arange(1,26)[3::5], total_times[3::5], c='purple', label=step_sizes[3])
        ax.scatter(np.arange(1,26)[4::5], total_times[4::5], c='orange', label=step_sizes[4])
        ax.legend(title="Step Sizes")
        ax.set_title("Total time of each optimization (finite-diff, adagrad)")
        ax.set_xlabel("Optimization Run Number (Sequential)")
        ax.set_ylabel("Time (s)")
        print("writing total_times to", directory+"total_times.png")
        plt.savefig(directory+"/total_times_"+ pk.datetime_now_string() + ".png", bbox_inches='tight')
    

    total_time = 0
    duty_cycles = []
    for timing_file in pkl_files:
        with open(timing_file, 'rb') as f:
            timing_data = pickle.load(f)
            off_times = [[] for i in range(len(timing_data))]
            min_time = np.min(np.array(timing_data)[:,0,0])
            max_time = np.max(np.array(timing_data))
            total_time += (max_time - min_time)
            print("Total time taken for" + timing_file + ":", max_time-min_time)
            timing_data = np.array(timing_data) - min_time
            run_times = [[] for _ in range(len(timing_data))]
            for i in range(len(timing_data)):
                for j in range(len(timing_data[0])-1):
                    off_times[i].append(timing_data[i][j+1][0]-timing_data[i][j][1])
                    run_times[i].append(timing_data[i][j][1]-timing_data[i][j][0])
                    if j == len(timing_data[0])-2:
                        run_times[i].append(timing_data[i][j+1][1]-timing_data[i][j+1][0])

            run_times = np.array(run_times)

        
            run_times_filtered = np.where(run_times > 0.5, run_times, 0)
        
            start_times = np.array(timing_data)[:,:,0]
            start_times_filtered = np.where(run_times > 0.5, start_times, 0)
        
            # # Use only the first 25 instances
            # run_times_filtered = run_times_filtered[:,0:25]
            # start_times_filtered = start_times_filtered[:,0:25]
            # print(np.sum(off_times_filtered, axis=1))
            # print(np.sum(run_times_filtered, axis=1)/np.sum(off_times_filtered, axis=1))
            if args.run_times:
                fig, ax = plt.subplots()

                # Plot filtered data
                ax.broken_barh(xranges=[(start_times_filtered[0][i], run_times_filtered[0][i]) for i in range(len(run_times_filtered[0]))], yrange=(0, 1), facecolors='blue')
                ax.broken_barh(xranges=[(start_times_filtered[1][i], run_times_filtered[1][i]) for i in range(len(run_times_filtered[0]))], yrange=(1, 1), facecolors='red')
                ax.broken_barh(xranges=[(start_times_filtered[2][i], run_times_filtered[2][i]) for i in range(len(run_times_filtered[0]))], yrange=(2, 1), facecolors='green')
                ax.broken_barh(xranges=[(start_times_filtered[3][i], run_times_filtered[3][i]) for i in range(len(run_times_filtered[0]))], yrange=(3, 1), facecolors='orange')

                # # Use unfiltered data
                # ax.broken_barh(xranges=[(start_times[0][i], run_times[0][i]) for i in range(len(run_times[0]))], yrange=(0, 1), facecolors='blue')
                # ax.broken_barh(xranges=[(start_times[1][i], run_times[1][i]) for i in range(len(run_times[0]))], yrange=(1, 1), facecolors='red')
                # ax.broken_barh(xranges=[(start_times[2][i], run_times[2][i]) for i in range(len(run_times[0]))], yrange=(2, 1), facecolors='green')
                # ax.broken_barh(xranges=[(start_times[3][i], run_times[3][i]) for i in range(len(run_times[0]))], yrange=(3, 1), facecolors='orange')

                title = 'Active time of each rotation stage\n' + str(timing_file.split('/')[3]) + "\n"+ str(timing_file.split('/')[4][0:19])
                ax.set_title(title)
                ax.set_yticks([0.5, 1.5, 2.5, 3.5], labels=['Prep HWP', 'Prep QWP', 'Meas HWP', 'Meas QWP'])
                ax.set_xlabel('Time (s)')
                Path(timing_file.rsplit('/',1)[0]+ "/timing_files/").mkdir(parents=True, exist_ok=True)
                fig_path = timing_file.rsplit('/', 1)[0] + "/timing_files/"+timing_file.split('/')[-1].split('.')[0]+'.png'
                print("writing to file:"+ fig_path)
                plt.savefig(fig_path, bbox_inches='tight')
                plt.close()
            diff = np.sum(run_times,axis=1) - np.sum(run_times_filtered, axis=1)
            duty_cycle = np.sum(run_times_filtered,axis=1)/(np.sum(off_times,axis=1) + diff)
            duty_cycles.append(duty_cycle)
            print("Duty cycle percentage:", np.sum(run_times_filtered,axis=1)/(np.sum(off_times,axis=1) + diff))

    # ax.plot(np.arange(1,26), total_times)
    # ax.scatter(np.arange(1,26), total_times, c=['red','blue','green','purple','orange']*5)

    if args.cycle:
        duty_cycles = np.array(duty_cycles)
        duty_cycle_average = np.mean(duty_cycles, axis=0)
        duty_std = np.std(duty_cycles, axis=0)
        print(duty_cycles.shape)
        print(duty_cycle_average.shape)
        fig, ax = plt.subplots()
        bars = ax.bar([0.5,1.5,2.5,3.5], height=duty_cycle_average, yerr=duty_std, color=['blue', 'red', 'green', 'orange'])
        ax.bar_label(bars, fmt='%.2f')
        ax.set_xticks([0.5, 1.5, 2.5, 3.5], labels=['Prep HWP', 'Prep QWP', 'Meas HWP', 'Meas QWP'])
        ax.set_title("Duty Cycle Average \n (finite-diff, adagrad)")
        ax.set_xlabel("Waveplate")
        ax.set_ylabel("Duty Cycle")
        path = directory+"/duty_cycle"+ pk.datetime_now_string() + ".png"
        print("writing total_times to", path)
        plt.savefig(path, bbox_inches='tight')
print("Total total time:", total_time)