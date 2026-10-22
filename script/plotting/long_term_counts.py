import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import MaxNLocator
import pickle

# TODO: move this to argparse
data_file = "./data/long_counts/counts_2026-02-11T13-39-02Z.txt"

if __name__ == "__main__":
    data = []
    with open(data_file, "rb") as file:
        data = pickle.load(file)

    data = np.array(data)
    print(data.shape)

    data_sum = np.sum(data, axis=1)
    data_norm = data / data_sum[:, None]
    fig = plt.figure(figsize=(20, 12))

    int_time = (data[1, 0] - data[0, 0]) * 1e-9
    print((data[1, 0] - data[0, 0]) * 1e-9, "integration time")

    fig.suptitle(f"Dark counts at {int_time} second integration time for 10 minutes")
    (ax1, ax2) = fig.subplots(nrows=2, sharey=True)
    xaxis = np.linspace(0, (data[:, 0][-1] - data[:, 0][0]) * 1e-9, data.shape[0])
    print(xaxis.shape)
    ax1.scatter(
        np.linspace(0, (data[:, 0][-1] - data[:, 0][0]) * 1e-9, data.shape[0]),
        data[:, 1],
        s=0.75,
        color="red",
        label="H",
    )
    ax2.scatter(
        np.linspace(0, (data[:, 0][-1] - data[:, 0][0]) * 1e-9, data.shape[0]),
        data[:, 2],
        s=0.75,
        color="blue",
        label="V",
    )
    ax1.legend()
    ax2.legend()
    ax1.set_title("Detector Counts (H)")
    ax2.set_title("Detector Counts (V)")
    ax1.set_xlabel("Time (seconds)")
    ax2.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Counts")
    ax1.grid()
    ax2.grid()
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    filename = data_file.split("/")[3].split(".")[0] + "2.png"

    print("Saving plot to", filename)
    plt.savefig("./data/long_counts/" + filename)
    plt.close()
    fig = plt.figure(figsize=(12, 8))

    fig.suptitle("Dark counts at 0.018632 integration time for 10 minutes")
    (ax1, ax2) = fig.subplots(nrows=2, sharey=True)

    counts, bins = np.histogram(data[:, 1], bins=np.arange(0, np.max(data[:, 1])))

    ax1.hist(data[:, 1], bins=np.arange(0, np.max(data[:, 1])), color="red")
    ax2.hist(data[:, 2], bins=np.arange(0, np.max(data[:, 2])), color="blue")
    fig.suptitle(f"Dark counts at {int_time}  second integration time for 10 minutes")
    ax1.set_title("H dark count distribution")
    ax2.set_title("V dark count distribution")
    ax1.grid()
    ax2.grid()
    ax1.set_ylabel("Binned count occurences")
    ax1.set_xlabel("Photon count bins")
    ax2.set_ylabel("Binned count occurences")
    ax2.set_xlabel("Photon count bins")

    plt.savefig("./data/long_counts/hist_" + filename + ".png")
    print(f"saved to ./data/long_counts/hist_{filename}.png")

    print(
        f"standard errors {np.std(data[:, 1]) / np.sqrt(len(data[:, 1]))},{np.std(data[:, 2]) / np.sqrt(len(data[:, 2]))}"
    )
    print(f"Means: {np.mean(data[:, 1])},{np.mean(data[:, 2])}")
    print(f"Range: [{np.min(data[:, 1])},{np.max(data[:, 1])}], [{np.min(data[:, 2])},{np.max(data[:, 2])}]")

    ax.plot(data_norm[:, 1], color="red", label="H")
    ax.plot(data_norm[:, 2], color="blue", label="V")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Normalized Counts")
    ax.set_title("Normalized Detector Counts over 0.5 hours")
    plt.savefig("./data/long_counts/normalized_" + filename)

    plt.cla()
    photon_energy = 2.452e-19
    data_power = data * photon_energy
    data_total_power = np.sum(data_power, axis=1)
    ax.plot(data_power[:, 1], color="red", label="H")
    ax.plot(data_power[:, 2], color="blue", label="V")
    ax.plot(data_total_power, color="purple", label="Total")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Power (W)")
    ax.set_title("Laser Power over 0.5 hours")
    plt.savefig("./data/long_counts/power_" + filename)
    print(f"Plot saved to {'./data/long_counts/power_' + filename}")
