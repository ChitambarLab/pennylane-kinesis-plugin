import matplotlib.pyplot as plt
import numpy as np
import os


# TODO: move this to argparse
data_file = "./data/counts/counts_2024-04-24T17-22-24Z.txt"

if __name__ == "__main__":

    data = []
    with open(data_file) as file:
        for line in file:
            datum = np.array([int(line.split(",")[0]), int(line.split(",")[1])])
            data.append(datum)

    data = np.array(data)

    data_sum = np.sum(data, axis=1)
    data_norm = data / data_sum[:, None]
    fig, ax = plt.subplots()
    ax.plot(data[:, 0], color="red", label="H")
    ax.plot(data[:, 1], color="blue", label="V")
    ax.plot(np.sum(data, axis=1), color="purple", label="Total")
    ax.legend()
    ax.set_title("Detector Counts over 2 hours")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Counts")
    filename = data_file.split("/")[3].split(".")[0] + ".png"

    print("Saving plot to", filename)
    plt.savefig("./data/counts/" + filename)
    plt.cla()
    ax.plot(data_norm[:, 0], color="red", label="H")
    ax.plot(data_norm[:, 1], color="blue", label="V")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Normalized Counts")
    ax.set_title("Normalized Detector Counts over 2 hours")
    plt.savefig("./data/counts/normalized_" + filename)

    plt.cla()
    photon_energy = 2.452e-19
    data_power = data * photon_energy
    data_total_power = np.sum(data_power, axis=1)
    print(data_power)
    ax.plot(data_power[:, 0], color="red", label="H")
    ax.plot(data_power[:, 1], color="blue", label="V")
    ax.plot(data_total_power, color="purple", label="Total")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Power (W)")
    ax.set_title("Laser Power over 2 hours")
    plt.savefig("./data/counts/power_" + filename)
