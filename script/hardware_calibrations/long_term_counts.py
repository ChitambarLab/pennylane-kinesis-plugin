import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from pennylane_kinesis import swabian, datetime_now_string
import time

CACHE_SIZE = 50000


def plot_all_counts(filepath):
    all_counts = []
    try:
        with open(filepath, "rb") as f:
            while True:
                counts = pkl.load(f)
                all_counts += counts.tolist()
    except EOFError:
        pass

    counts = np.array(all_counts)[:, 1:]
    timestamps = np.array(all_counts)[:, 0]
    timestamps -= np.ones_like(timestamps) * timestamps[0]
    time_xaxis = timestamps * 1e-9  # convert to seconds

    total_time = time_xaxis[-1]
    all_counts = counts
    data_sum = np.sum(all_counts, axis=1)
    data_norm = all_counts / data_sum[:, None]
    fig, axes = plt.subplots(2, 1, figsize=(5, 7), dpi=400)
    counts_ax = axes[0]
    norm_ax = axes[1]
    counts_ax.plot(time_xaxis, all_counts[:, 0], color="red", label="H")
    counts_ax.plot(time_xaxis, all_counts[:, 1], color="blue", label="V")
    counts_ax.plot(time_xaxis, np.sum(all_counts, axis=1), color="purple", label="Total")
    counts_ax.legend()
    counts_ax.set_title(f"Detector Counts over {total_time / 3600:.3f} hours")
    counts_ax.set_xlabel("Time (seconds)")
    counts_ax.set_ylabel("Counts")

    norm_ax.plot(time_xaxis, data_norm[:, 0], color="red", label="H")
    norm_ax.plot(time_xaxis, data_norm[:, 1], color="blue", label="V")
    norm_ax.set_xlabel("Time (seconds)")
    norm_ax.set_ylabel("Normalized Counts")
    norm_ax.set_title(f"Normalized Detector Counts over {total_time / 3600:.3f} hours")

    photon_energy = 2.452e-19
    data_power = all_counts * photon_energy
    data_total_power = np.sum(data_power, axis=1)

    filename = Path(filepath).stem
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./data/long_counts/{filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Take counts over a period of time",
        description="Take counts over a period of time using the Swabian Time Tagger.",
    )

    parser.add_argument(
        "-t",
        "--time",
        default=None,
        type=int,
        help="Time to take counts for in seconds.",
    )

    parser.add_argument(
        "-s",
        "--shots",
        default=None,
        type=int,
        help="Number of shots to take. Default: None. (will not truncate counts)",
    )

    parser.add_argument(
        "-i",
        "--integration",
        default=1,
        type=float,
        help="Integration time in seconds.",
    )

    parser.add_argument(
        "-a",
        "--auto",
        default=False,
        action="store_true",
        help="Automatically set the integration time based on calculated shot rate.",
    )

    parser.add_argument("--num_trials", default=None, type=int, help="Number of trials to take counts for.")

    parser.add_argument(
        "--plot", default=None, type=str, help="If only wanting to plot data, pass in the path to the data file"
    )

    total_counts = []
    dir = "./data/long_counts/"
    Path(dir).mkdir(parents=True, exist_ok=True)
    args = parser.parse_args()

    if args.plot:
        plot_all_counts(args.plot)
        exit()

    counter = swabian.Counter(shots=args.shots, integration_time=args.integration)
    counter.detector_setup()
    if args.auto:
        counter.auto_shot_rate()
        counter.auto_integration_time()

    counter.shots = None
    print("Starting to take counts...")
    start_time = time.time()
    date_time = datetime_now_string()
    filepath = f"{dir}counts_{date_time}.txt"
    total_counts = []

    try:
        while True:
            if args.time and time.time() - start_time > args.time:
                break
            if args.num_trials and len(total_counts) >= args.num_trials:
                break

            counts = counter.get_counts()
            total_counts.append([time.time_ns()] + counts.tolist())

            if len(total_counts) > CACHE_SIZE:
                with open(filepath, "ab") as f:
                    pkl.dump(np.array(total_counts), f)
                total_counts = []
        with open(filepath, "ab") as f:
            pkl.dump(np.array(total_counts), f)
        plot_all_counts(filepath)

    except KeyboardInterrupt:
        print("Saving partial counts")
        with open(filepath, "ab") as f:
            pkl.dump(np.array(total_counts), f)

        plot_all_counts(filepath)
