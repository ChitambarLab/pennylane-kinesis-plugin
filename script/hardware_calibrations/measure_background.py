import time
import pennylane_kinesis as pk
from pennylane_kinesis import swabian
from pennylane import numpy as qnp
import argparse
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        help="The file to load data from and calculate means.",
    )
    args = parser.parse_args()

    if args.file:
        with open(args.file, "r") as file:
            data = qnp.loadtxt(file)
            print(data)
            data = qnp.array(data)
            print(f"Mean: {qnp.mean(qnp.sum(data, axis=1))}")
            print(f"Standard deviation: {qnp.std(qnp.sum(data, axis=1))}")
            print(f"Variance: {qnp.var(data)}")

            print(f"Mean h: {qnp.mean(data[:, 0])}")
            print(f"Standard deviation h: {qnp.std(data[:, 0])}")
            print(f"Mean v: {qnp.mean(data[:, 1])}")
            print(f"Standard deviation v: {qnp.std(data[:, 1])}")

    else:
        print("Connecting to time-tagger")
        counter = swabian.Counter(shots=None, integration_time=1)
        counter.detector_setup()
        print("Connected!")

        measurement_time = 60  # seconds
        integration_time = 1  # seconds
        n_readings = int(measurement_time / integration_time)
        measurement_channels = [5, 6]
        readings = qnp.zeros((n_readings, 2))

        for i in tqdm(range(n_readings)):
            counts = counter.get_counts()
            print(counts)
            readings[i, 0] = counts[0]
            readings[i, 1] = counts[1]

        print("Completed!")
        combined_readings = readings[:, 0] + readings[:, 1]
        print(f"Readings: {readings}")
        print(f"Mean: {qnp.mean(combined_readings)}")
        print(f"Standard deviation: {qnp.std(combined_readings)}")

        # Save the data to a text file
        datetime_str = pk.datetime_now_string()
        filename = "background_" + datetime_str + ".txt"
        filepath = "./data/background/"
        qnp.savetxt(filepath + filename, readings)
        print("Data saved to " + filepath + filename)
