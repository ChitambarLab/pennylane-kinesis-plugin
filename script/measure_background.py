import time
import pennylane_kinesis as pk
from pennylane import numpy as qnp
from tqdm import tqdm

print("Connecting to time-tagger")
buffer = pk.ttag.TTBuffer(0)
print("Connected!")

measurement_time = 60  # seconds
integration_time = 1  # seconds
n_readings = int(measurement_time / integration_time)
measurement_channels = [5, 6]
readings = qnp.zeros((n_readings, 2))

for i in tqdm(range(n_readings)):
    time.sleep(integration_time)
    channels, _ = buffer(integration_time)
    readings[i, 0] = len(channels[channels == measurement_channels[0] - 1])
    readings[i, 1] = len(channels[channels == measurement_channels[1] - 1])

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
