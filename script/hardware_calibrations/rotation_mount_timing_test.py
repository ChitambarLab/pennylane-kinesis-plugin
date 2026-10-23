"""Test Elliptec mount rotation times."""

import elliptec
from pennylane_kinesis.file_utilities import read_config, winapi_path, datetime_now_string
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


wp_details = read_config()

controller = elliptec.Controller("COM4", timeout=5, write_timeout=3)
devices = []
# Connect to the controller


def zero_devices(devices, offsets):
    for i, device in enumerate(devices):
        device.set_angle(offsets[i])


offsets = []
for i in range(1, 5):
    device = elliptec.Rotator(controller, address=str(i))
    devices.append(device)
    device.home()
    status = device.send_instruction(b"sv", str(50))  # Set velocity
    print("status")
    offset = wp_details[i - 1]["angle_offset"]
    offsets.append(offset)

zero_devices(devices, offsets)

time_move_0 = [[], [], [], []]
time_move_90 = [[], [], [], []]
time_get_angle = [[], [], [], []]


def try_move(device, angle):
    angle_ret = None
    while angle_ret is None:
        print(f"Trying to move {device.info['Serial No.']} to {angle}")
        angle_ret = device.set_angle(angle)
        if not angle_ret:
            print("Failed to move, trying again")
    return angle_ret


for i in range(2):
    zero_devices(devices, offsets)
    for i, device in enumerate(devices):
        controller.s.flush()
        controller.s.reset_input_buffer()
        controller.s.reset_output_buffer()
        time.sleep(5)
        # Time how long it takes to get a response
        start = time.time()
        angle = device.get_angle()
        end = time.time()
        print(i)
        time_get_angle[i].append(end - start)

        # Time how long it takes to move 0 amount
        start = time.time()
        try_move(device, angle)
        end = time.time()
        time_move_0[i].append(end - start)

        time.sleep(0.5)
        # time how long it takes to move 90 degrees
        start = time.time()
        try_move(device, angle + 90)
        end = time.time()
        time_move_90[i].append(end - start)

time_move_0 = np.array(time_move_0)
time_move_90 = np.array(time_move_90)
time_get_angle = np.array(time_get_angle)


print("Time to get angle", time_get_angle)
print("Time to move 0", time_move_0)
print("Time to move 90", time_move_90)

mean_get = np.mean(time_get_angle, axis=1)
mean_move_0 = np.mean(time_move_0, axis=1)
mean_move_90 = np.mean(time_move_90, axis=1)

print("Average time to get angle", mean_get)
print("Average time to move 0", mean_move_0)
print("Average time to move 90", mean_move_90)

print("Difference between move 0 and get angle", mean_move_0 - mean_get)
print("Difference between move 90 and get angle", mean_move_90 - mean_get)
print("Difference between move 90 and move 0", mean_move_90 - mean_move_0)

print("Percentage of time 0 to time to move 90", mean_move_0 / mean_move_90)

fig, ax = plt.subplots()
bottom = np.zeros(4)
for i in range(4):
    ax.bar(i, mean_get[i], bottom=bottom[i], label="Get Angle", color="green")
    ax.bar(i, mean_move_0[i], bottom=bottom[i] + mean_get[i], label="Move 0", color="blue")
    ax.bar(
        i,
        mean_move_90[i],
        bottom=bottom[i] + mean_get[i] + mean_move_0[i],
        label="Move 90",
        color="red",
    )
    bottom[i] = mean_get[i] + mean_move_0[i] + mean_move_90[i]


handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys())
ax.set_ylabel("Average Time (s)")
ax.set_ylim(0, 1.0)
ax.set_xlabel("Device")
ax.set_xticks([0, 1, 2, 3])
ax.set_title("Timing Test of Rotator Devices")
Path.mkdir(Path(winapi_path("./data/timing_tests/")), parents=True, exist_ok=True)
plt.savefig(winapi_path("./data/timing_tests/" + datetime_now_string() + "timing_test.png"))
