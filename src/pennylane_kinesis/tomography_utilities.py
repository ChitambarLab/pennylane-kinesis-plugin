import numpy as np
from pennylane_kinesis import read_config, write_json, read_json, datetime_now_string
from pennylane_kinesis.swabian import Counter
import elliptec
import argparse
from pathlib import Path
import time

config_file = "./src/pennylane_kinesis/2024_04_04_timur_elliptec_config.txt"


def set_angle_raw(angle, index, rotation_mounts):
    angle_ret = None
    while angle_ret is None:
        angle_ret = rotation_mounts[index].set_angle(angle)
        if angle_ret is None:
            print(
                "Angle return is None when setting waveplate", index + 1, "to ", angle
            )
            print("Trying again...")
            time.sleep(0.01)


def set_mount_angle(angle, index, rotation_mounts, wp_details):
    if index == 0 or index == 2:
        set_angle = ((angle % 90) + float(wp_details[index]["angle_offset"])) % 360
    else:
        set_angle = ((angle % 180) + float(wp_details[index]["angle_offset"])) % 360
    angle_ret = None
    while angle_ret is None:
        angle_ret = rotation_mounts[index].set_angle(
            set_angle
        )  # This will block until the device reaches the desired angle
        if angle_ret is None:
            print(
                "Angle return is None when setting waveplate",
                index + 1,
                "to ",
                set_angle,
            )
            print("Trying again...")
            time.sleep(0.01)


def take_tomography_data(
    filepath,
    num_shots,
    integration_time,
    repetitions,
    state,
    config_file="./src/pennylane_kinesis/2025_04_04_timur_elliptec_config.txt",
    verbose=False,
):
    wp_details = read_config(config_file)
    dir = filepath + datetime_now_string() + "_process_tomography/"
    if num_shots is not None:
        dir += f"{num_shots}_shots/"
    elif integration_time is not None:
        dir += f"{integration_time}_integration_time/"
    Path(dir).mkdir(parents=True, exist_ok=True)

    # Setup mounts
    controller = elliptec.Controller("COM4", timeout=5, write_timeout=5, debug=verbose)
    mounts = []
    rotation_mounts = []
    for wp in wp_details:
        print("Connecting to waveplate at address: ", wp["address"])
        mount = elliptec.Rotator(controller, address=str(wp["address"]), debug=verbose)
        mount.home()
        rotation_mounts.append(mount)
        set_mount_angle(0, int(wp["address"]) - 1, rotation_mounts, wp_details)

    # Set meas HWP to 22.5 to balance counts.
    print(float(wp_details[2]["angle_offset"]))
    set_angle_raw(float(wp_details[2]["angle_offset"]) + 22.5, 2, rotation_mounts)

    # Setup Swabian
    counter = Counter(shots=num_shots, integration_time=integration_time)
    counter.detector_setup()

    qpt_dict = {"background_counts": []}

    # Collect background data
    for i, mount in enumerate(rotation_mounts):
        mount.home()
        set_angle_raw(wp_details[i]["background_angle"], i, rotation_mounts)

    counter.set_integration_time(1)
    counter.shots = None
    background_counts = [counter.get_counts().tolist()]
    qpt_dict["background_counts"] += background_counts
    print("Background counts: ", qpt_dict["background_counts"])

    background_counts = np.array(background_counts)
    for i in range(len(rotation_mounts)):
        set_mount_angle(0, i, rotation_mounts, wp_details)

    if num_shots:
        print(f"Running process tomography with {num_shots} as the number of shots.")
        counter.shots = num_shots
    else:
        counter.shots = None

    counter.auto_shot_rate()

    if integration_time:
        print(
            f"Running process tomography with {integration_time} as the integration time."
        )
        counter.integration_time = integration_time
        counter.auto_shot_rate()
    else:
        counter.auto_integration_time()

    qpt_dict["shot_rate"] = counter.shot_rate
    qpt_dict["num_shots"] = counter.shots
    qpt_dict["integration_time"] = counter.integration_time
    background_ratio = np.sum(background_counts) / counter.shot_rate
    qpt_dict["background_ratio"] = background_ratio

    # State angle definitions
    states_names = ["H", "V", "D", "A", "R", "L"]
    prep_states_dict = {
        "H": [0, 0],
        "V": [45, 0],
        "D": [22.5, 45],
        "A": [-22.5, 45],
        "R": [0, 135],
        "L": [0, 45],
    }
    meas_states_dict = {
        "H": [0, 0],
        "V": [45, 0],
        "D": [22.5, 0],
        "A": [67.5, 0],
        "R": [0, 135],
        "L": [0, 45],
    }
    # meas_states_dict = {"H": [45,0], "V": [0,0], "D": [67.5,0], "A": [22.5,0], "R": [0,45], "L": [0,135]}

    prep_states = [prep_states_dict[state] for state in ["H", "V", "D", "A", "R", "L"]]
    meas_states = [meas_states_dict[state] for state in ["H", "V", "D", "A", "R", "L"]]
    # Collect tomography data

    try:
        if state:
            for i in range(repetitions):
                for meas_state in states_names:
                    meas_angles = prep_states_dict[meas_state]
                    if meas_state not in qpt_dict:
                        qpt_dict[meas_state] = []
                    set_mount_angle(meas_angles[0], 0, rotation_mounts, wp_details)
                    set_mount_angle(meas_angles[1], 3, rotation_mounts, wp_details)
                    meas_counts = (counter.get_counts()).tolist()
                    qpt_dict[meas_state] += [meas_counts]
        else:
            for i in range(repetitions):
                print(f"Collecting data for repetition {i}")
                for prep_state in states_names:
                    for meas_state in states_names:
                        start_time = time.time()
                        print(f"Taking data for {prep_state}{meas_state}")
                        prep_angles = prep_states_dict[prep_state]
                        meas_angles = meas_states_dict[meas_state]
                        angles = prep_angles + meas_angles
                        measurement = prep_state + meas_state
                        if measurement not in qpt_dict:
                            qpt_dict[measurement] = []

                        for j in range(4):
                            set_mount_angle(angles[j], j, rotation_mounts, wp_details)
                        meas_counts = (counter.get_counts()).tolist()
                        qpt_dict[measurement] += [meas_counts]
                        end_time = time.time()
                        print(
                            f"Took {end_time - start_time} to collect {meas_state}{prep_state}."
                        )
                        print(
                            f"Estimated time remaining: {((repetitions - i) * 36) * (end_time - start_time)}"
                        )
    except KeyboardInterrupt:
        print("Quitting run and saving data")
        write_json(qpt_dict, dir + "qpt_data_ABORTED")
        exit()

    write_json(qpt_dict, dir + "qpt_data")
