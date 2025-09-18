import numpy as np
import elliptec
from pennylane_kinesis import read_config, datetime_now_string, write_json
from pennylane_kinesis.tomography_utilities import set_angle_raw, set_mount_angle
from pennylane_kinesis.swabian import Counter
from pathlib import Path
import argparse

config_file = "./src/pennylane_kinesis/2024_09_18_timur_elliptec_config.txt"

NUM_REPS = 50
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure the imbalance of the interferometer.")
    parser.add_argument(
        "-f", "--filepath", default="./data/detector_balance/", help="Path to the data folder."
    )
    parser.add_argument("-v", "--verbose", default=False)
    args = parser.parse_args()

    verbose = args.verbose
    num_shots = None
    integration_time = 1

    wp_details = read_config(config_file)
    dir = args.filepath + datetime_now_string() + "_imbalance/"
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

    # time.sleep(2)
    # Setup Swabian
    counter = Counter(shots=num_shots, integration_time=integration_time)
    counter.detector_setup()
    # counter.auto_shot_rate()
    # counter.auto_integration_time()

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

    for i in range(len(rotation_mounts)):
        set_mount_angle(0, i, rotation_mounts, wp_details)

    set_mount_angle(prep_states_dict["R"][0], 2, rotation_mounts, wp_details)
    set_mount_angle(prep_states_dict["R"][1], 3, rotation_mounts, wp_details)

    qpt_dict["D_counts"] = []
    for i in range(NUM_REPS):
        D_counts = counter.get_counts().tolist()
        qpt_dict["D_counts"] += [D_counts]

    set_mount_angle(prep_states_dict["L"][0], 2, rotation_mounts, wp_details)
    set_mount_angle(prep_states_dict["L"][1], 3, rotation_mounts, wp_details)

    qpt_dict["A_counts"] = []
    for i in range(NUM_REPS):
        A_counts = counter.get_counts().tolist()
        qpt_dict["A_counts"] += [A_counts]

    write_json(qpt_dict, dir + "imbalance.json")
