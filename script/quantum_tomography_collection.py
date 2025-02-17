import numpy as np
from pennylane_kinesis import read_config, write_json, read_json, datetime_now_string
from pennylane_kinesis.swabian import Counter
import elliptec
import argparse
from pathlib import Path
import time

config_file = "./src/pennylane_kinesis/2024_09_18_timur_elliptec_config.txt"


def set_angle_raw(angle, index):
    print("setting angle", angle, "for index", index)
    angle_ret = None
    while angle_ret is None:
        angle_ret = rotation_mounts[index].set_angle(angle)
        if angle_ret is None:
            print("Angle return is None when setting waveplate", index + 1, "to ", angle)
            print("Trying again...")
            time.sleep(0.01)


def set_mount_angle(angle, index):
    print("setting angle", angle, "for index", index)
    if index == 0 or index == 2:
        set_angle = (angle % 90) + float(wp_details[index]["angle_offset"])
    else:
        set_angle = (angle % 180) + float(wp_details[index]["angle_offset"])
    angle_ret = None
    while angle_ret is None:
        angle_ret = rotation_mounts[index].set_angle(
            set_angle
        )  # This will block until the device reaches the desired angle
        if angle_ret is None:
            print("Angle return is None when setting waveplate", index + 1, "to ", set_angle)
            print("Trying again...")
            time.sleep(0.01)
    print(angle_ret)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Process tomography collection",
        description="Collect counts for process tomography",
    )

    parser.add_argument(
        "-f",
        "--filepath",
        default="./data/process_tomography/",
        help="Path to the data folder.",
    )

    parser.add_argument(
        "-n",
        "--num_shots",
        default=10000,
        type=int,
        help="Number of shots to take.",
    )

    parser.add_argument(
        "-r",
        "--repetitions",
        default=1,
        type=int,
        help="Number of times to repeat the measurement (to be averaged).",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
    )
    parser.add_argument(
        "--state", default=False, action="store_true", help="If running state tomography instead."
    )

    args = parser.parse_args()

    wp_details = read_config(config_file)
    dir = args.filepath + datetime_now_string() + "_process_tomography/"
    Path(dir).mkdir(parents=True, exist_ok=True)

    # Setup mounts
    controller = elliptec.Controller("COM4", timeout=5, write_timeout=5, debug=args.verbose)
    mounts = []
    rotation_mounts = []
    for wp in wp_details:
        print("Connecting to waveplate at address: ", wp["address"])
        mount = elliptec.Rotator(controller, address=str(wp["address"]), debug=args.verbose)
        mount.home()
        rotation_mounts.append(mount)
        set_mount_angle(0, int(wp["address"]) - 1)

    # Set meas HWP to 22.5 to balance counts.
    print(float(wp_details[2]["angle_offset"]))
    set_angle_raw(float(wp_details[2]["angle_offset"]) + 22.5, 2)

    # time.sleep(2)
    # Setup Swabian
    counter = Counter(shots=args.num_shots)
    counter.detector_setup()
    # counter.auto_shot_rate()
    # counter.auto_integration_time()

    qpt_dict = {"background_counts": []}

    # Collect background data
    for i, mount in enumerate(rotation_mounts):
        mount.home()
        set_angle_raw(wp_details[i]["background_angle"], i)

    counter.set_integration_time(1)
    counter.shots = None
    background_counts = [counter.get_counts().tolist()]
    qpt_dict["background_counts"] += background_counts
    print("Background counts: ", qpt_dict["background_counts"])

    background_counts = np.array(background_counts)
    for i in range(len(rotation_mounts)):
        set_mount_angle(0, i)
    counter.shots = args.num_shots
    counter.auto_shot_rate()
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
        if args.state:
            for i in range(args.repetitions):
                for meas_state in states_names:
                    meas_angles = prep_states_dict[meas_state]
                    if meas_state not in qpt_dict:
                        qpt_dict[meas_state] = []
                    set_mount_angle(meas_angles[0], 0)
                    set_mount_angle(meas_angles[1], 3)
                    meas_counts = (counter.get_counts() / args.num_shots).tolist()
                    qpt_dict[meas_state] += [meas_counts]
        else:
            for i in range(args.repetitions):
                for prep_state in states_names:
                    for meas_state in states_names:
                        print("Taking measurement for: ", prep_state, meas_state)
                        prep_angles = prep_states_dict[prep_state]
                        meas_angles = meas_states_dict[meas_state]
                        angles = prep_angles + meas_angles
                        measurement = prep_state + meas_state
                        if measurement not in qpt_dict:
                            qpt_dict[measurement] = []

                        for j in range(4):
                            set_mount_angle(angles[j], j)
                        meas_counts = (counter.get_counts() / args.num_shots).tolist()
                        qpt_dict[measurement] += [meas_counts]
    except KeyboardInterrupt:
        print("Quitting run and saving data")
        write_json(qpt_dict, dir + "qpt_data_ABORTED")
        exit()

    write_json(qpt_dict, dir + "qpt_data")
