import elliptec
from pennylane_kinesis import swabian, read_config, ElliptecMotors, datetime_now_string, write_json
import numpy as np
import threading
import time
import pickle
from tqdm import tqdm
from pathlib import Path

NUM_SHOTS = 100000
NUM_MEASUREMENTS = 5
INTERVAL = 300 #seconds

def move_to_angle(device, angle, offset_angle):
    angle_ret = None
    while angle_ret is None:
        angle_ret = device.set_angle(
            (angle + offset_angle) % 360
        )  # This will block until the device reaches the desired angle
        if angle_ret is None:
            print(
                "Angle return is None when setting",
                device.info["Serial No."],
                "to ",
                angle,
            )
            print("Trying again...")
            time.sleep(0.01)


def parallel_set_angles(rotators, angles, offsets):
    threads = []
    for i, angle in enumerate(angles):
        x = threading.Thread(target=move_to_angle, args=(rotators[i], angle, offsets[i]))
        threads.append(x)
        x.start()

    for i, thread in enumerate(threads):
        thread.join()


if __name__ == "__main__":
    motors = ElliptecMotors()

    # meas_angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]

    # hwp_sweep_angles = np.linspace(0,180,180)
    # qwp_sweep_angles = np.linspace(0,180,180)
    rotators = [motors.prep_hwp, motors.prep_qwp, motors.meas_hwp, motors.meas_qwp]
    background_angles = [motors.wp_details[i]["background_angle"] for i in range(len(rotators))]
    zero_offsets = [motors.wp_details[i]["angle_offset"] for i in range(len(rotators))]
    parallel_set_angles(rotators, [0, 0, 0, 0], zero_offsets)

    print("background angles are", background_angles)
    timetagger = swabian.Counter()
    timetagger.shots = NUM_SHOTS
    timetagger.detector_setup()
    

    run_start_timedate = datetime_now_string()

    main_folder = f"./data/process_tomography/fiber_channel/{run_start_timedate}"
    Path(main_folder).mkdir(parents=True, exist_ok=True)

    states_names = ["H", "V", "D", "A", "R", "L"]
    prep_states_dict = {
        "H": [0, 0],
        "V": [45, 0],
        "D": [22.5, 45],
        "A": [67.5, 45],
        "R": [0, 135],
        "L": [0, 45],
    }
    meas_states_dict = {
        "H": [0, 0],
        "V": [45, 0],
        "D": [22.5, 45],
        "A": [67.5, 45],
        "R": [0, 45],
        "L": [0, 135],
    }
    # meas_states_dict = {"H": [45,0], "V": [0,0], "D": [67.5,0], "A": [22.5,0], "R": [0,45], "L": [0,135]}

    prep_states = [prep_states_dict[state] for state in states_names]
    meas_states = [meas_states_dict[state] for state in states_names]
    j = 0 
    while True:
        timetagger.shots = NUM_SHOTS
        qpt_dict = {}
        qpt_dict["shots"] = NUM_SHOTS
        qpt_dict["num_reps"] = NUM_MEASUREMENTS
        timetagger.auto_shot_rate()
        shot_rate = timetagger.shot_rate_each.tolist()
        qpt_dict["shot_rate"] = shot_rate

        timetagger.auto_integration_time()
        print("Taking background measurement")
        parallel_set_angles(rotators, background_angles, np.zeros(4))
        old_int_time = timetagger.integration_time

        qpt_dict["integration_time"] = old_int_time
        timetagger.set_integration_time(1)
        timetagger.shots = None
        background_counts = []
        for i in range(NUM_MEASUREMENTS):
            background_counts.append(timetagger.get_counts().tolist())

        qpt_dict["background_rate"] = background_counts
        timetagger.shots = None
        timetagger.set_integration_time(old_int_time)
        print("Background counts are", background_counts)

        tomography_timedate = datetime_now_string()

        folder = main_folder+f"/{tomography_timedate}"
        Path(folder).mkdir(parents=True, exist_ok=True)

        # Collect tomography data

        try:
            for i in tqdm(range(NUM_MEASUREMENTS), leave=False):
                # print(f"Collecting data for repetition {i}")
                for prep_state in tqdm(states_names, leave=False):
                    for meas_state in tqdm(states_names, leave=False):
                        # start_time = time.time()
                        # print(f"Taking data for {prep_state}{meas_state}")
                        prep_angles = prep_states_dict[prep_state]
                        meas_angles = meas_states_dict[meas_state]
                        angles = prep_angles + meas_angles
                        # print(angles)
                        measurement = prep_state + meas_state
                        # print(measurement)
                        if measurement not in qpt_dict:
                            qpt_dict[measurement] = []

                        parallel_set_angles(rotators, angles, zero_offsets)

                        meas_counts = (timetagger.get_counts()).tolist()
                        qpt_dict[measurement] += [meas_counts]
                        # end_time = time.time()
                        # print(f"Took {end_time - start_time} to collect {meas_state}{prep_state}.")
                        # print(f"Estimated time remaining: {((NUM_MEASUREMENTS - i) * 36) * (end_time - start_time)}")
        except KeyboardInterrupt:
            print("Quitting run and saving data")
            write_json(qpt_dict, str(folder) + "/" + "qpt_data_ABORTED")
            exit()

        write_json(qpt_dict, str(folder) + "/" + "qpt_data")
        print(f"Wrote qpt data at {str(folder)}")
        print(f"Sleeping for {INTERVAL}")
        print(f"Took data for {j*INTERVAL} seconds or {(j*INTERVAL)/60} minutes or {j*INTERVAL/360} hours.")
        time.sleep(INTERVAL)
        j+=1
