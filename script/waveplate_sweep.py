import elliptec
from pennylane_kinesis import swabian, read_config, ElliptecMotors, datetime_now_string, write_json
import numpy as np
import threading
import time
import pickle
from tqdm import tqdm
from pathlib import Path

NUM_SHOTS = 100000

def move_to_angle(device, angle, offset_angle):
    angle_ret = None
    while angle_ret is None:
        angle_ret = device.set_angle((angle+offset_angle)%360)  # This will block until the device reaches the desired angle
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

    meas_angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]

    hwp_sweep_angles = np.linspace(0,180,180)
    qwp_sweep_angles = np.linspace(0,180,180)
    rotators = [motors.prep_hwp, motors.prep_qwp, motors.meas_hwp, motors.meas_qwp]
    background_angles = [motors.wp_details[i]["background_angle"] for i in range(len(rotators))]

    parallel_set_angles(rotators, [0,0,0,0], [motors.wp_details[i]["angle_offset"] for i in range(len(rotators))])

    print("background angles are", background_angles)
    timetagger = swabian.Counter()
    timetagger.shots = NUM_SHOTS
    timetagger.detector_setup()

    start_timedate = datetime_now_string()
    metadatas = []
    skip = False
    for meas_qwp_angle in tqdm(meas_angles):
        for meas_hwp_angle in tqdm(meas_angles, leave=False):
            meas_qwp_angle_deg = meas_qwp_angle*180/np.pi
            meas_hwp_angle_deg = meas_hwp_angle*180/np.pi
            if skip:
                skip = False
                print(f"Skipping {meas_qwp_angle_deg}{meas_hwp_angle_deg}")
                continue
            counts = np.zeros((hwp_sweep_angles.shape[0], qwp_sweep_angles.shape[0],2))
            
            print(f"Taking data for {meas_qwp_angle_deg}{meas_hwp_angle_deg}")

            metadata = {"meas_qwp_angle": meas_qwp_angle_deg, "meas_hwp_angle": meas_hwp_angle_deg}

            timetagger.auto_shot_rate()
            shot_rate = timetagger.shot_rate_each.tolist()
            metadata["shot_rate"] = shot_rate

            timetagger.auto_integration_time()
            print("Taking background measurement")
            parallel_set_angles(rotators, background_angles, np.zeros(4))
            old_int_time = timetagger.integration_time

            metadata["integration_time"] = old_int_time
            timetagger.set_integration_time(1)
            timetagger.shots = None
            background_counts = timetagger.get_counts().tolist()
            metadata["background_rate"] = background_counts
            timetagger.shots = NUM_SHOTS
            timetagger.set_integration_time(old_int_time)
            print("Background counts are", background_counts)

            start_time = time.time()
            

            for i, prep_hwp_angle in tqdm(enumerate(hwp_sweep_angles),leave=False):
                for j, prep_qwp_angle in tqdm(enumerate(qwp_sweep_angles),leave=False):
                    angles = [prep_hwp_angle, prep_qwp_angle, meas_qwp_angle_deg, meas_hwp_angle_deg]
                    parallel_set_angles(rotators, angles, [motors.wp_details[i]["angle_offset"] for i in range(len(rotators))])
                    counts[i,j] = np.array(timetagger.get_counts())

            end_time = time.time()

            runtime = end_time - start_time
            print(f"Run took {runtime} seconds.")
            metadata["runtime"] = runtime
            metadatas.append(metadata)
            folder = f"./data/wp_sweeps/{start_timedate}"
            Path(folder).mkdir(parents=True, exist_ok=True)
            meas_setting_path = folder + f"/{meas_qwp_angle_deg}_{meas_hwp_angle_deg}"
            with open(Path(meas_setting_path+".pkl"), "wb") as f:
                pickle.dump(counts.tolist(), f)

            write_json(metadata, meas_setting_path)
            
