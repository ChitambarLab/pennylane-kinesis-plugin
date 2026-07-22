import elliptec
from pennylane_kinesis import swabian, read_config, ElliptecMotors, datetime_now_string, write_json
import numpy as np
import threading
import time
import pickle
from tqdm import tqdm
from pathlib import Path

NUM_SHOTS = 10000
NUM_MEASUREMENTS = 5

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


    # hwp_linspace = np.linspace(0,90*np.pi/180,90)
    # qwp_linspace = np.linspace(0,180*np.pi/180,180)

    # qwp_sweep_angles= (qwp_linspace - np.pi/2)**3
    # qwp_sweep_angles /= qwp_sweep_angles[-1]
    # qwp_sweep_angles= np.pi/2*qwp_sweep_angles
    # qwp_sweep_angles = qwp_sweep_angles*180/np.pi+90

    # hwp_sweep_angles = (hwp_linspace - np.pi/4)**3
    # hwp_sweep_angles /= hwp_sweep_angles[-1]
    # hwp_sweep_angles = np.pi/4*hwp_sweep_angles
    # hwp_sweep_angles = hwp_sweep_angles*180/np.pi+45

    # hwp_sweep_angles = np.append(hwp_sweep_angles, hwp_sweep_angles+np.ones_like(hwp_sweep_angles)*90)

    # print("qwp_sweep")
    # print(qwp_sweep_angles)
    # print("hwp_sweep")
    # print(hwp_sweep_angles)

    hwp_sweep_angles = np.linspace(0,360,360*10)
    
    hwp_sweep_angles = np.array(hwp_sweep_angles)
    qwp_sweep_angles = [0]

    qwp_sweep_angles = np.array(qwp_sweep_angles)

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
    # for meas_qwp_angle in tqdm(meas_angles):
    #     for meas_hwp_angle in tqdm(meas_angles, leave=False):
    #         meas_qwp_angle_deg = meas_qwp_angle*180/np.pi
    #         meas_hwp_angle_deg = meas_hwp_angle*180/np.pi
    #         # if skip:
    #         #     skip = False
    #         #     print(f"Skipping {meas_qwp_angle_deg}{meas_hwp_angle_deg}")
    #         #     continue
    #         counts = np.zeros((hwp_sweep_angles.shape[0], qwp_sweep_angles.shape[0],2))
    
    meas_settings = [[0,np.pi/4],[np.pi/4,0]]
    for meas_setting in meas_settings:
    # for i,rotator in tqdm(enumerate(rotators)):
        meas_hwp_angle_deg = meas_setting[0]*180/np.pi
        meas_qwp_angle_deg = meas_setting[1]*180/np.pi

        counts = np.zeros((NUM_MEASUREMENTS,hwp_sweep_angles.shape[0],qwp_sweep_angles.shape[0],2))

        print(f"Taking data for {meas_hwp_angle_deg}{meas_qwp_angle_deg}")
        # print(f"Taking data for {motors.wp_details[i]['device']}")

        metadata = {"meas_hwp_angle": meas_hwp_angle_deg, "meas_qwp_angle": meas_qwp_angle_deg}
        # metadata = {"rotator": f"{motors.wp_details[i]['device']}"}

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
        background_counts = []
        for k in range(NUM_MEASUREMENTS):
            background_counts.append(timetagger.get_counts().tolist())

        metadata["background_rate"] = background_counts
        timetagger.shots = NUM_SHOTS
        timetagger.set_integration_time(old_int_time)
        print("Background counts are", background_counts)

        start_time = time.time()
        

        # for j, prep_hwp_angle in tqdm(enumerate(hwp_sweep_angles),leave=False):
        #     # prep_qwp_angle = 30
        #     angles = np.zeros(4)
        #     angles[i] = prep_hwp_angle
        #     parallel_set_angles(rotators, angles, [motors.wp_details[l]["angle_offset"] for l in range(len(rotators))])
        #     for k in range(NUM_MEASUREMENTS):
        #         counts[k,j] = np.array(timetagger.get_counts())
        for i, prep_hwp_angle in tqdm(enumerate(hwp_sweep_angles)):
            for j, prep_qwp_angle in tqdm(enumerate(qwp_sweep_angles),leave=False):
            # prep_qwp_angle = 30
                angles = [prep_hwp_angle, prep_qwp_angle, meas_hwp_angle_deg, meas_qwp_angle_deg]
                parallel_set_angles(rotators, angles, [motors.wp_details[i]["angle_offset"] for i in range(len(rotators))])
                for k in range(NUM_MEASUREMENTS):
                    counts[k,i,j] = np.array(timetagger.get_counts())

        end_time = time.time()

        runtime = end_time - start_time
        print(f"Run took {runtime} seconds.")
        metadata["runtime"] = runtime
        metadatas.append(metadata)
        folder = f"./data/wp_sweeps/{start_timedate}"
        Path(folder).mkdir(parents=True, exist_ok=True)
        meas_setting_path = folder + f"/{meas_hwp_angle_deg}_{meas_qwp_angle_deg}"
        with open(Path(meas_setting_path+".pkl"), "wb") as f:
            pickle.dump(counts.tolist(), f)

        write_json(metadata, meas_setting_path)
        
