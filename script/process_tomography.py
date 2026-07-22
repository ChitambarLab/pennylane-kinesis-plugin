import elliptec
from pennylane_kinesis import swabian, read_config, ElliptecMotors, datetime_now_string, write_json
import numpy as np
import threading

from ThorlabsPM100 import ThorlabsPM100, USBTMC
import time
import pickle
from tqdm import tqdm
from pathlib import Path

# ports = [f"/dev/ttyUSB{i}" for i in range(4)]
ports = [f"/dev/ttyUSB{i}" for i in range(3)]

power_meters = ["P0025560"]  # "P0037258", "P0025560"]  # "P0027639"]  # "P0025560"]


NUM_SHOTS = 100000
NUM_MEASUREMENTS = 5


def get_serial_no(usbtmc_device):
    return usbtmc_device.query("*IDN?").split(",")[2]


class Meter:
    def __init__(self, swabian_mode, HV_meter=False):
        self.swabian_mode = swabian_mode
        self.HV_meter = HV_meter
        H_meter = None
        V_meter = None
        if swabian_mode:
            self.counter = swabian.Counter(shots=None, integration_time=1)
            self.counter.detector_setup()
        else:
            # rm = pyvisa.ResourceManager()

            # resources = rm.list_resources()
            # print(resources)
            list_check = [0, 1] if HV_meter else [0]
            insts = [USBTMC(device=f"/dev/usbtmc{x}") for x in list_check]
            for inst in insts:
                print(inst.query("*IDN?"))
                if get_serial_no(inst) == power_meters[0]:
                    H_meter = ThorlabsPM100(inst=inst)
                elif get_serial_no(inst) == power_meters[1]:
                    V_meter = ThorlabsPM100(inst=inst)

            # H_resource = [x for x in resources if 'P0027252' in x][0]

            # Change serial number here if you use a different meter
            # H_resource = [x for x in resources if power_meters[0] in x][0]

            # print(resources)
            # inst_H = rm.open_resource(H_resource, timeout=1)
            # H_meter = ThorlabsPM100(inst=inst_H)
            # H_meter.configure.scalar.power()

            # V_meter = None
            # If using two different power meters
            # if HV_meter:
            # Change serial number here if you use a different meter
            #    V_resource = [x for x in resources if power_meters[1] in x][0]
            #    inst_V = rm.open_resource(V_resource, timeout=1)
            #    V_meter = ThorlabsPM100(inst=inst_V)
            #    V_meter.configure.scalar.power()
            self.H_meter = H_meter
            self.V_meter = V_meter

    def get_counts(self):
        counts = None

        if self.swabian_mode:
            counts = self.counter.get_counts()
        else:
            if self.HV_meter:
                counts = (self.H_meter.read, self.V_meter.read)
            else:
                counts = (self.H_meter.read, 0)
        return counts

    def set_integration_time(self, time):
        if self.swabian_mode:
            self.counter.set_integration_time(time)


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

    for rotator in rotators:
        print("Homing Rotators")
        rotator.home()
    parallel_set_angles(rotators, [0, 0, 0, 0], zero_offsets)

    print("background angles are", background_angles)
    # timetagger = swabian.Counter()
    # timetagger.shots = NUM_SHOTS
    # timetagger.detector_setup()

    start_timedate = datetime_now_string()

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

    qpt_dict = {}
    qpt_dict["shots"] = NUM_SHOTS
    qpt_dict["num_reps"] = NUM_MEASUREMENTS
    # timetagger.auto_shot_rate()
    # shot_rate = timetagger.shot_rate_each.tolist()
    # qpt_dict["shot_rate"] = shot_rate

    # timetagger.auto_integration_time()
    # print("Taking background measurement")
    # parallel_set_angles(rotators, background_angles, np.zeros(4))
    # old_int_time = timetagger.integration_time

    # qpt_dict["integration_time"] = old_int_time
    # timetagger.set_integration_time(1)
    # timetagger.shots = None
    # background_counts = []
    # for i in range(NUM_MEASUREMENTS):
    #    background_counts.append(timetagger.get_counts().tolist())
    # qpt_dict["background_rate"] = background_counts
    # timetagger.shots = None
    # timetagger.set_integration_time(old_int_time)
    # print("Background counts are", background_counts)

    meter = Meter(swabian_mode=False)

    aoi = "45"
    glass = False

    if glass:
        folder = f"./data/process_tomography/SAMPLE_3_28MM_GLASS/{aoi}_aoi_{start_timedate}"
    else:
        folder = f"./data/process_tomography/SAMPLE_3_28MM_AR/{aoi}_aoi_{start_timedate}"

    folder = f"./data/process_tomography/SAMPLE_3_28MM_AR/air_{start_timedate}"
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

                    meas_counts = meter.get_counts()
                    qpt_dict[measurement] += [meas_counts]
                    # end_time = time.time()
                    # print(f"Took {end_time - start_time} to collect {meas_state}{prep_state}.")
                    # print(f"Estimated time remaining: {((NUM_MEASUREMENTS - i) * 36) * (end_time - start_time)}")
    except KeyboardInterrupt:
        print("Quitting run and saving data")
        write_json(qpt_dict, str(folder) + "/" + "qpt_data_ABORTED")
        exit()

    write_json(qpt_dict, str(folder) + "/" + "qpt_data")
