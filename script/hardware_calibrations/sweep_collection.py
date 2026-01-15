import pyvisa
from ThorlabsPM100 import ThorlabsPM100, USBTMC
import elliptec
from pennylane_kinesis.elliptec_rotator_device import ElliptecMotors
from pennylane_kinesis.swabian import Counter
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
from pennylane_kinesis.file_utilities import datetime_now_string, read_config, write_json
import scipy as sp
import numpy
import scipy
from scipy.optimize import fmin
from pennylane_kinesis import swabian
import pickle
from pathlib import Path
import re
from tqdm import tqdm

# ports = [f"/dev/ttyUSB{i}" for i in range(4)]
ports = [f"/dev/ttyUSB{i}" for i in range(3)]

class Meter:
    def __init__(self, swabian_mode, HV_meter=False):
        self.swabian_mode = swabian_mode
        self.HV_meter = HV_meter
        if swabian_mode:
            self.counter = swabian.Counter(shots=None, integration_time=1)
            self.counter.detector_setup()
        else:
            rm = pyvisa.ResourceManager()
            resources = rm.list_resources()

            # H_resource = [x for x in resources if 'P0027252' in x][0]

            # Change serial number here if you use a different meter
            H_resource = [x for x in resources if "P0027254" in x][0]

            # print(resources)
            inst_H = rm.open_resource(H_resource, timeout=1)
            H_meter = ThorlabsPM100(inst=inst_H)
            H_meter.configure.scalar.power()

            V_meter = None
            # If using two different power meters
            if HV_meter:
                # Change serial number here if you use a different meter
                V_resource = [x for x in resources if "P0027252" in x][0]
                inst_V = rm.open_resource(V_resource, timeout=1)
                V_meter = ThorlabsPM100(inst=inst_V)
                V_meter.configure.scalar.power()
            self.H_meter = H_meter
            self.V_meter = V_meter

    def get_counts(self):
        counts = None
        if self.swabian_mode:
            counts = self.counter.get_counts()
        else:
            if self.HV_meter:
                counter = (self.H_meter.read, self.V_meter.read)
            else:
                counter = (self.H_meter.read, 0)
        print(counts)
        return counts

    def set_integration_time(self, time):
        if self.swabian_mode:
            self.counter.set_integration_time(time)


def move_to_angle(device, angle, offset_angle=0):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Manual alignment",
        description="Manually align the rotation stages to the desired angles.",
    )

    parser.add_argument(
        "-p",
        "--num_points",
        type=int,
        default=360,
        help="Number of points for 360 degrees of rotation.",
    )
    parser.add_argument(
        "-m",
        "--num_measurements",
        type=int,
        default=5,
        help="Number of counts to collect at each point for statistics.",
    )

    parser.add_argument(
        "--repeatability",
        default=False,
        action='store_true',
        help="If num measurements is greater than 1, then check repeatability of the waveplate by rotating to x+360 then back to x\
        for each angle"
    )

    parser.add_argument(
        "--num_shots",
        type=int,
        default=10000,
        help="Num shots to take with each measurement."
    )

    parser.add_argument(
        "-n",
        "--name",
        help="Name of the waveplate being calibrated (changes filename)",
    )

    parser.add_argument("-l", "--mount", default=0,type=int, help="Which mount to rotate")

    args = parser.parse_args()

    # inst = USBTMC()

    # power_meter = ThorlabsPM100(inst = inst)
    # print(power_meter)
    # power_meter.configure.scalar.power()
    # print(power_meter.read)
    # exit()
    motors = ElliptecMotors()
    sweep_angles = np.linspace(0,360,args.num_points)

    rotators = [motors.prep_hwp, motors.prep_qwp, motors.meas_hwp, motors.meas_qwp]
    for i,rotator in enumerate(rotators):
        rotator.home()
        move_to_angle(rotator, 0, motors.wp_details[i]["angle_offset"])

    rotator = rotators[args.mount]
    background_angle = motors.wp_details[args.mount]["background_angle"]
    wp_details = motors.wp_details[args.mount]

    move_to_angle(rotator, 0, wp_details["angle_offset"])

    print("background angles are", background_angle)
    timetagger = swabian.Counter()
    timetagger.shots = args.num_shots
    timetagger.detector_setup()
    start_timedate = datetime_now_string()
    counts = np.zeros((args.num_measurements,sweep_angles.shape[0],2))

    print(f"Taking data for Mount {args.mount} ({wp_details['device']})")
    print(f"Taking data for {wp_details['device']}")

    metadata = {}
    metadata = {"rotator": f"{wp_details['device']}"}

    start_time = time.time()
        
    for i, angle in tqdm(enumerate(sweep_angles)):
        move_to_angle(rotator, angle, 0)
        for k in range(args.num_measurements):
            if args.repeatability:
                move_to_angle(rotator, (angle+180), wp_details["angle_offset"])
                move_to_angle(rotator, angle, wp_details["angle_offset"])
            # counts[k,i] = np.array(power_meter.read)
            counts[k,i] = np.array(timetagger.get_counts())

    end_time = time.time()

    runtime = end_time - start_time
    print(f"Run took {runtime} seconds.")
    metadata["runtime"] = str(runtime)
    folder = f"./data/wp_sweeps/single_sweeps/{args.name}/"
    Path(folder).mkdir(parents=True, exist_ok=True)
    filepath = folder+ f"{start_timedate}"
    with open(Path(filepath+".pkl"), "wb") as f:
        pickle.dump(counts.tolist(), f)

    write_json(metadata, filepath)

    # meas = np.loadtxt("data/wp_sweeps/single_sweeps/bQWP/DATA03.CSV",delimiter="	")

    # print(meas)
    # counts = np.array(meas)[:,0]
    # sweep_angles = np.linspace(0,360,248)
    # mean_vals = counts
    # exit()

    mean_vals = np.mean(counts,axis=0)
    std_vals = np.std(counts,axis=0)

    plt.plot(sweep_angles, mean_vals[:,0], color='r', label="H")
    plt.plot(sweep_angles, mean_vals[:,1], color='b', label="H")
    plt.fill_between(
        sweep_angles,
        (mean_vals - std_vals)[:,0],
        (mean_vals + std_vals)[:,0],
        alpha=1 / 4,
        color="r",
        # label="Standard Error",
    )
    plt.fill_between(
        sweep_angles,
        (mean_vals - std_vals)[:,1],
        (mean_vals + std_vals)[:,1],
        alpha=1 / 4,
        color="b",
        # label="Standard Error",
    )
    plt.title(f"Angle Sweep for {args.name}\n")
    plt.ylabel("Photon Counts")
    plt.xlabel("Angle of rotation ($^\circ$)")
    plt.grid()
    # plt.legend()
    plt.tight_layout()
    plt.savefig(filepath+".png")
    

