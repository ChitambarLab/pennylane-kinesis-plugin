import pyvisa
# from ThorlabsPM100 import ThorlabsPM100
import elliptec
from pennylane_kinesis.elliptec_rotator_device import ElliptecRotatorDevice
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
from pennylane_kinesis.file_utilities import datetime_now_string, read_config
import scipy as sp
import numpy
import scipy
from scipy.optimize import fmin
from pennylane_kinesis import swabian
import pickle
from pathlib import Path
import re

NUM_MEASUREMENTS = 1

ports = [f"/dev/ttyUSB{i}" for i in range(4)]


def set_angle(device, angle):
    angle_ret = None
    while angle_ret is None:
        angle_ret = device.set_angle(angle % 360)
        if angle_ret is None:
            print("ERROR")
            print("angle is none")
    print("set angle to", angle)
    print("actual angle is", angle_ret)

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



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Manual alignment",
        description="Manually align the rotation stages to the desired angles.",
    )

    parser.add_argument(
        "-a",
        "--apds",
        default=None,
        help="Use APD readings (this will only give power vs angle plot).",
    )

    parser.add_argument(
        "--average",
        default=False,
        action="store_true",
        help="Average the data from multiple trials.",
    )

    parser.add_argument(
        "--num_meters",
        default=1,
        type=int,
        help="Number of power meters connected.",
    )
    
    parser.add_argument(
        "--counts",
        default=False,
        action="store_true",
        help="Just output the counts of each APD.",
    )

    parser.add_argument(
        "-t",
        "--time",
        type=int,
        default=3,
        help="Time to wait between rotations to take data (in manual mode)",
    )

    parser.add_argument(
        "-s",
        "--step",
        type=float,
        default=22.5,
        help="Step size for rotation (in manual mode)",
    )


    parser.add_argument(
        "--serial_no",
        default=False,
        action="store_true",
        help="Print out all devices scanned by Elliptec controller.",
    )

    parser.add_argument("--home", default=False, action="store_true", help="test homing")

    parser.add_argument(
        "--plot",
        default=False,
        action="store_true",
        help="Plot the data from existing file.",
    )

    parser.add_argument(
        "-n",
        "--name",
        help="Name of the waveplate being calibrated (changes filename)",
    )

    parser.add_argument("--angle", help="Set waveplate to angle and get average reading.")

    parser.add_argument("--address", help="Set the address of the newly connected mount.")

    parser.add_argument(
        "--polarizer",
        default=False,
        action="store_true",
        help="Calibrate a polarizer (Manually rotating)",
    )

    parser.add_argument(
        "--meter",
        default=False,
        action="store_true",
        help="Use power meter to take data.",
    )

    parser.add_argument(
        "-r",
        "--repeat",
        default=False,
        action="store_true",
        help="Rotate the mount by 360 degrees and check if the power remains the same.",
    )

    parser.add_argument("-l", "--mount", default=None, help="Which mount to rotate")

    parser.add_argument("--zero", default=False, action="store_true")

    parser.add_argument("--background", default=False, action="store_true")

    parser.add_argument("--parallel", default=False, action="store_true")
    args = parser.parse_args()

    if args.parallel:
        port = ports[int(args.mount)]
        address = "0"
    # H_meter, V_meter = connect_power_meter(True)
    if args.meter:
        if args.apds:
            meter = Meter(True)
        else:
            meter = Meter(False, args.num_meters == 2)

    # Get counts of the APDs
    if args.counts:
        h_counts, v_counts = meter.get_counts()
        print(f"T counts: {h_counts}, R counts {v_counts}")
        exit()

    # If only taking background counts (no rotation)
    if args.background:
        wp_details = read_config()
        meter.set_integration_time(1)
        controller = elliptec.Controller(port, timeout=3, write_timeout=3)
        for i in range(1, 5):
            device = elliptec.Rotator(controller, address=str(i))
            device.home()
            offset = wp_details[i - 1]["background_angle"]
            ANGLE = offset
            set_ret = set_angle(device, ANGLE)
            print("return is", set_ret)
            print("set angle of ", i, "to", ANGLE)
        meter.get_counts()
        exit()

    if args.zero:
        wp_details = read_config()
        meter.set_integration_time(1)
        controller = elliptec.Controller(port, timeout=3, write_timeout=3)
        for i in range(1, 5):
            device = elliptec.Rotator(controller, address=str(i))
            device.home()
            offset = wp_details[i - 1]["angle_offset"]
            ANGLE = offset
            set_ret = set_angle(device, ANGLE)
            print("return is", set_ret)
            print("set angle of ", i, "to", ANGLE)
            # if i == 1:
            #     ANGLE = offset
            #     set_ret = device.set_angle(ANGLE)
            #     print("return is", set_ret)
            #     print("set angle of ", i, "to", ANGLE)
            # elif i == 3:
            #     ANGLE = offset + 22.5
            #     set_ret = device.set_angle(ANGLE)
            #     print("return is", set_ret)
            #     print("set angle of ", i, "to", ANGLE)
            # else:
            #     ANGLE = offset
            #     set_ret = device.set_angle(ANGLE)
            #     print("return is", set_ret)
            #     print("set angle of ", i, "to", ANGLE)
        meter.get_counts()
        exit()

    if args.apds and not args.mount and not args.polarizer:
        print(args.apds)
        if not args.name:
            print("Need to set a name for the calibration output.")

        wp_details = read_config()
        datetime = datetime_now_string()
        path = Path(
            "./data/calibrations/zero_angle/" + args.name + "/power_vs_angle_auto_" + args.name + "_" + datetime
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        h_counts = []
        v_counts = []
        angles = []
        print("Setting all angles to zero.")
        print("Connecting to elliptec controller")
        controller = elliptec.Controller(port, timeout=3, write_timeout=3)
        devices = []
        for i in range(1, 5):
            # if i == 2:
            # continue
            print("Connecting to ", i)
            device = elliptec.Rotator(controller, address=str(i))
            device.home()
            devices.append(device)
            offset = wp_details[i - 1]["angle_offset"]
            print("Setting angle of ", i, "to", offset)

            set_angle(device, offset)
            time.sleep(0.5)

        print(len(devices), "devices connected")
        print()
        device = devices[int(args.apds) - 1]
        num_trials = 1
        # TODO: Change this to 1 if you want finer resolution
        for j in range(num_trials):
            start_time = time.time()
            for k, i in enumerate(range(0, 360, 5)):
                if j == 0:
                    h_counts.append([])
                    v_counts.append([])
                    angles.append(i)
                set_angle(device, i)
                (h, v) = meter.get_counts()
                h_counts[k].append(h)
                v_counts[k].append(v)
            end_time = time.time()
            print(f"Trial {j} done. Took {end_time - start_time} seconds.")
            print(h_counts)
            print(v_counts)
        with open(str(path) + ".pkl", "wb") as f:
            pickler = pickle.Pickler(f)
            pickler.dump((angles, h_counts, v_counts))

        mean_h_counts = np.mean(np.array(h_counts), axis=1)
        mean_v_counts = np.mean(np.array(v_counts), axis=1)
        sum_mean_counts = mean_h_counts + mean_v_counts
        std_dev_sum_counts = np.std(np.array(h_counts) + np.array(v_counts), axis=1)
        std_dev_h_counts = np.std(np.array(h_counts), axis=1)
        std_dev_v_counts = np.std(np.array(v_counts), axis=1)
        plt.plot(angles, mean_h_counts, color="red", label="H")
        plt.plot(angles, mean_v_counts, color="blue", label="V")
        plt.plot(
            angles,
            np.array(mean_h_counts) + np.array(mean_v_counts),
            color="green",
            label="Total",
        )
        plt.fill_between(
            angles,
            sum_mean_counts - std_dev_sum_counts,
            sum_mean_counts + std_dev_sum_counts,
            color="darkgreen",
            alpha=0.3,
        )
        plt.axvline(
            x=wp_details[int(args.apds) - 1]["angle_offset"],
            color="black",
            linestyle="--",
            label="Zero angle",
        )
        plt.xlabel("Angle of rotation (degrees)")
        plt.ylabel("Photon Counts")
        plt.title("Photon counts vs angle of rotation " + args.name)
        plt.legend()
        plt.savefig(str(path) + ".png")
        exit()

    # Print out scanned devices
    if args.serial_no:
        controller = elliptec.Controller(port, timeout=3, write_timeout=3)

        print(elliptec.scan.scan_for_devices(controller, 0, 5))

        exit()

    # Check repeatability of waveplate rotations
    elif args.repeat:
        controller = elliptec.Controller(port, timeout=3, write_timeout=3)

        device = elliptec.Rotator(controller, address="0")

        device.home()

        device.set_angle(0)
        device.set_jog_step(360)
        powers = []
        angles = []
        iter = []

        for i in range(10):
            itr_powers = []
            device.set_angle(0)
            time.sleep(0.1)
            for j in range(NUM_MEASUREMENTS):
                power = meter.read
                itr_powers.append(power)
            avg_power = np.mean(np.array(itr_powers))
            angle = device.get_angle()
            print(angle, i, avg_power)

            angles.append(angle)
            iter.append(i)
            powers.append(avg_power)

            itr_powers = []
            device.set_angle(359.99)
            time.sleep(0.1)
            for j in range(NUM_MEASUREMENTS):
                power = meter.read
                itr_powers.append(power)
            avg_power = np.mean(np.array(itr_powers))
            angle = device.get_angle()
            print(angle, i, avg_power)
            angles.append(angle)
            iter.append(i)
            powers.append(avg_power)
            # device.jog()
            # time.sleep(0.1)

        with open(
            "./data/calibrations/zero_angle/"
            + args.name
            + "/power_vs_angle_repeat_"
            + args.name
            + "_"
            + datetime_now_string()
            + ".pkl",
            "wb",
        ) as f:
            pickler = pickle.Pickler(f)
            pickler.dump((angles, powers, iter))

    # Home the device twice
    elif args.home:
        controller = elliptec.Controller(port)
        device = elliptec.Rotator(controller, address="0")

        device.home()
        time.sleep(1)
        device.set_angle(90)
        time.sleep(1)
        device.home()
        
    # Calibrating a polarizer (with manual rotation)
    elif args.polarizer:
        from pathlib import Path, PurePath

        powers = []
        angles = []

        for i in range(int(360 / 22.5)):
            angle = 22.5 * i
            angles.append(22.5 * i)
            print("Press enter to take data..")
            input()
            # time.sleep(3)
            h_counts, v_counts = meter.get_counts()
            powers.append(np.array([h_counts, v_counts]))
            print("Took data for angle", 22.5 * i)
        path = Path(
            "./data/calibrations/zero_angle/"
            + args.name
            + "/power_vs_angle_"
            + args.name
            + "_"
            + datetime_now_string()
            + ".pkl"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickler = pickle.Pickler(f)
            pickler.dump((angles, powers))
        exit()

    else:
        print("opening port")
        print(port)
        print("address", address)
        controller = elliptec.Controller(port)

        device = elliptec.Rotator(controller, address=address)

        device.home()

        device.set_angle(0)
        device.set_jog_step(1)
        powers_h = []
        powers_v = []
        angles = []

        while True:
            angle = device.get_angle()
            itr_powers_h = []
            itr_powers_v = []
            for i in range(NUM_MEASUREMENTS):
                h_counts, v_counts = meter.get_counts()
                itr_powers_h.append(h_counts)
                itr_powers_v.append(v_counts)
            avg_power_h = np.mean(np.array(itr_powers_h))
            avg_power_v = np.mean(np.array(itr_powers_v))
            print(angle, avg_power_h, avg_power_v)
            angles.append(angle)
            powers_h.append(avg_power_h)
            powers_v.append(avg_power_v)
            if angle >= 360:
                break
            device.jog()
            time.sleep(0.1)
        folder = Path("./data/calibrations/zero_angle/" + args.name)
        folder.mkdir(parents=True, exist_ok=True)
        path_plot = str(folder) + "/power_vs_angle_auto_" + args.name + "_" + datetime_now_string() + ".png"
        path_pkl = str(folder) + "/power_vs_angle_auto_" + args.name + "_" + datetime_now_string() + ".pkl"

        with open(path_pkl, "wb") as f:
            pickler = pickle.Pickler(f)
            pickler.dump((angles, powers_h, powers_v))

        plt.plot(angles, powers_h, color="red", label="H")
        plt.plot(angles, powers_v, color="blue", label="V")
        plt.xlabel("Angle of rotation (degrees)")
        plt.ylabel("Laser Power (W)")
        plt.title("Power vs angle of rotation " + args.name)
        plt.legend()

        plt.savefig(path_plot)
