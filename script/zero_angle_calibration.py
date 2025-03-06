import pyvisa
from ThorlabsPM100 import ThorlabsPM100
import elliptec
from pennylane_kinesis.elliptec_rotator_device import ElliptecRotatorDevice
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
from pennylane_kinesis.file_utilities import datetime_now_string
import scipy as sp
import numpy
import scipy
from scipy.optimize import fmin
from pennylane_kinesis import swabian
import pickle
from pathlib import Path
import re

NUM_MEASUREMENTS = 100


def set_angle(device, angle):
    angle_ret = None
    while angle_ret is None:
        angle_ret = device.set_angle(angle)
        if angle_ret is None:
            print("ERROR")
            print("angle is none")
    print("set angle to", angle)
    print("actual angle is", angle_ret)


def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = numpy.array(tt)
    yy = numpy.array(yy)
    ff = numpy.fft.fftfreq(len(tt), (tt[1] - tt[0]))  # assume uniform spacing
    Fyy = abs(numpy.fft.fft(yy))
    guess_freq = abs(
        ff[numpy.argmax(Fyy[1:]) + 1]
    )  # excluding the zero frequency "peak", which is related to offset
    guess_amp = numpy.std(yy) * 2.0**0.5
    guess_offset = numpy.mean(yy)
    guess = numpy.array([guess_amp, 2.0 * numpy.pi * guess_freq, 0.0, guess_offset])

    def sinfunc(t, A, w, p, c):
        return A * numpy.sin(w * t + p) + c

    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w / (2.0 * numpy.pi)
    fitfunc = lambda t: A * numpy.sin(w * t + p) + c
    return {
        "amp": A,
        "omega": w,
        "phase": p,
        "offset": c,
        "freq": f,
        "period": 1.0 / f,
        "fitfunc": fitfunc,
        "maxcov": numpy.max(pcov),
        "rawres": (guess, popt, pcov),
        "perr": np.sqrt(np.diag(pcov)),
    }


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


def param_sin(x, offset, amplitude, phase, freq):
    return amplitude * np.sin(phase + freq * x) + offset


def read_config():
    config_file = "./src/pennylane_kinesis/2024_09_18_timur_elliptec_config.txt"

    with open(config_file) as f:
        config_lines = f.readlines()

    wp_details = []

    for i in range(1, 5):
        config_line = re.sub(r"[\n\t\s]*", "", config_lines[i])

        entries = config_line.strip().split(",")
        rot_mount = {}
        rot_mount["device"] = entries[0]
        rot_mount["address"] = entries[1]
        rot_mount["serial_number"] = entries[2]
        rot_mount["angle_offset"] = float(entries[3])
        rot_mount["name"] = entries[4]
        rot_mount["background_angle"] = float(entries[5])
        wp_details.append(rot_mount)

    return wp_details


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
        "--num_meters",
        default=1,
        type=int,
        help="Number of power meters connected.",
    )
    parser.add_argument(
        "-m", "--manual", default=False, action="store_true", help="Manual mode (rotate manually)"
    )

    parser.add_argument(
        "--counts", default=False, action="store_true", help="Just output the counts of each APD."
    )

    parser.add_argument(
        "-t",
        "--time",
        type=int,
        default=3,
        help="Time to wait between rotations to take data (in manual mode)",
    )

    parser.add_argument(
        "-s", "--step", type=float, default=22.5, help="Step size for rotation (in manual mode)"
    )

    parser.add_argument(
        "-f", "--fit", default=False, action="store_true", help="fit data (put data into array)"
    )

    parser.add_argument(
        "--file", default=None, help="File to use for fitting data. (with --plot or --fit option)"
    )

    parser.add_argument(
        "--serial_no",
        default=False,
        action="store_true",
        help="Print out all devices scanned by Elliptec controller.",
    )

    parser.add_argument("--home", default=False, action="store_true", help="test homing")

    parser.add_argument(
        "--plot", default=False, action="store_true", help="Plot the data from existing file."
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
        "--gv", default=None, type=int, help="Get the velocity from the motor at address specified."
    )

    parser.add_argument(
        "--sv", default=None, type=int, help="Set the velocity of the motor at address specified."
    )

    parser.add_argument(
        "--velocity", default=None, type=int, help="Velocity to set for the motor (with sv)."
    )

    parser.add_argument(
        "-r",
        "--repeat",
        default=False,
        action="store_true",
        help="Rotate the mount by 360 degrees and check if the power remains the same.",
    )

    parser.add_argument("-l", "--mount", default=1, help="Which mount to rotate")

    parser.add_argument("--zero", default=False, action="store_true")

    parser.add_argument("--background", default=False, action="store_true")
    args = parser.parse_args()
    # H_meter, V_meter = connect_power_meter(True)
    if args.apds:
        meter = Meter(True)
    else:
        meter = Meter(False, args.num_meters == 2)

    # Get counts of the APDs
    if args.counts:
        h_counts, v_counts = meter.get_counts()
        print(f"T counts: {h_counts}, R counts {v_counts}")
        exit()

    # Get velocity of the mounts
    if args.gv:
        controller = elliptec.Controller("COM4", timeout=3, write_timeout=3)
        device = elliptec.Rotator(controller, address=str(args.gv))
        print("Velocity of" + str(args.gv) + "is ", device.send_instruction(b"gv"))
        exit()

    # Set velocity of mounts
    if args.sv and args.velocity:
        controller = elliptec.Controller("COM4", timeout=3, write_timeout=3)
        device = elliptec.Rotator(controller, address=str(args.sv))
        print(device.send_instruction(b"sv" + str(args.velocity).encode()))
        exit()

    # If only taking background counts (no rotation)
    if args.background:
        wp_details = read_config()
        meter.set_integration_time(1)
        controller = elliptec.Controller("COM4", timeout=3, write_timeout=3)
        for i in range(1, 5):
            device = elliptec.Rotator(controller, address=str(i))
            device.home()
            offset = wp_details[i - 1]["background_angle"]
            ANGLE = offset
            set_ret = device.set_angle(ANGLE)
            print("return is", set_ret)
            print("set angle of ", i, "to", ANGLE)
        meter.get_counts()
        exit()

    if args.zero:
        wp_details = read_config()
        meter.set_integration_time(1)
        controller = elliptec.Controller("COM4", timeout=3, write_timeout=3)
        for i in range(1, 5):
            device = elliptec.Rotator(controller, address=str(i))
            device.home()
            offset = wp_details[i - 1]["angle_offset"]
            if i == 1:
                ANGLE = offset
                set_ret = device.set_angle(ANGLE)
                print("return is", set_ret)
                print("set angle of ", i, "to", ANGLE)
            elif i == 3:
                ANGLE = offset + 22.5
                set_ret = device.set_angle(ANGLE)
                print("return is", set_ret)
                print("set angle of ", i, "to", ANGLE)
            else:
                ANGLE = offset
                set_ret = device.set_angle(ANGLE)
                print("return is", set_ret)
                print("set angle of ", i, "to", ANGLE)
        meter.get_counts()
        exit()

    if args.apds:
        print(args.apds)
        if not args.name:
            print("Need to set a name for the calibration output.")

        wp_details = read_config()
        datetime = datetime_now_string()
        path = Path(
            "./data/calibrations/zero_angle/"
            + args.name
            + "/power_vs_angle_auto_"
            + args.name
            + "_"
            + datetime
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        h_counts = []
        v_counts = []
        angles = []
        print("Setting all angles to zero.")
        print("Connecting to elliptec controller")
        controller = elliptec.Controller("COM4", timeout=3, write_timeout=3)
        devices = []
        for i in range(1, 5):
            device = elliptec.Rotator(controller, address=str(i))
            device.home()
            devices.append(device)
            offset = wp_details[i - 1]["angle_offset"]
            print("Setting angle of ", i, "to", offset)

            set_angle(device, offset)

        device = devices[int(args.apds) - 1]
        # TODO: Change this to 1 if you want finer resolution
        for i in range(0, 360, 2):
            angles.append(i)
            set_angle(device, i)
            (h, v) = meter.get_counts()
            h_counts.append(h)
            v_counts.append(v)
        with open(str(path) + ".pkl", "wb") as f:
            pickler = pickle.Pickler(f)
            pickler.dump((angles, h_counts, v_counts))

        plt.plot(angles, h_counts, color="red", label="H")
        plt.plot(angles, v_counts, color="blue", label="V")
        plt.xlabel("Angle of rotation (degrees)")
        plt.ylabel("Photon Counts")
        plt.title("Photon counts vs angle of rotation " + args.name)
        plt.legend()
        plt.savefig(str(path) + ".png")
        exit()

    # Print out scanned devices
    if args.serial_no:
        controller = elliptec.Controller("COM4", timeout=3, write_timeout=3)

        print(elliptec.scan.scan_for_devices(controller, 0, 5))

        exit()

    """
    To set the address of each mount, they need to be connected one by one to the controller
    then the newly connected mount can be accessed at address 0 and assigned a different address.
    """
    if args.address:
        controller = elliptec.Controller("COM4", timeout=3, write_timeout=3)
        print(elliptec.scan_for_devices(controller, start_address=0, stop_address=4))
        device = elliptec.Rotator(controller, address="0")
        device.change_address(str(args.address))
        device.load_motor_info()
        print(device.info)
        exit()

    # Set angle of the waveplate at address 0 and get the average power reading
    if args.angle:
        controller = elliptec.Controller("COM4", timeout=3, write_timeout=3)

        device = elliptec.Rotator(controller, address=str(args.mount))

        print(device.info["Serial No."])

        device.home()

        device.set_angle(float(args.angle))
        itr_powers_h = []
        itr_powers_v = []
        for i in range(NUM_MEASUREMENTS):
            # itr_powers_h.append(H_meter.r)
            h_counts, v_counts = meter.get_counts()
            # itr_powers_v.append(V_meter.read)
        # avg_power_h = np.mean(np.array(itr_powers_h))
        # avg_power_v = np.mean(np.array(itr_powers_v))
        print(f"Device at {device.get_angle()}\n T counts: {h_counts}, R counts {v_counts}")
        exit()

    # Plot the data from the file
    if args.plot:

        with open(args.file, "rb") as f:
            (angles, h_counts, v_counts) = pickle.load(f)

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

            ax.plot(angles, np.array(h_counts) + np.array(v_counts))
            ax.xaxis.set_ticks(np.arange(0, 360, 10))
            ax.axvline(x=204, color="r", linestyle="--")
            ax.axvline(x=294, color="r", linestyle="--")
            ax.axvline(x=204 - 180, color="b", linestyle="--")
            labels = ax.xaxis.get_majorticklabels()
            new_labels = []
            for label in labels:
                if float(label.get_text()) % 30 < 1:
                    new_labels.append(label)
                else:
                    new_labels.append(label.set_text(""))
            ax.xaxis.set_ticklabels(new_labels)
            ax.xaxis.set_tick_params(labelsize=10)

            ax.set_yscale("log")
            ax.set_xlabel("Angle of rotation (degrees)")
            ax.set_ylabel("Photon Counts")
            ax.set_title("Photon counts vs angle of rotation " + args.name)
            ax.legend()
            plt.savefig(
                "./data/calibrations/zero_angle/"
                + args.name
                + "/power_vs_angle_auto_"
                + args.name
                + "_"
                + datetime_now_string()
                + ".png"
            )
        exit()

    # Check repeatability of waveplate rotations
    elif args.repeat:
        controller = elliptec.Controller("COM4", timeout=3, write_timeout=3)

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
        controller = elliptec.Controller("COM4")
        device = elliptec.Rotator(controller, address="0")

        device.home()
        time.sleep(1)
        device.set_angle(90)
        time.sleep(1)
        device.home()

    # Fit existing data
    elif args.fit:
        if args.file == None:
            print("Need to specify a file to use for fitting!")
            exit()
        # Plot fit for a polarizer (with manual rotation)
        if args.polarizer:
            with open(args.file, "rb") as f:
                (angles, powers_h) = pickle.load(f)

                powers_h = np.array(powers_h)
                print(len(powers_h))
                fig = plt.figure()
                x_data = np.linspace(0, 360, num=1000)
                fit_dict_h = fit_sin(angles, powers_h)
                func_h = fit_dict_h["fitfunc"]
                ax = fig.add_subplot(1, 1, 1)

                ax.scatter
                ax.scatter(angles, powers_h, color="red", label="Power data (H)")
                ax.plot(x_data, func_h(x_data), color="orange", label="Sinusoidal fit (H)")

                ax.xaxis.set_ticks(np.arange(0, 360, 10))
                labels = ax.xaxis.get_majorticklabels()
                new_labels = []
                for label in labels:
                    if float(label.get_text()) % 30 < 1:
                        new_labels.append(label)
                    else:
                        new_labels.append(label.set_text(""))
                ax.xaxis.set_ticklabels(new_labels)
                ax.xaxis.set_tick_params(labelsize=10)

                ax.set_title("Power vs Angle of Rotation " + args.name)
                ax.set_xlabel("Angle of rotation (degrees)")
                ax.set_ylabel("Laser Power (μW)")

                ax.legend()
                ax.grid()
                plt.savefig(args.file.split(".")[0] + "fit.png")

                print("zero angle v is", 90 - fit_dict_h["phase"] * (180 / (np.pi)))

                opt = fmin(lambda x: -1 * func_h(x), 90, full_output=True)
                print("max is at", opt[0], "with value", opt[1])

                exit()

        else:
            with open(args.file, "rb") as f:
                (angles, powers_h, powers_v) = pickle.load(f)

                powers_h = np.array(powers_h)
                powers_v = np.array(powers_v)

                fig = plt.figure()
                angles = np.array(angles)
                angles = angles[45:130]
                powers_h = powers_h[45:130]
                powers_v = powers_v[45:130]
                x_data = np.linspace(90, 260, num=1000)

                fit_dict_h = fit_sin(angles, powers_h)
                func_h = fit_dict_h["fitfunc"]
                fit_dict_v = fit_sin(angles, powers_v)
                func_v = fit_dict_v["fitfunc"]

                ax = fig.add_subplot(1, 1, 1)

                ax.scatter(angles, powers_h, color="red", label="Power data (H)")
                ax.scatter(angles, powers_v, color="blue", label="Power data (V)")
                ax.plot(x_data, func_h(x_data), color="orange", label="Sinusoidal fit (H)")
                ax.plot(x_data, func_v(x_data), color="lightblue", label="Sinusoidal fit (V)")

                ax.xaxis.set_ticks(np.arange(0, 360, 10))
                labels = ax.xaxis.get_majorticklabels()
                new_labels = []
                for label in labels:
                    if float(label.get_text()) % 30 < 1:
                        new_labels.append(label)
                    else:
                        new_labels.append(label.set_text(""))
                ax.xaxis.set_ticklabels(new_labels)
                ax.xaxis.set_tick_params(labelsize=10)

                ax.set_title("Power vs Angle of Rotation " + args.name)
                ax.set_xlabel("Angle of rotation (degrees)")
                ax.set_ylabel("Laser Power (μW)")

                ax.legend()
                ax.grid()
                plt.savefig(args.file.split(".")[0] + "fit.png")

                print("zero angle v is", 90 - fit_dict_h["phase"] * (180 / (np.pi)))

                opt = fmin(lambda x: -1 * func_h(x), 180, full_output=True)
                print("max is at", opt[0], "with value", opt[1])

                exit()
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
            powers.append(meter.read)
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

    elif args.manual:
        controller = elliptec.Controller("COM4", timeout=3, write_timeout=3)
        device = elliptec.Rotator(controller, address=str(args.sv))
        wp_details = read_config()

        device.home()
        device.set_angle(wp_details[args.sv - 1]["background_angle"])

    else:
        controller = elliptec.Controller("COM4")

        device = elliptec.Rotator(controller, address="1")

        device.home()

        device.set_angle(0)
        device.set_jog_step(10)
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

        with open(
            "./data/calibrations/zero_angle/"
            + args.name
            + "/power_vs_angle_auto_"
            + args.name
            + "_"
            + datetime_now_string()
            + ".pkl",
            "wb",
        ) as f:
            pickler = pickle.Pickler(f)
            pickler.dump((angles, powers_h, powers_v))

        plt.plot(angles, powers_h, color="red", label="H")
        plt.plot(angles, powers_v, color="blue", label="V")
        plt.xlabel("Angle of rotation (degrees)")
        plt.ylabel("Laser Power (W)")
        plt.title("Power vs angle of rotation " + args.name)
        plt.legend()
        plt.savefig(
            "./data/calibrations/zero_angle/"
            + args.name
            + "/power_vs_angle_auto_"
            + args.name
            + "_"
            + datetime_now_string()
            + ".png"
        )
