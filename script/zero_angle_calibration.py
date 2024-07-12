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

NUM_MEASUREMENTS = 100


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


def connect_power_meter(apds):
    """Connect to the power meter.

    :returns: Power meter object.
    :rtype: ThorlabsPM100
    """

    if apds:
        counter = swabian.Counter()
        counter.detector_setup()
        return counter
    else:
        rm = pyvisa.ResourceManager()
        resources = rm.list_resources()

        # H_resource = [x for x in resources if 'P0027252' in x][0]
        H_resource = [x for x in resources if "P0027254" in x][0]

        # print(resources)
        inst_H = rm.open_resource(H_resource, timeout=1)
        H_meter = ThorlabsPM100(inst=inst_H)
        H_meter.configure.scalar.power()

        V_meter = None

        # inst_V = rm.open_resource(V_resource, timeout=1)
        # V_meter = ThorlabsPM100(inst=inst_V)
        # V_meter.configure.scalar.power()
        return (H_meter, V_meter)


def param_sin(x, offset, amplitude, phase, freq):
    return amplitude * np.sin(phase + freq * x) + offset


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="Manual alignment",
        description="Manually align the rotation stages to the desired angles.",
    )

    parser.add_argument(
        "-a",
        "--apds",
        default=False,
        action="store_true",
        help="Use APD readings (this will only give power vs angle plot).",
    )
    parser.add_argument(
        "-m", "--manual", default=False, action="store_true", help="Manual mode (rotate manually)"
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
        "-n", "--name", help="Name of the waveplate being calibrated (changes filename)"
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
        "-r",
        "--repeat",
        default=False,
        action="store_true",
        help="Rotate the mount by 360 degrees and check if the power remains the same.",
    )
    args = parser.parse_args()
    H_meter, V_meter = None, None

    if args.apds:
        controller = elliptec.Controller("COM4", timeout=3, write_timeout=3)
        device = elliptec.Rotator(controller, address="4")
        device.home()
        counter = connect_power_meter(args.apds)
        counter.auto_integration_time()
        h_counts = []
        v_counts = []
        angles = []
        # Repeat 4 full rotations
        for j in range(4):
            for i in range(360):
                angles.append(i + j * 360)
                angle = device.set_angle(i)
                if angle is None:
                    print("Error")
                    print("angle is none")
                    exit()
                if np.abs(angle - i) > 0.5:
                    print("ERROR")
                    print("actual angle", angle)
                    print("set angle", i)
                    exit()
                (h, v) = counter.get_counts()
                h_counts.append(h)
                v_counts.append(v)
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
            pickler.dump((angles, h_counts, v_counts))
        exit()
    # else:
    # H_meter, V_meter = connect_power_meter(args.apds)
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

        device = elliptec.Rotator(controller, address="0")

        print(device.info["Serial No."])

        device.home()

        device.set_angle(float(args.angle))
        itr_powers_h = []
        itr_powers_v = []
        for i in range(NUM_MEASUREMENTS):
            # power = power_meter.read
            itr_powers_h.append(H_meter.read)
            itr_powers_v.append(V_meter.read)
        avg_power_h = np.mean(np.array(itr_powers_h))
        avg_power_v = np.mean(np.array(itr_powers_v))
        print(device.get_angle(), avg_power_h * 10**6, avg_power_v * 10**6)
        exit()

    # Plot the data from the file
    if args.plot:
        with open(args.file, "rb") as f:
            (angles, powers, iter) = pickle.load(f)

            powers = np.array(powers)
            std_dev = np.std(powers)
            angles = np.abs(np.round(angles))

            angles = [359.99 if angle == 360 else angle for angle in angles]

            for i, angle in enumerate(angles):
                sum_0 = 0
                sum_359 = 0

                if angle == 0.0:
                    sum_0 += powers[i]
                else:
                    sum_359 += powers[i]
            mean_0 = sum_0 / (len(angles) / 2)
            mean_359 = sum_359 / (len(angles) / 2)

            angles = [str(angle) for angle in angles]

            plt.bar(["0", "359.99"], [mean_0, mean_359], yerr=std_dev)
            plt.savefig(args.file.split(".")[0] + ".png")
        exit()

    # Check repeatability of waveplate rotations
    if args.repeat:
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
                power = power_meter.read
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
                power = power_meter.read
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

    if args.home:
        controller = elliptec.Controller("COM4")
        device = elliptec.Rotator(controller, address="0")

        device.home()
        time.sleep(1)
        device.set_angle(90)
        time.sleep(1)
        device.home()

    if args.fit:
        if args.file == None:
            print("Need to specify a file to use for fitting!")
            exit()
        if args.polarizer:
            with open(args.file, "rb") as f:
                (angles, powers_h) = pickle.load(f)

                powers_h = np.array(powers_h)

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

                x_data = np.linspace(0, 360, num=1000)

                fit_dict_h = fit_sin(angles, powers_h)
                func_h = fit_dict_h["fitfunc"]
                fit_dict_v = fit_sin(angles, powers_v)
                func_v = fit_dict_v["fitfunc"]

                ax = fig.add_subplot(1, 1, 1)

                ax.scatter
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

    if args.polarizer:
        from pathlib import Path, PurePath

        powers = []
        angles = []

        for i in range(int(360 / 22.5)):
            angle = 22.5 * i
            angles.append(22.5 * i)
            print("Press enter to take data..")
            input()
            # time.sleep(3)
            powers.append(H_meter.read)
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
        # print(path)
        # print(path.parent)
        with path.open("wb") as f:
            pickler = pickle.Pickler(f)
            pickler.dump((angles, powers))
        exit()

    if args.manual:
        power = []
        for i in range(int(360 / args.step)):
            print(i)
            power.append(power_meter.read)
            time.sleep(args.time)

        print(np.arange(0, 360, args.step))
        print(power)
        plt.plot(np.arange(0, 360, args.step), power)
        plt.savefig(
            "./data/calibrations/zero_angle/"
            + args.name
            + "/power_vs_angle_manual_"
            + args.name
            + "_"
            + datetime_now_string()
            + ".png"
        )
    else:
        controller = elliptec.Controller("COM4")

        device = elliptec.Rotator(controller, address="0")

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
                itr_powers_h.append(H_meter.read)
                itr_powers_v.append(V_meter.read)
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
