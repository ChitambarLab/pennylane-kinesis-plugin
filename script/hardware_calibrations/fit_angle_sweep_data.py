import scipy
import numpy as np
import argparse
from pennylane_kinesis.file_utilities import datetime_now_string, read_config, get_date_from_filename
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

def param_sin(x, offset, amplitude, phase, freq):
    return amplitude * np.sin(phase + freq * x) + offset

def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(tt, dtype=float)
    yy = np.array(yy, dtype=float)
    ff = np.fft.fftfreq(len(tt), (tt[1] - tt[0]))  # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(
        ff[np.argmax(Fyy[1:]) + 1],
    )  # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.0**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.0 * np.pi * guess_freq, 0.0, guess_offset])

    def sinfunc(t, A, w, p, c):
        return A * np.sin(w * t + p) + c

    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w / (2.0 * np.pi)
    fitfunc = lambda t: A * np.sin(w * t + p) + c
    return {
        "amp": A,
        "omega": w,
        "phase": p,
        "offset": c,
        "freq": f,
        "period": 1.0 / f,
        "fitfunc": fitfunc,
        "maxcov": np.max(pcov),
        "rawres": (guess, popt, pcov),
        "perr": np.sqrt(np.diag(pcov)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Fit angle sweep.",
        description="Fit angle sweep data for mounted waveplate to calibrate zero angle. Use zero_angle_calibration.py to get data files.",
    )

    parser.add_argument("--mount", "-m", default=0, type=int, help="Which mount to set/get velocity for.")

    parser.add_argument(
        "-f",
        "--file",
        default=None,
        help="File to use for fitting data. (with --plot or --fit option)",
    )
    parser.add_argument(
        "--polarizer",
        default=False,
        action="store_true",
        help="Fitting a linear polarizer data instead.",
    )

    parser.add_argument(
        "-p",
        "--plot",
        default=False,
        action="store_true",
        help="Plot the data from the file (no fitting)."
    )

    parser.add_argument(
        "-n",
        "--name",
        default="None",
        help="Name of the waveplate (for the plot)."
    )

    args = parser.parse_args()
    path = Path(args.file)

    # Plot the data from the file
    if args.plot:
        with open(path, "rb") as f:
            (angles, h_counts, v_counts) = pickle.load(f)

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

            wp_details = read_config()

            zero_angle = wp_details[int(args.apds) - 1]["angle_offset"]

            print(np.array(angles).shape)
            print(np.array(h_counts).shape)
            print(np.array(v_counts).shape)
            # angles = angles[0:72]
            if args.average:
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
                ax.axvline(x=zero_angle, color="black", linestyle="--", label="Zero angle")
                plt.fill_between(
                    angles,
                    sum_mean_counts - std_dev_sum_counts,
                    sum_mean_counts + std_dev_sum_counts,
                    color="darkgreen",
                    alpha=0.3,
                )
            else:
                ax.plot(
                    angles,
                    np.array(h_counts) + np.array(v_counts),
                    color="green",
                    label="Total",
                )
                ax.plot(angles, h_counts, color="red", label="H")
                ax.plot(angles, v_counts, color="blue", label="V")
                ax.axvline(x=zero_angle, color="black", linestyle="--", label="Zero angle")
            ax.xaxis.set_ticks(np.arange(0, 360, 10))
            # ax.axvline(x=204, color="r", linestyle="--")
            # ax.axvline(x=294, color="r", linestyle="--")
            # ax.axvline(x=204 - 180, color="b", linestyle="--")
            labels = ax.xaxis.get_majorticklabels()
            new_labels = []
            for label in labels:
                if float(label.get_text()) % 30 < 1:
                    new_labels.append(label)
                else:
                    new_labels.append(label.set_text(""))
            ax.xaxis.set_ticklabels(new_labels)
            ax.xaxis.set_tick_params(labelsize=10)

            # ax.set_yscale("log")
            ax.set_xlabel("Angle of rotation (degrees)")
            ax.set_ylabel("Photon Counts")
            ax.set_title("Photon counts vs angle of rotation " + args.name)
            ax.legend()
            save_location = path.parents[0]
            datetime = get_date_from_filename(path.name)
            save_path = Path(
                str(save_location)
                +"power_vs_angle_auto_"
                + args.name
                + "_"
                + datetime.strftime("%Y-%m-%dT%H-%M-%SZ")
                + ".png"
            )
            plt.savefig(save_path)
        exit()

     # Fit existing data
    else:
        if args.file == None:
            print("Need to specify a file to use for fitting!")
            exit()
        # Plot fit for a polarizer (with manual rotation)
        if args.polarizer:
            with open(args.file, "rb") as f:
                (angles, powers) = pickle.load(f)
                print(powers)
                powers = np.array(powers)
                powers_h = powers[:, 0]
                powers_v = powers[:, 1]

                # powers_h = np.array(powers_h)
                print(len(powers_h))
                fig = plt.figure()
                x_data = np.linspace(0, 360, num=1000)
                fit_dict_h = fit_sin(angles, powers_h)
                func_h = fit_dict_h["fitfunc"]
                ax = fig.add_subplot(1, 1, 1)

                # ax.scatter
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
                # print(args.file.split('.'))
                save_location = path.parents[0]
                datetime = get_date_from_filename(path.name)
                save_path = Path(
                    str(save_location)
                    +"/"
                    + datetime.strftime("%Y-%m-%dT%H-%M-%SZ")
                    + "_fit.png"
                )
                plt.savefig(save_path)

                print("zero angle v is", 90 - fit_dict_h["phase"] * (180 / (np.pi)))

                opt = np.fmin(lambda x: -1 * func_h(x), 90, full_output=True)
                print("max is at", opt[0], "with value", opt[1])

                exit()

        else:
            with open(args.file, "rb") as f:
                (angles, powers_h, powers_v) = pickle.load(f)

                powers_h = np.array(powers_h, dtype=float).flatten()
                powers_v = np.array(powers_v, dtype=float).flatten()
                # powers_h = (powers_h - np.min(powers_h)) / (np.max(powers_h) - np.min(powers_h))
                # powers_v = (powers_v - np.min(powers_v)) / (np.max(powers_v) - np.min(powers_v))

                print(angles)
                print(powers_h)
                print(powers_h + powers_v)

                print(len(angles))
                print(len)
                fig = plt.figure()
                angles = np.array(angles, dtype=float)
                # angles = angles[45:130]
                # powers_h = powers_h[45:130]
                # powers_v = powers_v[45:130]
                x_data = np.linspace(0, 360, num=1000)

                fit_dict_h = fit_sin(angles, powers_h)
                func_h = fit_dict_h["fitfunc"]
                fit_dict_v = fit_sin(angles, powers_v)
                func_v = fit_dict_v["fitfunc"]

                ax = fig.add_subplot(1, 1, 1)

                ax.scatter(angles, powers_h, color="red", label="Power data (H)")
                ax.scatter(angles, powers_v, color="blue", label="Power data (V)")
                ax.plot(x_data, func_h(x_data), color="orange", label="Sinusoidal fit (H)")
                ax.plot(
                    x_data,
                    func_v(x_data),
                    color="lightblue",
                    label="Sinusoidal fit (V)",
                )

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
                save_location = path.parents[0]
                datetime = get_date_from_filename(path.name)
                save_path = Path(
                    str(save_location)
                    +"/"
                    + datetime.strftime("%Y-%m-%dT%H-%M-%SZ")
                    + "_fit.png"
                )
                plt.savefig(save_path)

                print("zero angle v is", 90 - fit_dict_h["phase"] * (180 / (np.pi)))

                opt = scipy.optimize.fmin(lambda x: -1 * func_h(x), 180, full_output=True)
                print("max is at", opt[0], "with value", opt[1])

                exit()
