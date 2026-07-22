import numpy as np
import scipy as sp
from scipy.optimize import minimize
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

def get_wedge_val_interpd(wedge_loc, wedge_val, wedge_num, x):
    xp = np.linspace(0,2*np.pi, wedge_num)
    fp = np.ones_like(xp)
    fp[wedge_loc] = wedge_val
    return np.interp(x, xp, fp)

def waveplate(theta, retardance=np.pi):
    cos = np.cos(theta)
    sin = np.sin(theta)
    e = np.exp(1.j*retardance)

    mat = np.exp(-1.j*retardance/2)*np.array([
        [cos**2 + e*sin**2, (1-e)*cos*sin],
        [(1-e)*cos*sin, sin**2+e*cos**2]]
        )
    return mat

def get_meas_result(x):
    start_state = np.array([1,0],dtype=np.complex128)

    # retardance, wedge_loc, wedge_num, wedge_val, offset = x
    retardance, offset = x[:2]
    wedge_vals = x[2:]

    # wedge_num = int(wedge_num)
    # wedge_loc = int(wedge_loc)

    angles = np.linspace(0,2*np.pi,360)
    offset = np.ones_like(angles)*offset
    angles = angles+offset
    wps = waveplate(angles, retardance)
    wps = np.transpose(wps,axes=[2,0,1])
    # wedge = get_wedge_val_interpd(wedge_loc, wedge_val, wedge_num, angles)
    # print(wedge)
    wedge = wedge_vals
    fit_counts = np.zeros_like(angles)
    for i,wp in enumerate(wps):
        # print(wp)
        # print(wedge[i])
        state = wedge[i]*wp@start_state
        density = np.outer(state, state.conj())
        fit_counts[i] = density[0,0].real
    return fit_counts

def fit_func(x, counts_h):
    # print("loc",wedge_loc)
    # print("num",wedge_num)
    # if int(wedge_loc) >= int(wedge_num):
    #     print('return inf')
    #     return np.inf

    fit_counts = get_meas_result(x)
    return np.sum(np.abs(fit_counts - counts_h)**2)

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

    start_retardance = np.pi/2
    start_wedge_loc = 4
    start_wedge_num = 10
    start_wedge_val = 0.6
    start_offset = 0

    x0 = [start_retardance,start_offset]
    wedge_vals = np.ones(360)

    x0 = x0+wedge_vals.tolist()

    # (angles, powers_h, powers_v) = pickle.load(f)
    with open(path, "rb") as f:
        counts = pickle.load(f)
    counts = np.array(counts)
    # norm = counts[:,:,0] + counts[:,:,1]
    # norm_counts = counts
    # norm_counts[:,:,0] /= norm
    # norm_counts[:,:,1] /= norm

    mean_data = np.mean(counts, axis=0)/100000
    # std_data = np.std(norm_counts,axis=0)

    mean_data_h = mean_data[:,0]

    res = minimize(lambda x: fit_func(x, mean_data_h),x0,options={'disp': True})

    print(res.x)

    angles = np.linspace(0,360,360)
    mean_data_v = mean_data[:,1]

    # powers_h = (powers_h - np.min(powers_h)) / (np.max(powers_h) - np.min(powers_h))
    # powers_v = (powers_v - np.min(powers_v)) / (np.max(powers_v) - np.min(powers_v))

    fig = plt.figure()
    # angles = angles[45:130]
    # powers_h = powers_h[45:130]
    # powers_v = powers_v[45:130]


    # fit_dict_h = fit_sin(angles, mean_data_h)
    # func_h = fit_dict_h["fitfunc"]
    # fit_dict_v = fit_sin(angles, mean_data_v)
    # func_v = fit_dict_v["fitfunc"]
    fit_counts = get_meas_result(res.x)
    axes = fig.subplots(2,1)
    ax = axes[0]
    ax1 = axes[1]
    # ax = fig.add_subplot(1, 1, 1)

    ax.scatter(angles, mean_data_h, color="red", label="P(H)")
    ax.scatter(angles, mean_data_v, color="blue", label="P(V)")
    ax.plot(angles, fit_counts, color="orange", label="Sinusoidal fit (H)")

    ax1.plot(angles,res.x[2:],label="Wedge values")
    # ax.plot(
    #     x_data,
    #     func_v(x_data),
    #     color="lightblue",
    #     label="Sinusoidal fit (V)",
    # )

    # ax.xaxis.set_ticks(np.arange(angle_cutoffs[0], angle_cutoffs[1], 10))
    # labels = ax.xaxis.get_majorticklabels()
    # new_labels = []
    # for label in labels:
    #     if float(label.get_text()) % 30 < 1:
    #         new_labels.append(label)
    #     else:
    #         new_labels.append(label.set_text(""))
    # ax.xaxis.set_ticklabels(new_labels)
    # ax.xaxis.set_tick_params(labelsize=10)

    ax.set_title("Normalized Counts vs Angle of Rotation " + args.name)
    ax.set_xlabel("Angle of rotation (degrees)")
    ax.set_ylabel("Normalized Detection Probability")

    ax.legend()
    ax.grid()
    save_location = path.parents[0]
    datetime = get_date_from_filename(path.name)
    save_path = Path(
        str(save_location)
        +"/"
        + datetime.strftime("%Y-%m-%dT%H-%M-%SZ")
        + "_fit_retardance.png"
    )
    plt.savefig(save_path)

    # print("zero angle v is", 90 - fit_dict_h["phase"] * (180 / (np.pi)))

    # opt = scipy.optimize.fmin(lambda x: -1 * func_h(x), -5, full_output=True)
    # print("max is at", opt[0], "with value", opt[1])

    exit()
