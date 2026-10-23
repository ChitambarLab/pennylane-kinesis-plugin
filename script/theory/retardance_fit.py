"""Fit retardance of waveplates given waveplate sweeps, using different models for the waveplate."""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pickle
import argparse
from pathlib import Path

def qwp_mat(theta, retardance=np.pi / 2):
    c = np.cos(theta)
    s = np.sin(theta)
    e = np.exp(1.0j * retardance)

    mat = np.exp(-1.0j * retardance / 2) * np.array(
        [
            [c**2 + e * s**2, (1 - e) * c * s],
            [(1 - e) * c * s, e * c**2 + s**2],
        ]
    )
    return np.array(mat)


def hwp_mat(theta, retardance=np.pi):
    c = np.cos(theta)
    s = np.sin(theta)
    e = np.exp(1.0j * retardance)

    mat = np.exp(-1.0j * retardance / 2) * np.array(
        [
            [c**2 + e * s**2, (1 - e) * c * s],
            [(1 - e) * c * s, e * c**2 + s**2],
        ]
    )
    return np.array(mat)


def arb_mixed_state(a1, a2, a3, a4):
    L = np.array([[a1, 0], [a3 + a4 * 1.0j, a2]])
    density = L.conj().T @ L
    return density / np.trace(density)


def fit_h(
    x,
    retardance,
    amp,
    offset,
    meas_v=False,
):
    input_rho = np.array([[1, 0], [0, 0]])
    if meas_v:
        meas_rho = np.array([[0, 0], [0, 1]])
    else:
        meas_rho = input_rho

    mats = qwp_mat(x, retardance)

    repeated_input_rho = np.repeat(input_rho[:, :, np.newaxis], 360, axis=2)
    repeated_meas_rho = np.repeat(meas_rho[:, :, np.newaxis], 360, axis=2)

    out_counts_2 = np.einsum("ijk,jlk->ilk", repeated_input_rho, np.transpose(mats.conj(), (1, 0, 2)))
    out_counts_2 = np.einsum("ijk,jlk->ilk", mats, out_counts_2)
    out_counts_2 = np.einsum("ijk,jlk->ilk", repeated_meas_rho, out_counts_2)
    out_counts_2 = np.trace(out_counts_2, axis1=0, axis2=1).real * amp + np.ones(360) * offset
    return out_counts_2


def fit(x, retardance, amp, offset, meas_density, input_density):
    input_rho = input_density
    mats = qwp_mat(x, retardance)
    meas_rho = meas_density

    global bhwp

    if bhwp:
        repeated_input_rho = np.repeat(input_rho[:, :, np.newaxis], 135, axis=2)
        repeated_meas_rho = np.repeat(meas_rho[:, :, np.newaxis], 135, axis=2)
    else:
        repeated_input_rho = np.repeat(input_rho[:, :, np.newaxis], 360, axis=2)
        repeated_meas_rho = np.repeat(meas_rho[:, :, np.newaxis], 360, axis=2)

    out_counts_2 = np.einsum("ijk,jlk->ilk", repeated_input_rho, np.transpose(mats.conj(), (1, 0, 2)))
    out_counts_2 = np.einsum("ijk,jlk->ilk", mats, out_counts_2)
    out_counts_2 = np.einsum("ijk,jlk->ilk", repeated_meas_rho, out_counts_2)
    out_counts_2 = np.trace(out_counts_2, axis1=0, axis2=1).real
    return out_counts_2


def cost_fun(x, orig_h_counts, orig_v_counts):
    ret, amp_h, amp_v, off_h, off_v, zero_angle = x
    zero_angle = int(zero_angle)
    angles = np.linspace(0, np.pi * 2, 360)
    h_counts = np.array(orig_h_counts[zero_angle:]).tolist() + np.array(orig_h_counts[:zero_angle]).tolist()
    v_counts = np.array(orig_v_counts[zero_angle:]).tolist() + np.array(orig_v_counts[:zero_angle]).tolist()

    amp_h = 1
    amp_v = 1

    sim_counts_h = fit_h(angles, ret, amp_h, off_h, meas_v=False)
    sim_counts_v = fit_h(angles, ret, amp_v, off_v, meas_v=True)

    return np.sum((h_counts - sim_counts_h) ** 2) + np.sum((v_counts - sim_counts_v) ** 2)


def cost_fun_2(x, orig_h_counts, orig_v_counts):
    (
        ret,
        amp_h,
        amp_v,
        off_h,
        off_v,
        zero_angle,
        a1_mh,
        a2_mh,
        a3_mh,
        a4_mh,
        a1_mv,
        a2_mv,
        a3_mv,
        a4_mv,
    ) = x

    meas_h_density = arb_mixed_state(a1_mh, a2_mh, a3_mh, a4_mh)
    meas_v_density = arb_mixed_state(a1_mv, a2_mv, a3_mv, a4_mv)
    zero_angle = int(zero_angle)
    angles = np.linspace(0, np.pi * 2, 360)
    h_counts = np.array(orig_h_counts[zero_angle:]).tolist() + np.array(orig_h_counts[:zero_angle]).tolist()
    v_counts = np.array(orig_v_counts[zero_angle:]).tolist() + np.array(orig_v_counts[:zero_angle]).tolist()

    sim_counts_h = fit(angles, ret, amp_h, off_h, meas_h_density, np.array([[1, 0], [0, 0]]))
    sim_counts_v = fit(angles, ret, amp_v, off_v, meas_v_density, np.array([[0, 0], [0, 1]]))
    return np.sum((h_counts - sim_counts_h) ** 2) + np.sum((v_counts - sim_counts_v) ** 2)


def cost_fun_3(x, orig_h_counts, orig_v_counts):
    (
        ret,
        amp_h,
        amp_v,
        off_h,
        off_v,
        zero_angle,
        a1_i,
        a2_i,
        a3_i,
        a4_i,
    ) = x

    global bhwp
    input_density = arb_mixed_state(a1_i, a2_i, a3_i, a4_i)
    input_density = np.array([[1, 0], [0, 0]])
    zero_angle = int(zero_angle)

    h_counts = np.array(orig_h_counts[zero_angle:]).tolist() + np.array(orig_h_counts[:zero_angle]).tolist()
    v_counts = np.array(orig_v_counts[zero_angle:]).tolist() + np.array(orig_v_counts[:zero_angle]).tolist()

    if bhwp:
        h_counts = h_counts[0:135]
        v_counts = v_counts[0:135]
        angles = np.linspace(0, 3 * np.pi / 4, 135)
    else:
        angles = np.linspace(0, np.pi * 2, 360)
    amp_h = 1
    amp_v = 1
    off_h = 0
    off_v = 0

    sim_counts_h = fit(angles, ret, amp_h, off_h, np.array([[1, 0], [0, 0]]), input_density)
    sim_counts_v = fit(angles, ret, amp_v, off_v, np.array([[0, 0], [0, 1]]), input_density)
    return np.sum((h_counts - sim_counts_h) ** 2) + np.sum((v_counts - sim_counts_v) ** 2)


if __name__ == "__main__":
    global bhwp
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--file", type=str, help="file to fit")

    parser.add_argument("-q", "--qwp", action="store_true", default=False)
    parser.add_argument("--bhwp", action="store_true", default=False)
    parser.add_argument("-z", "--zero_angle", type=int, default=0)
    args = parser.parse_args()

    bhwp = args.bhwp
    path = Path(args.file)
    with open(path, "rb") as f:
        angles, orig_h_counts, orig_v_counts = pickle.load(f)

    # Get rid of extraneous inner arrays
    orig_h_counts = np.array(orig_h_counts, dtype=float).squeeze()
    orig_v_counts = np.array(orig_v_counts, dtype=float).squeeze()

    # Normalize counts

    zero_angle = args.zero_angle

    orig_h_counts = np.array(orig_h_counts)
    orig_v_counts = np.array(orig_v_counts)
    total_counts = orig_h_counts + orig_v_counts
    orig_h_counts /= total_counts
    orig_v_counts /= total_counts
    angles = np.array(angles, float).squeeze()
    angles *= np.pi / 180

    if args.qwp:
        base_retardance = np.pi / 2
    else:
        base_retardance = np.pi

    num_trials = 50
    retardances = np.zeros((num_trials,))
    amps_h = np.zeros((num_trials,))
    amps_v = np.zeros((num_trials,))
    offs_h = np.zeros((num_trials,))
    offs_v = np.zeros((num_trials,))
    zero_angles = np.zeros((num_trials,))
    costs = np.zeros((num_trials,))
    a1_mhs = np.zeros((num_trials,))
    a2_mhs = np.zeros((num_trials,))
    a3_mhs = np.zeros((num_trials,))
    a4_mhs = np.zeros((num_trials,))
    a1_mvs = np.zeros((num_trials,))
    a2_mvs = np.zeros((num_trials,))
    a3_mvs = np.zeros((num_trials,))
    a4_mvs = np.zeros((num_trials,))

    a1_is = np.zeros((num_trials,))
    a2_is = np.zeros((num_trials,))
    a3_is = np.zeros((num_trials,))
    a4_is = np.zeros((num_trials,))
    for i in range(num_trials):
        print(f"{i}th trial")
        fitted_v = sp.optimize.basinhopping(
            lambda x: cost_fun_3(x, orig_h_counts, orig_v_counts),
            [base_retardance, 1, 1, 0, 0, zero_angle, 1, 0, 0, 0],
            niter=20,
            disp=True,
            minimizer_kwargs={
                "method": "SLSQP",
            },
        )

        fitted_v.x.tolist()
        (
            ret,
            amp_h,
            amp_v,
            off_h,
            off_v,
            zero_angle,
            a1_i,
            a2_i,
            a3_i,
            a4_i,
        ) = fitted_v.x.tolist()
        costs[i] = fitted_v.fun
        retardances[i] = ret
        amps_h[i] = amp_h
        amps_v[i] = amp_v
        offs_h[i] = off_h
        offs_v[i] = off_v

        a1_is[i] = a1_i
        a2_is[i] = a2_i
        a3_is[i] = a3_i
        a4_is[i] = a4_i

        zero_angles[i] = zero_angle

    min_cost = min(costs)
    min_retardances = []
    for i, retardance in enumerate(retardances):
        print(f"Retardance: {retardance:.12f}")
        print(f"zero angle: {zero_angles[i]:.4f}")
        print(f"Cost: {costs[i]:.12f}")
        if costs[i] <= min_cost:
            min_retardances.append(retardance)

    print(f"Min cost retardances {min_retardances}")
    print(f"Min cost: {min(costs)}")
    print(f"Average retardance: {np.mean(np.array(min_retardances))}")
    print(retardances)
    print(f"Retardance:{ret}")
    print(f"Amp v:{amp_v}")
    print(f"off v:{off_v}")
    print(f"Amp h:{amp_h}")
    print(f"off h:{off_h}")

    if bhwp:
        xx = np.linspace(0, 3 * np.pi / 4, 135)
    else:
        xx = np.linspace(0, np.pi * 2, 360)

    fig, ax = plt.subplots(5, 1, figsize=(20, 25))
    axes = [axis for axis in ax]
    for i, axis in enumerate(ax):
        ret = retardances[i]
        amp_h = amps_h[i]
        amp_v = amps_v[i]
        off_h = offs_h[i]
        off_v = offs_v[i]
        zero_angle = int(zero_angles[i])
        cost = costs[i]
        a1_i = a1_is[i]
        a2_i = a1_is[i]
        a3_i = a3_is[i]
        a4_i = a4_is[i]
        amp_h = 1

        amp_v = 1
        off_h = 0
        off_v = 0
        input_density = arb_mixed_state(a1_i, a2_i, a3_i, a4_i)
        input_density = np.array([[1, 0], [0, 0]])
        off_h = 0
        off_v = 0
        yy_h = fit(xx, ret, amp_h, off_h, np.array([[1, 0], [0, 0]]), input_density)
        yy_v = fit(xx, ret, amp_v, off_v, np.array([[0, 0], [0, 1]]), input_density)
        h_counts = np.array(orig_h_counts[zero_angle:]).tolist() + np.array(orig_h_counts[:zero_angle]).tolist()
        v_counts = np.array(orig_v_counts[zero_angle:]).tolist() + np.array(orig_v_counts[:zero_angle]).tolist()
        if bhwp:
            h_counts = h_counts[0:135]
            v_counts = v_counts[0:135]
        axis.plot(xx, h_counts, label="H Counts")
        axis.plot(xx, v_counts, label="V counts")
        axis.plot(xx, yy_h, label="Fit func (H)")
        axis.plot(xx, yy_v, label="Fit func (V)")
        axis.set_title(f"{ret:.10f},cost:{cost:.10f}")

    fig.legend()

    plt.savefig("fit_angle_sweep.png")
