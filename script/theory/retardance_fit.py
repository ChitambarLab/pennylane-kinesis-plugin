import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pickle
import argparse
from pathlib import Path
from script.theory.check_retardance import simulate_angle_sweep, simulate_angle_sweep_hwp


# def cost_fun(x, h_counts,v_counts):
#    ret,amp_h, amp_v,off_h, off_v,a1_i,a2_i,a3_i,a4_i,a1_mh,a2_mh,a3_mh,a4_mh,a1_mv,a2_mv,a3_mv,a4_mv,phase = x
#    angles = np.linspace(0,np.pi,180)
#    sim_counts_h = fit_h(angles,ret,amp_h,off_h,a1_i,a2_i,a3_i,a4_i,a1_mh,a2_mh,a3_mh,a4_mh,phase)
#    sim_counts_v = fit_h(angles,ret,amp_v,off_v,a1_i,a2_i,a3_i,a4_i,a1_mv,a2_mv,a3_mv,a4_mv,phase)
#
#    return np.sum((h_counts-sim_counts_h)**2) + np.sum((v_counts - sim_counts_v)**2)
# def cost_fun(x, h_counts,v_counts):
#
#    ret,amp_h, amp_v,off_h, off_v,a1_i,a2_i,a3_i,a4_i,a1_mh,a2_mh,a3_mh,a4_mh,a1_mv,a2_mv,a3_mv,a4_mv,phase = x
#    angles = np.linspace(0,2*np.pi,360)
#    phase=0
#    #sim_counts_h = fit_h(angles,ret,amp_h,off_h,a1_i,a2_i,a3_i,a4_i,a1_mh,a2_mh,a3_mh,a4_mh,phase)
#
#    sim_counts_h = fit_h(angles,ret,amp_h,off_h,a1_i,a2_i,a3_i,a4_i,1,0,0,0,phase)
#    #sim_counts_v = fit_h(angles,ret,amp_v,off_v,a1_i,a2_i,a3_i,a4_i,a1_mv,a2_mv,a3_mv,a4_mv,phase)
#    sim_counts_v = fit_h(angles,ret,amp_v,off_v,a1_i,a2_i,a3_i,a4_i,0,1,0,0,phase)
#
#    #sim_counts_h = sim_counts_h/amp_h - off_h
#    #sim_counts_v = sim_counts_v/amp_v - off_v
#
#    return np.sum((h_counts-sim_counts_h)**2) + np.sum((v_counts-sim_counts_v)**2)
def cost_fun(x, orig_h_counts, orig_v_counts):
    # ret,amp_h, amp_v,off_h, off_v,a1_i,a2_i,a3_i,a4_i = x
    ret, amp_h, amp_v, off_h, off_v, zero_angle = x
    zero_angle = int(zero_angle)
    angles = np.linspace(0, np.pi * 2, 360)
    h_counts = (
        np.array(orig_h_counts[zero_angle:]).tolist()
        + np.array(orig_h_counts[:zero_angle]).tolist()
    )
    # h_counts = h_counts[0:90]
    v_counts = (
        np.array(orig_v_counts[zero_angle:]).tolist()
        + np.array(orig_v_counts[:zero_angle]).tolist()
    )
    # v_counts = v_counts[0:90]

    phase = 0

    a1_i = 1
    a2_i = 0
    a3_i = 0
    a4_i = 0

    # amp_h = 0.5
    # amp_v = 0.5
    off_h = 0
    off_v = 0
    # amp_v = amp_h
    # sim_counts_h = fit_h(angles,ret,amp_h,off_h,a1_i,a2_i,a3_i,a4_i,a1_mh,a2_mh,a3_mh,a4_mh,phase)

    sim_counts_h = fit_h(
        angles, ret, amp_h, off_h, a1_i, a2_i, a3_i, a4_i, 1, 0, 0, 0, phase
    )
    # sim_counts_v = fit_h(angles,ret,amp_v,off_v,a1_i,a2_i,a3_i,a4_i,a1_mv,a2_mv,a3_mv,a4_mv,phase)
    sim_counts_v = fit_h(
        angles, ret, amp_v, off_v, a1_i, a2_i, a3_i, a4_i, 0, 1, 0, 0, phase
    )

    # sim_counts_h = sim_counts_h/amp_h - off_h
    # sim_counts_v = sim_counts_v/amp_v - off_v

    return np.sum((h_counts - sim_counts_h) ** 2) + np.sum(
        (v_counts - sim_counts_v) ** 2
    )


def fit_all(x, h_counts, v_counts):
    retardance, phase, A_h, A_v, k_h, k_v = x
    sim_angle, sim_counts = simulate_angle_sweep(retardance=retardance, phase=phase)
    sim_h_counts = sim_counts[:, 0] * A_h + k_h
    sim_v_counts = sim_counts[:, 1] * A_v + k_v

    return np.sum((h_counts - sim_h_counts) ** 2) + np.sum(
        (v_counts - sim_v_counts) ** 2
    )


def fit_retardance(x, h_counts, v_counts, amplitude_h, amplitude_v, offset_h, offset_v):
    retardance, phase = x
    sim_angle, sim_counts = simulate_angle_sweep(retardance=retardance, phase=phase)
    sim_h_counts = sim_counts[:, 0] * amplitude_h + offset_h
    sim_v_counts = sim_counts[:, 1] * amplitude_v + offset_v

    return np.sum((h_counts - sim_h_counts) ** 2) + np.sum(
        (v_counts - sim_v_counts) ** 2
    )


def fit_retardance_hwp(x, h_counts, v_counts):
    retardance, phase = x
    sim_angle, sim_counts = simulate_angle_sweep_hwp(retardance=retardance, phase=phase)
    sim_h_counts = sim_counts[:, 0]
    sim_v_counts = sim_counts[:, 1]

    return np.sum((h_counts - sim_h_counts) ** 2) + np.sum(
        (v_counts - sim_v_counts) ** 2
    )


def qwp_mat(theta, retardance=np.pi / 2):
    # off_retardance = retardance * np.pi / 2

    c = np.cos(theta)
    s = np.sin(theta)
    e = np.exp(1.0j * retardance)

    # If anything is not tensorflow, it has to be casted and then

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

    # If anything is not tensorflow, it has to be casted and then

    mat = np.exp(-1.0j * retardance / 2) * np.array(
        [
            [c**2 + e * s**2, (1 - e) * c * s],
            [(1 - e) * c * s, e * c**2 + s**2],
        ]
    )
    return np.array(mat)


def arb_mixed_state(a1, a2, a3, a4):
    L = np.array([[a1, 0], [a3 + a4 * 1.0j, a2]])
    return L @ L.conj().T


def fit_h(
    x,
    retardance,
    amp,
    offset,
    a1_ih,
    a2_ih,
    a3_ih,
    a4_ih,
    a1_m,
    a2_m,
    a3_m,
    a4_m,
    phase,
):
    # input_state = np.array([1, 0])
    # input_state = np.array([input_0_amp_real+1.j*input_0_amp_imag, input_1_amp_real+1.j*input_1_amp_imag])
    # input_state /= np.linalg.norm(input_state)
    input_rho = arb_mixed_state(a1_ih, a2_ih, a3_ih, a4_ih)
    input_rho /= np.trace(input_rho)
    meas_rho = arb_mixed_state(a1_m, a2_m, a3_m, a4_m)
    meas_rho /= np.trace(meas_rho)
    mats = qwp_mat(x + phase, retardance)
    # mats_2 = qwp_mat(x+phase, retardance)
    # out_counts = np.zeros_like(x)
    # out_counts_2 = np.zeros_like(x)
    # print(input_rho[:,:,np.newaxis].shape)
    # print(np.repeat(input_rho[:,:,np.newaxis],180,axis=2).shape)
    # print(np.transpose(mats.conj(),(1,0,2)).shape)

    repeated_input_rho = np.repeat(input_rho[:, :, np.newaxis], 360, axis=2)
    repeated_meas_rho = np.repeat(meas_rho[:, :, np.newaxis], 360, axis=2)

    out_counts_2 = np.einsum(
        "ijk,jlk->ilk", repeated_input_rho, np.transpose(mats.conj(), (1, 0, 2))
    )
    out_counts_2 = np.einsum("ijk,jlk->ilk", mats, out_counts_2)
    out_counts_2 = np.einsum("ijk,jlk->ilk", repeated_meas_rho, out_counts_2)
    out_counts_2 = (
        np.trace(out_counts_2, axis1=0, axis2=1).real * amp + np.ones(360) * offset
    )
    return out_counts_2
    # out_counts_2 = np.trace(meas_rho[:,:,np.newaxis]@mats@input_rho[:,:,np.newaxis]@np.transpose(mats.conj(),(1,0,2)),axis1=0,axis2=1)*amp + offset
    # out_counts_2
    # print(out_counts_2.shape)


# out_counts_2 = np.multiply(mats,out_counts_2)
# print(mats_2.shape)
# print(meas_rho[:,:,np.newaxis].shape)
# out_counts_2 = np.multiply(meas_rho[:,:,np.newaxis],out_counts_2)
# print(out_counts_2.shape)
# out_counts_2 = np.einsum('ab,abk->abk',input_rho,mats_2)
# print(out_counts.shape)
# out_counts_2 = np.einsum('abk,abk->abk',mats_2,out_counts_2)
# print(out_counts.shape)
# out_counts_2 = np.einsum('ab,abk->abk',meas_rho,out_counts_2)
# print(out_counts.shape)
# out_counts_2 = np.einsum('iik->k',out_counts_2)
# out_counts
# out_counts_2 = np.trace(out_counts_2,axis1=0,axis2=1).real
# out_counts_2 = np.real(out_counts_2)
# print(out_counts.shape)
# out_counts_2 = out_counts_2*amp+offset
# rho_k = np.einsum('a b k, b c, c d k->a d k', mats, input_rho, mats.conj())
# out_counts = np.einsum('aa,aak->k', meas_rho, rho_k).real
# out_counts = np.real(np.einsum('ii,iij,ii,iij->j',meas_rho,mats,input_rho,np.transpose(mats.conj(),(1,0,2))))
# out_counts = np.real(np.trace(meas_rho@(mats.T@input_rho@np.transpose(mats.conj(),(1,0,2)))))
# for i in range(mats.shape[2]):
#    mat = mats[:,:,i]
#    out_counts[i] = np.real(np.trace(meas_rho@(mat@(input_rho@mat.conj().T))))

# out_counts = out_counts*amp + offset
# print(np.isclose(out_counts,out_counts_2))
# print(out_counts_2)
# print(out_counts)
# return out_counts*amp + offset
# return amp*np.real(np.trace(((qwp_mat(x,retardance)[0,0]@qwp_mat(x,retardance).conj().T[0,0]))))
# return out_counts
# return amp*(np.sin(retardance)**2*np.sin(x)**4+np.cos(retardance)**2*np.sin(x)**4+2*np.cos(retardance)*np.sin(x)**2*np.cos(x)**2+np.cos(x)**4)


# def fit_h(x, retardance,amp,offset,input_0_amp_real,input_0_amp_imag, input_1_amp_real,input_1_amp_imag, meas_0_amp_real,meas_0_amp_imag, meas_1_amp_real,meas_1_amp_imag,phase):
#    # input_state = np.array([1, 0])
#    input_state = np.array([input_0_amp_real+1.j*input_0_amp_imag, input_1_amp_real+1.j*input_1_amp_imag])
#    input_state /= np.linalg.norm(input_state)
#    input_rho = np.outer(input_state, input_state.conj().T)
#    meas_state = np.array([meas_0_amp_real+1.j*meas_0_amp_imag, meas_1_amp_real+1.j*meas_1_amp_imag])
#    meas_state /= np.linalg.norm(meas_state)
#    meas_rho = np.outer(meas_state,meas_state.conj().T)
#    mats = qwp_mat(x+phase,retardance).T
#    out_counts = np.zeros_like(x)
#    for i,mat in enumerate(mats):
#        out_counts[i] = np.real(np.trace(meas_rho@(mat@input_rho@mat.conj().T)))
#    return amp*out_counts+offset


def fit_v(x, retardance, amp, offset, input_0_amp, input_1_amp, meas_0_amp, meas_1_amp):
    # input_state = np.array([1, 0])
    input_state = np.array([input_0_amp, input_1_amp])
    input_state /= np.linalg.norm(input_state)
    input_rho = np.outer(input_state, input_state.conj().T)
    meas_state = np.array([meas_0_amp, meas_1_amp])
    meas_state /= np.linalg.norm(meas_state)
    meas_rho = np.outer(meas_state, meas_state.conj().T)
    mats = qwp_mat(x, retardance).T
    out_counts = np.zeros_like(x)
    for i, mat in enumerate(mats):
        out_counts[i] = np.real(np.trace(meas_rho @ (mat @ input_rho @ mat.conj().T)))
    return amp * out_counts + offset


def fit_sin(x, A, phase, freq, offset):
    return A * np.sin(freq * x + phase) + offset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--file", type=str, help="file to fit")

    parser.add_argument("-q", "--qwp", action="store_true", default=False)
    parser.add_argument("-z", "--zero_angle", type=int, default=0)
    args = parser.parse_args()

    path = Path(args.file)
    with open(path, "rb") as f:
        angles, orig_h_counts, orig_v_counts = pickle.load(f)

    # Get rid of extraneous inner arrays
    orig_h_counts = np.array(orig_h_counts, dtype=float).squeeze()
    orig_v_counts = np.array(orig_v_counts, dtype=float).squeeze()

    # Normalize counts

    zero_angle = args.zero_angle

    # h_counts = (
    #     np.array(h_counts[zero_angle:]).tolist()
    #     + np.array(h_counts[:zero_angle]).tolist()
    # )
    # # h_counts = h_counts[0:90]
    # v_counts = (
    #     np.array(v_counts[zero_angle:]).tolist()
    #     + np.array(v_counts[:zero_angle]).tolist()
    # )
    # v_counts = v_counts[0:90]
    orig_h_counts = np.array(orig_h_counts)
    orig_v_counts = np.array(orig_v_counts)
    total_counts = orig_h_counts + orig_v_counts
    # h_counts = h_counts[211:211+180]
    # v_coutns= v_counts[211:211+180]
    orig_h_counts /= total_counts
    orig_v_counts /= total_counts
    angles = np.array(angles, float).squeeze()
    angles *= np.pi / 180
    # angles = angles[0:90]
    print(angles)
    print(orig_h_counts)
    # fitted_h = curve_fit(fit_sin, angles, h_counts, p0=[400000,1,3.5,10])
    # fitted_h = curve_fit(fit_h, angles, h_counts, p0=[3.5,400000])
    # print(fitted_h[0])
    # A_h, phase_h, freq_h, offset_h = fitted_h[0]
    # print(f"H amplitude: {A_h}")
    # print(f"H offset: {offset_h}")

    # retardance, amplitude, offset, input_0_real, input_0_imag, input_1_real, input_1_imag,meas_0_real, meas_0_imag, meas_1_real, meas_1_imag,
    # fitted_h = curve_fit(fit_h, angles, h_counts, p0=[np.pi/2,1,0,1,0,0,0,1,0,0,0,0],max_nfev=100000,method="trf")
    # ret_h,amp_h, off_h,input_0_amp_h_real,input_0_amp_h_imag, input_1_amp_h_real,input_1_amp_h_imag, meas_0_amp_h_real,meas_0_amp_h_imag, meas_1_amp_h_real,meas_1_amp_h_imag,phase_h = fitted_h[0]

    # fitted_v = curve_fit(fit_h, angles, v_counts, p0=[np.pi/2,1,0,1,0,0,0,0,0,1,0,0],max_nfev=100000,method="trf")
    # ret_v,amp_v, off_v,input_0_amp_v_real,input_0_amp_v_imag, input_1_amp_v_real,input_1_amp_v_imag, meas_0_amp_v_real,meas_0_amp_v_imag, meas_1_amp_v_real,meas_1_amp_v_imag,phase_v = fitted_v[0]
    # ret,amp_h, amp_v,off_h, off_v,input_0_amp_real,input_0_amp_imag, input_1_amp_real,input_1_amp_imag,meas_0_amp_h_real,meas_0_amp_h_imag,meas_1_amp_h_real,meas_1_amp_h_imag, meas_0_amp_v_real,meas_0_amp_v_imag, meas_1_amp_v_real,meas_1_amp_v_imag,phase = x

    # fitted_v = sp.optimize.basinhopping(
    #    lambda x: cost_fun(x, h_counts,v_counts),
    #    #ret, amp_h, amp_v, off_h,off_v, a1_i,a2_i,a3_i,a4_i,a1_mh,a2_mh,a3_mh,a4_mh,a1_mv,a2_mv,a3_mv,a4_mv,phase
    #    [np.pi/2,1,1,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0],
    #    niter=10,
    #    disp=True,
    #    minimizer_kwargs= {
    #        'bounds':[[0.5*np.pi/2,1.5*np.pi/2],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None]],
    #    }
    # )

    if args.qwp:
        base_retardance = np.pi / 2
    else:
        base_retardance = np.pi
    # fitted_v = minimize(
    #    lambda x: cost_fun(x, h_counts,v_counts),
    #    [base_retardance,1,1,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0],
    #    bounds=[[0.94*base_retardance,1.06*base_retardance],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None]],
    #    method="Powell"
    #
    num_trials = 100
    retardances = np.zeros((num_trials,))
    amps_h = np.zeros((num_trials,))
    amps_v = np.zeros((num_trials,))
    offs_h = np.zeros((num_trials,))
    offs_v = np.zeros((num_trials,))
    a1_is = np.zeros((num_trials,))
    a2_is = np.zeros((num_trials,))
    a3_is = np.zeros((num_trials,))
    a4_is = np.zeros((num_trials,))
    zero_angles = np.zeros((num_trials,))
    costs = np.zeros((num_trials,))
    for i in range(num_trials):
        print(f"{i}th trial")
        # fitted_v = sp.optimize.basinhopping(
        #    lambda x: cost_fun(x, h_counts,v_counts),
        #    [0.95*base_retardance,1,1,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0],
        #    niter=100,
        #    disp=True,
        #    minimizer_kwargs={
        #        'bounds':[[0.95*base_retardance,1.05*base_retardance],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None]],
        #        #'method':"Powell",
        #        'method':"SLSQP",
        #
        #    }
        # )
        # fitted_v = sp.optimize.basinhopping(
        #    lambda x: cost_fun(x, h_counts,v_counts),
        #    [0.95*base_retardance,1,1,0,0,1,0,0,0],
        #    niter=100,
        #    disp=True,
        #    minimizer_kwargs={
        #        'bounds':[[0.95*base_retardance,1.05*base_retardance],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None]],
        #        'method':"Powell",
        #        #'method':"SLSQP",
        #
        #    }
        # )
        fitted_v = sp.optimize.basinhopping(
            lambda x: cost_fun(x, orig_h_counts, orig_v_counts),
            [base_retardance, 1, 1, 0, 0, zero_angle],
            niter=100,
            disp=True,
            minimizer_kwargs={
                #'bounds':[[0.95*base_retardance,1.05*base_retardance],[None,None],[None,None],[None,None],[None,None]],
                "method": "Powell",
                # "method": "SLSQP",
            },
        )

        # ret,amp_h, amp_v,off_h, off_v,a1_i,a2_i,a3_i,a4_i,a1_mh,a2_mh,a3_mh,a4_mh,a1_mv,a2_mv,a3_mv,a4_mv,phase = fitted_v.x.tolist()
        # ret,amp_h, amp_v,off_h, off_v,a1_i,a2_i,a3_i,a4_i = fitted_v.x.tolist()
        ret, amp_h, amp_v, off_h, off_v, zero_angle = fitted_v.x.tolist()
        costs[i] = fitted_v.fun
        retardances[i] = ret
        amps_h[i] = amp_h
        amps_v[i] = amp_v
        offs_h[i] = off_h
        offs_v[i] = off_v
        zero_angles[i] = zero_angle
        # a1_is[i] = a1_i
        # a2_is[i] = a2_i
        # a3_is[i] = a3_i
        # a4_is[i] = a4_i

    # print(fitted_v)
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
    # print(f"Cost at pi: {
    print(f"Average retardance: {np.mean(np.array(min_retardances))}")
    print(retardances)
    # ret,amp_h, amp_v,off_h, off_v,a1_i,a2_i,a3_i,a4_i,a1_mh,a2_mh,a3_mh,a4_mh,a1_mv,a2_mv,a3_mv,a4_mv,phase = fitted_v.x.tolist()
    # print(f"Retardance v:{ret_v}")
    # print(f"Amp v:{amp_v}")
    # print(f"off v:{off_v}")
    # print(f"Retardance h:{ret_h}")
    # print(f"Amp h:{amp_h}")
    # print(f"off h:{off_h}")

    print(f"Retardance:{ret}")
    print(f"Amp v:{amp_v}")
    print(f"off v:{off_v}")
    print(f"Amp h:{amp_h}")
    print(f"off h:{off_h}")

    # input_state_h = np.array([input_0_amp_real+1.j*input_0_amp_imag,input_1_amp_real+1.j*input_1_amp_imag])
    # input_state_h /= np.linalg.norm(input_state_h)

    # meas_state_h = np.array([meas_0_amp_h_real+1.j*meas_0_amp_h_imag,meas_1_amp_h_real+1.j*meas_1_amp_h_imag])
    # meas_state_h /= np.linalg.norm(meas_state_h)

    # input_state_v = np.array([input_0_amp_v_real+1.j*input_0_amp_v_imag,input_1_amp_v_real+1.j*input_1_amp_v_imag])
    # input_state_v /= np.linalg.norm(input_state_v)

    # meas_state_v = np.array([meas_0_amp_v_real+1.j*meas_0_amp_v_imag,meas_1_amp_v_real+1.j*meas_1_amp_v_imag])
    # meas_state_v /= np.linalg.norm(meas_state_v)
    a1_i = 1
    a2_i = 0
    a3_i = 0
    a4_i = 0
    input_rho = arb_mixed_state(a1_i, a2_i, a3_i, a4_i)
    input_rho /= np.trace(input_rho)

    a1_mh = 1
    a2_mh = 0
    a3_mh = 0
    a4_mh = 0

    a1_mv = 0
    a2_mv = 1
    a3_mv = 0
    a4_mv = 0

    meas_rho_h = arb_mixed_state(a1_mh, a2_mh, a3_mh, a4_mh)
    meas_rho_h /= np.trace(meas_rho_h)

    meas_rho_v = arb_mixed_state(a1_mv, a2_mv, a3_mv, a4_mv)
    meas_rho_v /= np.trace(meas_rho_v)

    print(f"Input state h:[{input_rho}]")
    # print(f"Input state v:[{input_state_v}]")

    print(f"meas state h:[{meas_rho_h}]")
    print(f"meas state v:[{meas_rho_v}]")

    phase = 0
    # print(f"phase H:{phase_h}")
    print(f"phase:{phase}")
    # print(f"phase V:{phase_v}")

    # fitted_v = curve_fit(fit_sin, angles, v_counts, p0=[400000,1,3.5,100000])

    xx = np.linspace(0, np.pi * 2, 360)
    # yy_h = fit_h(xx, ret_h, amp_h,off_h,input_0_amp_h_real,input_0_amp_h_imag, input_1_amp_h_real,input_1_amp_h_imag, meas_0_amp_h_real,meas_0_amp_h_imag, meas_1_amp_h_real,meas_1_amp_h_imag,phase_h)
    yy_h = fit_h(
        xx, ret, amp_h, off_h, a1_i, a2_i, a3_i, a4_i, a1_mh, a2_mh, a3_mh, a4_mh, phase
    )

    # yy_h = fit_sin(xx, A_h, phase_h, freq_h,offset_h)

    # A_v, phase_v, freq_v, offset_v = fitted_v[0]
    # print(f"V amplitude: {A_v}")
    # print(f"V offset: {offset_v}")

    # yy_v = fit_sin(xx, A_v, phase_v, freq_v,offset_v)
    # yy_v = fit_h(xx, ret_v, amp_v,off_v,input_0_amp_v_real,input_0_amp_v_imag, input_1_amp_v_real,input_1_amp_v_imag, meas_0_amp_v_real,meas_0_amp_v_imag, meas_1_amp_v_real,meas_1_amp_v_imag,phase_v)
    yy_v = fit_h(
        xx, ret, amp_v, off_v, a1_i, a2_i, a3_i, a4_i, a1_mv, a2_mv, a3_mv, a4_mv, phase
    )

    print(yy_h)
    print(orig_h_counts)

    fig, ax = plt.subplots(5, 1, figsize=(20, 25))
    axes = [axis for axis in ax]
    phase = 0
    for i, axis in enumerate(ax):
        ret = retardances[i]
        amp_h = amps_h[i]
        amp_v = amps_v[i]
        off_h = offs_h[i]
        off_v = offs_v[i]
        a1_i = a1_is[i]
        a2_i = a2_is[i]
        a3_i = a3_is[i]
        a4_i = a4_is[i]
        zero_angle = int(zero_angles[i])
        a1_i = 1
        a2_i = 0
        a3_i = 0
        a4_i = 0
        off_h = 0
        off_v = 0
        # amp_v = amp_h
        yy_h = fit_h(
            xx,
            ret,
            amp_h,
            off_h,
            a1_i,
            a2_i,
            a3_i,
            a4_i,
            a1_mh,
            a2_mh,
            a3_mh,
            a4_mh,
            phase,
        )
        yy_v = fit_h(
            xx,
            ret,
            amp_v,
            off_v,
            a1_i,
            a2_i,
            a3_i,
            a4_i,
            a1_mv,
            a2_mv,
            a3_mv,
            a4_mv,
            phase,
        )
        h_counts = (
            np.array(orig_h_counts[zero_angle:]).tolist()
            + np.array(orig_h_counts[:zero_angle]).tolist()
        )
        # h_counts = h_counts[0:360]
        v_counts = (
            np.array(orig_v_counts[zero_angle:]).tolist()
            + np.array(orig_v_counts[:zero_angle]).tolist()
        )
        # v_counts = v_counts[0:90]
        axis.plot(angles, h_counts, label="H Counts")
        axis.plot(angles, v_counts, label="V counts")
        axis.plot(xx, yy_h, label="Fit func (H)")
        axis.plot(xx, yy_v, label="Fit func (V)")
        axis.set_title(f"{ret:.7f}")

    # axes[0].plot(angles, h_counts,label= "H Counts")
    # axes[0].plot(angles, v_counts, label="V counts")
    # axes[0].plot(xx, yy_h, label="Fit func (H)")
    # axes[0].plot(xx, yy_v, label="Fit func (V)")
    out_h = []
    out_v = []

    # for retardance in retardances:
    #    out_h.append(fit_h(np.array([np.pi/4]), retardance, amp_v,off_v,a1_i,a2_i,a3_i,a4_i,a1_mv,a2_mv,a3_mv,a4_mv,phase))
    #    out_v.append(fit_h(np.array([np.pi/4]),retardance,amp_v,off_v,a1_i,a2_i,a3_i,a4_i,a1_mv,a2_mv,a3_mv,a4_mv,phase))

    # print(out_h)
    # print(out_v)
    # axes[1].plot(xx, out_h,label="H counts")
    # axes[1].plot(xx,out_v,label="V counts")

    # print(f"H phase: {phase_h%(2*np.pi)}")
    # print(f"V phase: {phase_v%(2*np.pi)}")

    # if args.qwp:
    # res = minimize(
    # lambda x: fit_retardance(x, h_counts,v_counts,A_h,A_v,offset_h,offset_v),
    # [1,0],
    # )
    # sim_angle, counts = simulate_angle_sweep(res.x[0],res.x[1])
    # res = minimize(
    #     lambda x: fit_all(x, h_counts,v_counts),
    #     [1,0,1,1,1,1],
    #     bounds=[[-3,3],[-3*np.pi,3*np.pi],[-1e6,1e6],[-1e6,1e6],[0,1e6],[0,1e6]],
    #     method="L-BFGS-B",
    #     options={
    #         "max_iter": 50000,
    #         "ftol":1e-8,
    #         "maxcor":10,
    #         # "eps":10,
    #         "maxls": 50,
    #         "disp": True
    #     }
    # )
    # res = sp.optimize.fmin_l_bfgs_b(
    #     lambda x: fit_all(x, h_counts,v_counts),
    #     # [1,1,A_h,A_v,offset_h,offset_v],
    #     [np.pi/2,0,1,1,0,0],
    #     # bounds=[[None,None],[-15*np.pi,15*np.pi],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6]],
    #     factr = 1e-2,
    #     approx_grad=True,
    #     m=1000,
    #     epsilon=1e-10,
    #     maxls=1000,
    #     )
    #     # options={
    #     #     "max_iter": 50000,
    #     #     "ftol":1e-8,
    #     #     "maxcor":10,
    #     #     # "eps":10,
    #     #     "maxls": 50,
    #     #     "disp": True
    #     # })
    # print(res)
    # vals = res[0]
    #   sim_angle, counts = simulate_angle_sweep(vals[0],vals[1])
    # else:
    #   res = minimize(
    #      lambda x: fit_retardance_hwp(x, h_counts,v_counts,A_h,A_v,offset_h,offset_v),
    #     [1,0],
    # )
    # sim_angle, counts = simulate_angle_sweep_hwp(res.x[0],res.x[1])

    # print(res)
    # print(res.x)

    # print(f"Retardance:{vals[0]}")
    # print(f"Phase:{vals[1]}")
    # print(f"Amplitude H:{vals[2]}")
    # print(f"Amplitude V: {vals[3]}")
    # print(f"Offset H: {vals[4]}")
    # print(f"Offset V:{vals[5]}")

    # sim_h_counts = counts[:,0]*vals[2]+vals[4]
    # sim_v_counts = counts[:,1]*vals[3]+vals[5]

    # sim_h_counts = counts[:,0]*A_h+offset_h
    # sim_v_counts = counts[:,1]*A_v+offset_v
    # print(angles)

    # print(sim_angle)

    # axes[1].plot(angles, sim_h_counts, label="Simulated H counts")
    # axes[1].plot(angles, sim_v_counts, label="Simulated V counts")

    fig.legend()

    plt.savefig("fit_angle_sweep.png")
