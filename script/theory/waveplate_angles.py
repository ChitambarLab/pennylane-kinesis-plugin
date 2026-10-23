"""Calculate the required waveplate angles needed to prepare/measure polarization states (for process tomography)."""

import numpy as np


def HWP(theta):
    return np.exp(-1.0j * np.pi / 2) * np.array(
        [
            [np.cos(theta) ** 2 - np.sin(theta) ** 2, 2 * np.cos(theta) * np.sin(theta)],
            [2 * np.cos(theta) * np.sin(theta), np.sin(theta) ** 2 - np.cos(theta) ** 2],
        ]
    )


def QWP(theta):
    return np.exp(-1.0j * np.pi / 4) * np.array(
        [
            [
                np.cos(theta) ** 2 + 1j * np.sin(theta) ** 2,
                (1 - 1j) * np.cos(theta) * np.sin(theta),
            ],
            [
                (1 - 1j) * np.cos(theta) * np.sin(theta),
                np.sin(theta) ** 2 + 1j * np.cos(theta) ** 2,
            ],
        ]
    )


# Note that operators are applied from right to left, so the order of the operators in the return statement is reversed.


# Quarter Waveplate then Half Waveplate (Typical state preparation order for QST)
# Angle 1 is the angle of the HWP, Angle 2 is the angle of the QWP
def qwp_hwp(theta1, theta2):
    angle1 = (theta1 % 90) * np.pi / 180
    angle2 = (theta2 % 180) * np.pi / 180
    return HWP(angle1) @ QWP(angle2)


# Half Waveplate then Quarter Waveplate (Backward ordering for QST)
# This is also the correct order for measurements in a QPT setup.
# Angle 1 is the angle of the HWP, Angle 2 is the angle of the QWP
def hwp_qwp(theta1, theta2):
    angle1 = (theta1 % 90) * np.pi / 180
    angle2 = (theta2 % 180) * np.pi / 180
    return QWP(angle2) @ HWP(angle1)


# Defining polarization states and density matrices
H = np.array([1, 0], dtype=complex)
V = np.array([0, 1], dtype=complex)
D = 1 / np.sqrt(2) * (H + V)
A = 1 / np.sqrt(2) * (H - V)
R = 1 / np.sqrt(2) * (H + 1.0j * V)
L = 1 / np.sqrt(2) * (H - 1.0j * V)

H_rho = np.outer(H, H.conj().T)
V_rho = np.outer(V, V.conj().T)
D_rho = np.outer(D, D.conj().T)
A_rho = np.outer(A, A.conj().T)
R_rho = np.outer(R, R.conj().T)
L_rho = np.outer(L, L.conj().T)

state_vecs = [H, V, D, A, R, L]
states = [H_rho, V_rho, D_rho, A_rho, R_rho, L_rho]
names = ["H", "V", "D", "A", "R", "L"]

# Calcualte angles for each state preparation
prep_angles = []
for i in range(len(names)):
    found_angles = False
    print(f"Finding correct angles to prepare {names[i]} using HWP->QWP ordering.")
    # Loop through all possible angles for HWP and QWP
    for angle1 in np.arange(-360, 360, 22.5):
        for angle2 in np.arange(-360, 360, 22.5):
            # Prepare the state (assuming horizontally polarized input H_rho)
            prepared_state = hwp_qwp(angle1, angle2) @ H
            prepared_state = np.outer(prepared_state, prepared_state.conj())
            prepared_state /= np.trace(prepared_state)

            expected_value = state_vecs[i].conj().T @ prepared_state @ state_vecs[i]
            # Check if the prepared state matches the state we are looking for
            if np.isclose(expected_value, 1):
                print(f"HWP: {angle1 % 90}, QWP: {angle2 % 180}")
                found_angles = True
                prep_angles.append((angle1 % 90, angle2 % 180))
                break
        if found_angles:
            break

print(prep_angles)
# Calculate angles for each state preparation
meas_angles = []
for i in range(len(names)):
    found_angles = False
    print(f"Finding correct angles to measure {names[i]} using QWP->HWP ordering.")
    # Loop through all possible angles for HWP and QWP
    for angle1 in np.arange(0, 360, 22.5):
        for angle2 in np.arange(0, 360, 22.5):
            # Prepare the state (assuming horizontally polarized input H_rho)
            prepared_state = qwp_hwp(angle1, angle2) @ state_vecs[i]
            prepared_state = np.outer(prepared_state, prepared_state.conj())
            prepared_state /= np.trace(prepared_state)

            expected_value = state_vecs[0].conj().T @ prepared_state @ state_vecs[0]
            # Check if the prepared state matches the state we are looking for
            if np.isclose(expected_value, 1):
                print(f"HWP: {angle1 % 90}, QWP: {angle2 % 180}")
                found_angles = True
                meas_angles.append((angle1 % 90, angle2 % 180))
                break
        if found_angles:
            break


print(meas_angles)
