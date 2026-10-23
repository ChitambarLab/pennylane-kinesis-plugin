"""Generate a rotational waveplate sweep."""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


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


def system(theta1, theta2, theta3, theta4):
    return (
        qwp_hwp(theta3, theta4)
        @ hwp_qwp(theta1, theta2)
        @ H_rho
        @ hwp_qwp(theta1, theta2).conj().T
        @ qwp_hwp(theta3, theta4).conj().T
    )


def measurement(state, projector):
    return projector.conj().T @ state @ projector


if __name__ == "__main__":
    folder = "./data/theory/zero_angle/"
    Path(folder).mkdir(parents=True, exist_ok=True)

    angles = np.linspace(0, 360, 180)
    measurements = []
    for angle in angles:
        measurements.append(measurement(system(0, angle, 0, 135), H))

    plt.plot(angles, measurements)
    plt.xlabel("Angles (degrees)")
    plt.ylabel("P(H)")
    plt.title("Theoretical angle sweep (measure L, aQWP)")
    plt.savefig(folder + "aQWP_L.png")
