import pennylane as qml
from pennylane import numpy as qnp
from matplotlib import pyplot as plt

import pennylane_kinesis as pk

dev = qml.device("kinesis.simulator", wires=[0], shots=10000)
theta_period = qnp.pi / 2
phi_period = qnp.pi
filepath = "./data/fiber_inversion_hadamard_noise/"


@qml.qnode(dev, diff_method="parameter-shift")
def cost(settings):
    theta, phi = settings
    pk.HWPRotStage(theta, wires=dev.wires[0])
    pk.QWPRotStage(phi, wires=dev.wires[0])

    # fiber "noise"
    # qml.S(wires=dev.wires[0])
    qml.Hadamard(wires=dev.wires[0])

    return qml.expval(qml.Projector([1], wires=dev.wires[0]))


def plot_cost_contour(title, scan_dict, opt_dict=None):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.grid()

    fig.suptitle(title)

    CS = ax1.contour(scan_dict["x_grid"], scan_dict["y_grid"], scan_dict["z_grid"])
    ax1.clabel(CS, inline=True, fontsize=10)
    ax1.set_xlabel(r"HWP Angle ($\theta$ Radians)")
    ax1.set_ylabel(r"QWP Angle ($\phi$ Radians)")

    if opt_dict:
        theta_settings = []
        phi_settings = []
        for settings in opt_dict["settings_history"]:
            theta_settings += [settings[0]]
            phi_settings += [settings[1]]

        ax1.plot(theta_settings, phi_settings, marker="o", label="Gradient Descent")
        ax1.plot(theta_settings[0], phi_settings[0], "rx", markersize=8, label="Starting Point")
        ax1.plot(
            theta_settings[-1],
            phi_settings[-1],
            color="lime",
            marker="v",
            linestyle="",
            markersize=8,
            label="Ending Point",
        )

    if opt_dict:
        ax2.semilogy(
            range(len(opt_dict["cost_vals"])), opt_dict["cost_vals"], label="Gradient Descent"
        )
        ax2.grid()

    plt.legend()

    plt.show()


if __name__ == "__main__":
    print("Running optimization from fixed initial settings.")
    fixed_init_settings = qnp.array([qnp.pi / 3 + 0.1, qnp.pi / 4 + 0.1], requires_grad=True)
    opt_dict = pk.optimize(
        cost,
        fixed_init_settings,
        step_size=0.1,
        num_steps=12,
        filepath=filepath,
        filename="fixed_init_optimize",
        domain_periods=[theta_period, phi_period],
    )

    print("Scanning cost Landscape")
    theta_set = qnp.arange(0, theta_period + 0.1, 0.1)
    phi_set = qnp.arange(0, phi_period + 0.1, 0.1)
    cost_landscape = pk.scan_two_param_fn(
        cost,
        theta_set,
        phi_set,
        filename="cost_landscape",
        filepath=filepath,
    )

    plot_cost_contour("Inversion of Hadamard Transform", cost_landscape, opt_dict)
