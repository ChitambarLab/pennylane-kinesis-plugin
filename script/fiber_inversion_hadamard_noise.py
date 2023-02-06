import pennylane as qml
from pennylane import numpy as qnp
from matplotlib import pyplot as plt

import pennylane_kinesis as pk

"""This numerical script scans over the cost landscape to create a contour plot
then performs a gradient descent from the peak.
"""

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


if __name__ == "__main__":
    datetime_str = pk.datetime_now_string()

    print("Running optimization from fixed initial settings.")
    fixed_init_settings = qnp.array([qnp.pi / 3 + 0.1, qnp.pi / 4 + 0.1], requires_grad=True)
    opt_dict = pk.optimize(
        cost,
        fixed_init_settings,
        step_size=0.1,
        num_steps=12,
        filepath=filepath,
        filename="fixed_init_optimize_" + datetime_str,
        domain_periods=[theta_period, phi_period],
    )

    print("Scanning cost Landscape")
    theta_set = qnp.arange(0, theta_period + 0.1, 0.1)
    phi_set = qnp.arange(0, phi_period + 0.1, 0.1)
    cost_landscape_dict = pk.scan_two_param_fn(
        cost,
        theta_set,
        phi_set,
        filename="cost_landscape_" + datetime_str,
        filepath=filepath,
    )

    pk.plot_trajectory_on_cost_contour(
        filepath + "cost_contour_" + datetime_str,
        cost_landscape_dict,
        opt_dict,
    )
