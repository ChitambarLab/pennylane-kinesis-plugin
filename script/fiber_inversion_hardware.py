import pennylane as qml
from pennylane import numpy as qnp
from matplotlib import pyplot as plt

import pennylane_kinesis as pk

"""This script scans over the domain to construct a contour plot of the cost function.
A gradient descent optimization is then run starting from a peak.
"""

theta_period = qnp.pi / 2
phi_period = qnp.pi
filepath = "./data/fiber_inversion_hardware/"
datetime_str = pk.datetime_now_string()


shots = 100000
shot_rate = 100000
dev = qml.device("kinesis.cage.rotator", wires=[0], shots=shots, shot_rate=shot_rate)


@qml.qnode(dev, diff_method="parameter-shift")
def cost(settings):
    theta, phi = settings
    pk.HWPRotStage(theta, wires=dev.wires[0])
    pk.QWPRotStage(phi, wires=dev.wires[0])

    return qml.expval(qml.Projector([1], wires=dev.wires[0]))


print("Scanning cost Landscape")
theta_set = qnp.arange(0, theta_period + 0.11, 0.1)
phi_set = qnp.arange(0, phi_period + 0.11, 0.1)
cost_landscape_dict = pk.scan_two_param_fn(
    cost,
    theta_set,
    phi_set,
    filename="cost_landscape_" + datetime_str,
    filepath=filepath,
    shots=shots,
    shot_rate=shot_rate,
)

print("Running optimization from fixed initial settings.")
# fixed_init_settings = qnp.array(
#     [(100.29 * 2 * qnp.pi / 360) % theta_period, (112.21 * 2 * qnp.pi / 360) % phi_period],
#     requires_grad=True,
# )
# opt_dict = pk.optimize(
#     cost,
#     fixed_init_settings,
#     step_size=0.15,  # previously 0.1
#     num_steps=10,  # previously 12
#     filepath=filepath,
#     filename="fixed_init_optimize_" + datetime_str,
#     domain_periods=[theta_period, phi_period],
#     shots=shots,
#     shot_rate=shot_rate,
# )

# pk.plot_trajectory_on_cost_contour(
#     filepath + "cost_contour_plot_" + datetime_str,
#     cost_landscape_dict,
#     opt_dict,
# )

dev.disconnect()
