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


shots = 3000000
shot_rate = 3000000
dev = qml.device("kinesis.cage.rotator", wires=[0], shots=shots, shot_rate=shot_rate)


# def cost(settings):
@qml.qnode(dev, diff_method="parameter-shift")
def circ(circ_settings):
    theta1, phi1, theta2, phi2 = circ_settings
    pk.HWPRotStage(theta1, wires=dev.wires[0])
    pk.QWPRotStage(phi1, wires=dev.wires[0])
    # noise happens
    pk.HWPRotStage(theta2, wires=dev.wires[0])
    pk.QWPRotStage(phi2, wires=dev.wires[0])

    # return qml.expval(qml.Projector([1], wires=dev.wires[0]))
    return qml.probs(wires=dev.wires)


# print(circ(qnp.array([0, 0, 0, 0])))


def cost(settings):
    theta_a0, theta_a1, phi_a0, phi_a1, theta_b, phi_b = settings

    probs_0 = circ([theta_a0, phi_a0, theta_b, phi_b])
    probs_1 = circ([theta_a1, phi_a1, theta_b, phi_b])
    return -(probs_0[0] + probs_1[1])


# print(cost(qnp.zeros(6)))


print("Scanning cost Landscape")
theta_set = qnp.arange(0, theta_period + 0.11, 0.1)
phi_set = qnp.arange(0, phi_period + 0.11, 0.1)
# cost_landscape_dict = pk.scan_two_param_fn(
#     cost,
#     theta_set,
#     phi_set,
#     filename="cost_landscape_" + datetime_str,
#     filepath=filepath,
#     shots=shots,
#     shot_rate=shot_rate,
# )

opt_dict = pk.optimize(
    cost,
    qnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
    num_steps=10,
    step_size=0.8,
    filepath="./",
    filename="test_state_discrimination",
    init_opt_dict=None,
    domain_periods=None,
    use_adam=False,
    verbose=True,
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
