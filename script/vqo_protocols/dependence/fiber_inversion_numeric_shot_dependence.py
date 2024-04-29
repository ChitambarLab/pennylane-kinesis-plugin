import pennylane as qml
from pennylane import numpy as qnp
from matplotlib import pyplot as plt

import pennylane_kinesis as pk

"""This script performs a gradient desecent optimization from a random starting point.
The optimization is run many times for different numbers of shots to determine the
effect of shots on the precision of the gradient descent optimization.
"""

theta_period = qnp.pi / 2
phi_period = qnp.pi
filepath = "./data/fiber_inversion_numeric_shot_dependence/"

# shots_list = [10, 35, 100, 350, 1000, 3500, 10000]
shots_list = [1000, 10000]
background_list = [5, 49]
datetime_str = pk.datetime_now_string()

opt_dicts = []

i = 0
for shots in shots_list:
    dev = qml.device(
        "kinesis.simulator", wires=[0], shots=shots, background_counts=background_list[i]
    )

    @qml.qnode(dev, diff_method="parameter-shift")
    def cost(settings):
        theta, phi = settings
        pk.HWPRotStage(theta, wires=dev.wires[0])
        pk.QWPRotStage(phi, wires=dev.wires[0])

        # fiber "noise"
        # qml.S(wires=dev.wires[0])
        qml.Hadamard(wires=dev.wires[0])

        return qml.expval(qml.Projector([1], wires=dev.wires[0]))

    shots_filepath = filepath + "shots_" + str(shots) + "/"
    init_settings = qnp.random.rand(2, requires_grad=True) * qnp.array([theta_period, phi_period])

    print("Running optimization for shots ", shots)
    opt_dict = pk.optimize(
        cost,
        init_settings,
        step_size=0.1,
        num_steps=50,
        filepath=shots_filepath,
        filename="background_noise_optimization_" + datetime_str,
        domain_periods=[theta_period, phi_period],
        background_counts=background_list[i],
        shots=shots,
    )
    i += 1


#     plt.semilogy(
#         range(len(opt_dict["cost_vals"])), opt_dict["cost_vals"], label="shots = " + str(shots)
#     )

# plt.legend()
# plt.show()
