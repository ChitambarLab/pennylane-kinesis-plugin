import pennylane as qml
from pennylane import numpy as qnp
from matplotlib import pyplot as plt

import pennylane_kinesis as pk

"""This script performs a gradient desecent optimization from a random starting point.
The optimization is run many times for different step sizes to determine the
effect of step size on the precision of the gradient descent optimization.
"""

theta_period = qnp.pi / 2
phi_period = qnp.pi
filepath = "./data/fiber_inversion_numeric_step_size_dependence/"

shots = 10000
datetime_str = pk.datetime_now_string()

opt_dicts = []


step_size_list = [1.0, 0.2, 0.15, 0.1, 0.05, 0.01, 0.001]

for use_adam in [True, False]:
    for step_size in step_size_list:
        dev = qml.device("kinesis.simulator", wires=[0], shots=shots)

        @qml.qnode(dev, diff_method="parameter-shift")
        def cost(settings):
            theta, phi = settings
            pk.HWPRotStage(theta, wires=dev.wires[0])
            pk.QWPRotStage(phi, wires=dev.wires[0])

            # fiber "noise"
            # qml.S(wires=dev.wires[0])
            qml.Hadamard(wires=dev.wires[0])

            return qml.expval(qml.Projector([1], wires=dev.wires[0]))

        step_size_filepath = filepath + "step_size_" + str(step_size).replace(".", "_") + "/"
        init_settings = qnp.random.rand(2, requires_grad=True) * qnp.array([theta_period, phi_period])

        optimizer_str = "adam_optimization_" if use_adam else "standard_optimization_"

        print("Running optimization for step_size ", step_size)
        opt_dict = pk.optimize(
            cost,
            init_settings,
            step_size=step_size,
            num_steps=int(4 * 1 / step_size),
            filepath=step_size_filepath,
            filename=optimizer_str + datetime_str,
            use_adam=use_adam,
            domain_periods=[theta_period, phi_period],
            verbose=False,
        )

        # plt.loglog(
        #     range(len(opt_dict["cost_vals"])),
        #     opt_dict["cost_vals"],
        #     label=optimizer_str + "step_size = " + str(step_size).replace(".", "_"),
        # )

# plt.legend()
# plt.show()
