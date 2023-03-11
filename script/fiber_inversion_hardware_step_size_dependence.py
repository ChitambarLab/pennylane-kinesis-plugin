import pennylane as qml
from pennylane import numpy as qnp
from matplotlib import pyplot as plt

import pennylane_kinesis as pk

"""This script is intended to run on hardware. Its goal is to perform optimizations over a
range of step sizes and to save the corresponding results.
After many runs of this script, data can be aggregated to analyze the effect of shots on
optimization precision.


NOTE: Before running script make sure that the "./data/fiber_inversion_hardware_step_size_dependence"
      directory exists with subdirectories "/step_size_0_N" where `_` replaces `.` in the stepsize float.
"""

theta_period = qnp.pi / 2
phi_period = qnp.pi
filepath = "./data/fiber_inversion_hardware_step_size_dependence/"
shot_rate = 60000
shots = 1000


step_size_list = [0.15]

# step_size_list = [0.2, 0.15, 0.1, 0.05, 0.01]

opt_dicts = []

for step_size in step_size_list:
    datetime_str = pk.datetime_now_string()

    dev = qml.device("kinesis.cage.rotator", wires=[0], shots=shots, shot_rate=shot_rate)

    @qml.qnode(dev, diff_method="parameter-shift")
    def cost(settings):
        theta, phi = settings
        pk.HWPRotStage(theta, wires=dev.wires[0])
        pk.QWPRotStage(phi, wires=dev.wires[0])

        return qml.expval(qml.Projector([1], wires=dev.wires[0]))

    step_size_filepath = filepath + "step_size_" + str(step_size).replace(".", "_") + "/"
    init_settings = qnp.random.rand(2, requires_grad=True) * qnp.array([theta_period, phi_period])

    print("Running optimization for step_size ", step_size)
    opt_dict = pk.optimize(
        cost,
        init_settings,
        step_size=step_size,
        num_steps=int(4 * 1 / step_size),
        filepath=step_size_filepath,
        filename="optimization_" + datetime_str,
        domain_periods=[theta_period, phi_period],
    )

    plt.loglog(
        range(len(opt_dict["cost_vals"])), opt_dict["cost_vals"], label="step_size = " + str(step_size).replace(".","_")
    )

    dev.disconnect()

plt.legend()
plt.show()