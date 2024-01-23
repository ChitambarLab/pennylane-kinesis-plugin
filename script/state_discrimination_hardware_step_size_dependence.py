import pennylane as qml
from pennylane import numpy as qnp
from matplotlib import pyplot as plt

import pennylane_kinesis as pk

"""This script is intended to run on hardware. Its goal is to perform optimizations over a
range of step sizes and to save the corresponding results.
After many runs of this script, data can be aggregated to analyze the effect of shots on
optimization precision.


NOTE: Before running script make sure that the "./data/state_discrimination_hardware_step_size_dependence"
      directory exists with subdirectories "/step_size_0_N" where `_` replaces `.` in the stepsize float.
"""

theta_period = qnp.pi / 2
phi_period = qnp.pi
period_array = qnp.array([theta_period, theta_period, phi_period, phi_period])
filepath = "./data/state_discrimination_hardware_step_size_dependence/"
shot_rate = 100000
shots = 10000


step_size_list = [0.5, 0.2, 0.1, 0.05, 0.01] * 5

# step_size_list = [0.2, 0.15, 0.1, 0.05, 0.01]

opt_dicts = []

for step_size in step_size_list:
    datetime_str = pk.datetime_now_string()

    dev = qml.device("kinesis.cage.rotator", wires=[0], shots=shots, shot_rate=shot_rate)
    # dev = qml.device("kinesis.simulator", wires=[0], shots=shots, shot_rate=shot_rate)

    @qml.qnode(dev, diff_method="parameter-shift")
    def circ(circ_settings):
        # theta1, phi1, theta2, phi2 = circ_settings
        theta, phi = circ_settings
        # pk.HWPRotStage(theta1, wires=dev.wires[0])
        # pk.QWPRotStage(phi1, wires=dev.wires[0])
        # pk.HWPRotStage(theta2, wires=dev.wires[0])
        # pk.QWPRotStage(phi2, wires=dev.wires[0])
        pk.HWPRotStage(theta, wires=dev.wires[0])
        pk.QWPRotStage(phi, wires=dev.wires[0])

        return qml.probs(wires=dev.wires)

    def cost(settings):
        theta_a0, theta_a1, phi_a0, phi_a1 = settings

        probs_0 = circ([theta_a0, phi_a0])
        probs_1 = circ([theta_a1, phi_a1])
        return -(probs_0[0] + probs_1[1])

    step_size_filepath = filepath + "step_size_" + str(step_size).replace(".", "_") + "/"
    init_settings = qnp.random.rand(4, requires_grad=True) * period_array

    print("Running optimization for step_size ", step_size)
    opt_dict = pk.optimize(
        cost,
        init_settings,
        step_size=step_size,
        num_steps=50,
        filepath=step_size_filepath,
        filename="optimization_" + datetime_str,
        domain_periods=period_array.tolist(),
        shots=shots,
        shot_rate=shot_rate,
    )

    plt.loglog(
        range(len(opt_dict["cost_vals"])),
        qnp.array(opt_dict["cost_vals"]) + 2,
        label="step_size = " + str(step_size).replace(".", "_"),
    )

    dev.disconnect()

plt.legend()
plt.show()
