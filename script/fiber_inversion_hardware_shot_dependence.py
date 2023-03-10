import pennylane as qml
from pennylane import numpy as qnp
from matplotlib import pyplot as plt

import pennylane_kinesis as pk

"""This script is intended to run on hardware. Its goal is to perform optimizations over a
range of shots and to save the corresponding results.
After many runs of this script, data can be aggregated to analyze the effect of shots on
optimization precision.


NOTE: Before running script make sure that the "./data/fiber_inversion_hardware_shot_dependence"
      directory exists with subdirectories "/shots_N" where N in [10,35,100,350,1000,3500,10000].
"""

theta_period = qnp.pi / 2
phi_period = qnp.pi
filepath = "./data/fiber_inversion_hardware_shot_dependence/"
shot_rate = 50000

shots_list = [1000, 10000]
# shots_list = [10, 35, 100, 350, 1000, 3500, 10000]
datetime_str = pk.datetime_now_string()

opt_dicts = []

for shots in shots_list:
    dev = qml.device("kinesis.cage.rotator", wires=[0], shots=shots, shot_rate=shot_rate)

    @qml.qnode(dev, diff_method="parameter-shift")
    def cost(settings):
        theta, phi = settings
        pk.HWPRotStage(theta, wires=dev.wires[0])
        pk.QWPRotStage(phi, wires=dev.wires[0])

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
        filename="optimization_" + datetime_str,
        domain_periods=[theta_period, phi_period],
    )

    plt.semilogy(
        range(len(opt_dict["cost_vals"])), opt_dict["cost_vals"], label="shots = " + str(shots)
    )

plt.legend()
plt.show()
