import pennylane as qml
from pennylane import numpy as qnp
from matplotlib import pyplot as plt

import pennylane_kinesis as pk

"""This script is intended to run on hardware. Its goal is to perform optimizations over a
range of shots and to save the corresponding results.
After many runs of this script, data can be aggregated to analyze the effect of shots on
optimization precision.


NOTE: Before running script make sure that the "./data/state_discrimination_hardware_shot_dependence"
      directory exists with subdirectories "/shots_N" where N in [10,100,1000,10000, 100000].
"""

theta_period = qnp.pi / 2
phi_period = qnp.pi
period_array = qnp.array(
    [theta_period, theta_period, phi_period, phi_period, theta_period, phi_period]
)
filepath = "./data/state_discrimination_hardware_shot_dependence/"
shot_rate = 120000

shots_list = [100, 1000, 10000, 100000, 10] * 2
# shots_list = [10000, 10, 1000, 100]
# shots_list = [10, 35, 100, 350, 1000, 3500, 10000]

opt_dicts = []

print("testing device setup")
dev_test = qml.device("kinesis.cage.rotator", wires=[0], shots=1000, shot_rate=shot_rate)
# dev_test = qml.device("kinesis.simulator", wires=[0], shots=1000, shot_rate=shot_rate)
print("testing device disconnect")
dev_test.disconnect()

for shots in shots_list:
    datetime_str = pk.datetime_now_string()

    dev = qml.device("kinesis.cage.rotator", wires=[0], shots=shots, shot_rate=shot_rate)
    # dev = qml.device("kinesis.simulator", wires=[0], shots=shots, shot_rate=shot_rate)

    @qml.qnode(dev, diff_method="parameter-shift")
    def circ(circ_settings):
        theta1, phi1, theta2, phi2 = circ_settings
        pk.HWPRotStage(theta1, wires=dev.wires[0])
        pk.QWPRotStage(phi1, wires=dev.wires[0])
        pk.HWPRotStage(theta2, wires=dev.wires[0])
        pk.QWPRotStage(phi2, wires=dev.wires[0])

        return qml.probs(wires=dev.wires)

    def cost(settings):
        theta_a0, theta_a1, phi_a0, phi_a1, theta_b, phi_b = settings

        probs_0 = circ([theta_a0, phi_a0, theta_b, phi_b])
        probs_1 = circ([theta_a1, phi_a1, theta_b, phi_b])
        return -(probs_0[0] + probs_1[1])

    shots_filepath = filepath + "shots_" + str(shots) + "/"
    init_settings = qnp.random.rand(6, requires_grad=True) * period_array

    print("Running optimization for shots ", shots)
    opt_dict = pk.optimize(
        cost,
        init_settings,
        step_size=0.05,
        num_steps=50,
        filepath=shots_filepath,
        filename="optimization_" + datetime_str,
        domain_periods=period_array.tolist(),
        shot_rate=shot_rate,
        shots=shots,
    )

    plt.semilogy(
        range(len(opt_dict["cost_vals"])),
        qnp.array(opt_dict["cost_vals"]) + 2,
        label="shots = " + str(shots),
    )

    dev.disconnect()

plt.legend()
plt.show()
