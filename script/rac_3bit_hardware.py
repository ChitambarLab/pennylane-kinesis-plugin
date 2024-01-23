import pennylane as qml
from pennylane import numpy as qnp
from matplotlib import pyplot as plt
import qnetvo
import sys

import pennylane_kinesis as pk

"""This script is intended to run on hardware. Its goal is to optimize the 3-bit random access coding task.

To run on simulator use: `python script/rac_3bit_hardware.py numerics`
"""

theta_period = qnp.pi / 2
phi_period = qnp.pi
shot_rate = 120000
shots = 10000

dir = pk.make_dir("./data/rac_3bit/")

if sys.argv[1] == "numerics":
    subdir = "numerics/"
    dev_kwargs = {
        "name": "kinesis.simulator",
        "shots": shots,
        "qwp_domain_period": phi_period,
        "hwp_domain_period": theta_period,
    }
else:
    subdir = "hardware/"
    dev_kwargs = {
        "name": "kinesis.cage.rotator",
        "shots": shots,
        "shot_rate": shot_rate,
    }

filepath = pk.make_dir(dir + subdir)


def arb_unitary(settings, wires):
    pk.HWPRotStage(settings[0], wires=[wires[0]])
    pk.QWPRotStage(settings[1], wires=[wires[0]])


datetime_str = pk.datetime_now_string()


ansatz = qnetvo.NetworkAnsatz(
    [qnetvo.PrepareNode(num_in=8, wires=[0], ansatz_fn=arb_unitary, num_settings=2)],
    [qnetvo.MeasureNode(num_in=3, num_out=2, wires=[0], ansatz_fn=arb_unitary, num_settings=2)],
    dev_kwargs=dev_kwargs,
)

cost = qnetvo.linear_probs_cost_fn(
    ansatz,
    qnp.array(
        [
            [1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1],
        ]
    ),
    qnode_kwargs={"diff_method": "parameter-shift"},
)

print("Running optimization 3bit random access code")
opt_dict = pk.optimize(
    lambda settings: cost(*settings),
    qnp.array(ansatz.rand_network_settings()),
    step_size=0.04,
    num_steps=50,
    filepath=filepath,
    filename="optimization_" + datetime_str,
    domain_periods=[theta_period, phi_period] * 11,
    shots=shots,
    shot_rate=shot_rate,
)

fig, (ax1) = plt.subplots(ncols=1)

fig.suptitle("VQO of 3-bit Random Access Code")

ax1.plot(
    range(len(opt_dict["cost_vals"])),
    -1 * qnp.array(opt_dict["cost_vals"]),
    label="Optimization Trajectory",
)
ax1.plot(
    range(len(opt_dict["cost_vals"])),
    [18] * len(opt_dict["cost_vals"]),
    label="Classical Bound",
)

ax1.plot(
    range(len(opt_dict["cost_vals"])),
    [18.472135] * len(opt_dict["cost_vals"]),
    label="Real Quantum Bound",
)

plt.legend()
plt.show()
