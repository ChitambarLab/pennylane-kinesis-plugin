import pennylane as qml
import pennylane_kinesis as pk

from pennylane import numpy as qnp
from pennylane.devices.tests import test_device

import matplotlib.pyplot as plt

pk.__version__

dev = qml.device("kinesis.simulator", shots=10000, wires=[0])

dev2 = pk.KinesisSimulatorDevice(wires=[0], shots=10000)

@qml.qnode(dev2, diff_method="parameter-shift")
def circ(theta, phi):
    qml.RY(-1*theta, wires=[0])
    qml.PauliZ(wires=[0])
    qml.RY(theta, wires=[0])

    qml.RY(-1*phi, wires=[0])
    qml.S(wires=[0])
    qml.RY(phi, wires=[0])

    qml.Rot(0.5,0.7,0.9, wires=[0])

    # return qml.probs(0)
    return qml.expval(qml.Projector([1], wires=[0]))

circ(0,0)

def cost(theta):
    return circ(theta)[1]

@qml.qnode(dev, diff_method="finite-diff")
def fd_circ(theta):
    pk.HWP(theta, wires=[0])
    return qml.probs(0)

fd_circ(qnp.pi/2)

def fd_cost(theta):
    return fd_circ(theta)[1]

fd_cost(qnp.pi/2)



@qml.qnode(dev, diff_method="parameter-shift")
def ps_circ(theta):
    pk.HWP(theta, wires=[0])
    return qml.probs(0)

ps_circ(qnp.pi/2)

def ps_cost(theta):
    return ps_circ(theta)[1]

ps_cost(1)


opt = qml.GradientDescentOptimizer(stepsize=0.1)

# settings, cost_val = opt.step_and_cost(fd_cost, qnp.array(0.5))

# settings, cost_val = opt.step_and_cost(ps_cost, qnp.array(0.5))

settings = qnp.random.rand(2)
cost_vals = []
num_steps = 100
for i in range(num_steps):
    settings, cost_val = opt.step_and_cost(circ, *settings)
    cost_vals += [cost_val]


plt.semilogy(range(num_steps), cost_vals)

@qml.qnode(dev)
def hwp_circ(theta):
    pk.HWP(theta,wires=[0])
    return qml.expval(qml.PauliZ(0))

qml.grad(hwp_circ)(qnp.array(0.5))


test_device("default.qubit", shots=10000)
qml.devices.tests.get_device_tests()