import pennylane as qml
from pennylane import numpy as qnp

import matplotlib.pyplot as plt

import pennylane_kinesis as pk

print("pennylane kinesis version ", pk.__version__)

dev = qml.device("kinesis.simulator", shots=10000, wires=[0])


@qml.qnode(dev, diff_method="parameter-shift")
def circ(theta, phi):
    pk.HWPRotStage(theta, wires=[0])
    pk.QWPRotStage(phi, wires=[0])

    # fiber transformation
    qml.Rot(0.5, 0.7, 0.9, wires=[0])

    return qml.expval(qml.Projector([1], wires=[0]))


# minimize the probability of detecting the |1> state
def cost(settings):
    return circ(settings[0], settings[1]) ** 2


settings = qnp.array([0.5, 0.5], requires_grad=True)

opt = qml.GradientDescentOptimizer(stepsize=1)

cost_vals = []
for i in range(250):
    settings, cost_val = opt.step_and_cost(cost, settings)
    cost_vals += [cost_val]
    if i % 10 == 0:
        print("Step i = ", i)
        print(cost_val)

plt.semilogy(range(250), cost_vals)
plt.show()


# evaluating the parameter_frequencies of the QWP circuit
@qml.qnode(dev)
def simple_circuit_marked(theta):
    qml.RY(-1 * theta * 2, wires=[0], id="x")
    qml.PauliZ(wires=[0])
    qml.RY(theta * 2, wires=[0], id="x")

    return qml.expval(qml.PauliZ(0))


print("\nparameter frequencies of QWP :")
from pennylane.fourier import circuit_spectrum

freqs = circuit_spectrum(simple_circuit_marked)([0.5])
for k, v in freqs.items():
    print(k, ":", v)
