from pennylane_kinesis import VariableHWPRotStage, VariableQWPRotStage
import numpy as np
import pennylane as qml

if __name__ == "__main__":
    op = VariableHWPRotStage(np.pi / 2, 0)
    # print(op.parameter_frequencies)
    gen = qml.generator(op, format="observable")
    gen_eigvals = qml.eigvals(gen)
    print(qml.gradients.eigvals_to_frequencies(tuple(gen_eigvals)))
