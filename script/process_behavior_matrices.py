import numpy as np
from pennylane_kinesis.file_utilities import read_json, write_json
import math
from tqdm import tqdm
from itertools import permutations
import pickle


def get_cost(matrix):
    return -(np.trace(matrix) / 2)


param_space_data = read_json("behavior_matrices_1_angle.json")

angles = param_space_data["angles"]
matrices = param_space_data["behavior_matrices"]

matrices = np.array(matrices, dtype=float)
matrices /= 10000


print(len(angles))
print(len(matrices))
costs = []
new_angles = []
behavior_matrices = []
num_buffer = 0

for i in tqdm(range(len(angles)), position=0, desc="i"):  # state 0 angles
    for j in range(len(angles)):  # state 1 angles
        # row_0 = matrices[i]
        # row_0 /= np.sum(row_0,dtype=float)
        # print("Row 0: ", row_0)
        # row_1 = matrices[j]
        # row_1 /= np.sum(row_1,dtype=float)
        # print("Row 1: ",row_1)
        # print(j)
        new_angles.append([angles[i], angles[j]])
        matrix = np.vstack((matrices[i], matrices[j]))
        behavior_matrices.append(matrix)
        costs.append(get_cost(matrix))
        num_buffer += 1

        if num_buffer >= 500:
            with open("behavior_matrices_2.pkl", "ab") as f:
                pickle.dump([new_angles, costs, behavior_matrices], f)
            new_angles = []
            behavior_matrices = []
            costs = []
            num_buffer = 0
        # print("Matrix: ", matrix)
        # cost = get_cost(matrix)
        # costs.append(costs)
        # print(f"Cost for angles {angles[i]} and {angles[j]}: {cost}")
        # break


# write_json({"angles": angles, "behavior_matrices": behavior_matrices, "costs": costs}, "behavior_matrices_2_angles.json")
