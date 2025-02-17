import numpy as np
from pennylane_kinesis.file_utilities import read_json
import math
from tqdm import tqdm
from itertools import permutations


def get_cost(matrix):
    return -(np.trace(matrix)/2)


param_space_data = read_json("behavior_matrices_1_angle.json")

angles = param_space_data["angles"]
matrices = param_space_data["behavior_matrices"]

matrices = np.array(matrices,dtype=float)
matrices /= 10000

print(matrices.shape)
behavior_matrices = np.array(list(permutations(matrices,2)))
print(behavior_matrices.shape)
print(behavior_matrices[0])

costs = []
for i in tqdm(range(len(angles)), position =0, desc='i'): # state 0 angles
    for j in tqdm(range(len(angles)), position=1, desc='j'): # state 1 angles
        matrix = np.vstack((matrices[i], matrices[j]))
        costs.append(get_cost(matrix))
print(costs)