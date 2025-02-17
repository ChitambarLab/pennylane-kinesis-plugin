import numpy as np
from pennylane_kinesis.file_utilities import read_json
import math
from tqdm import tqdm
from itertools import permutations


def get_cost(matrix):
    return -(np.trace(matrix)/2)
    # game = np.eye(2)
    # score = 0
    # for i in range(2):
    #     score += np.sum(game[:, i] * matrix[i])
    #     # print("game[:, i]: ", game[:, i])
    #     # print("matrix[i]: ", game[:,i]*matrix[i])

    # return -(score/2)

param_space_data = read_json("behavior_matrices_1_angle.json")

angles = param_space_data["angles"]
matrices = param_space_data["behavior_matrices"]

matrices = np.array(matrices,dtype=float)
matrices /= 10000

print(matrices.shape)
behavior_matrices = np.array(list(permutations(matrices,2)))
print(behavior_matrices.shape)
print(behavior_matrices[0])

# xx,yy = np.meshgrid(matrices,matrices,indexing='ij')
# print(xx[0])
# print(yy[0])
# print(xx.shape)
# print(yy.shape)
# print(xx[0].shape)
exit()
# print(angles)
# print(matrices)
costs = []
for i in tqdm(range(len(angles)), position =0, desc='i'): # state 0 angles
    for j in tqdm(range(len(angles)), position=1, desc='j'): # state 1 angles
        # row_0 = matrices[i]
        # row_0 /= np.sum(row_0,dtype=float)
        # print("Row 0: ", row_0)
        # row_1 = matrices[j]
        # row_1 /= np.sum(row_1,dtype=float)
        # print("Row 1: ",row_1)
        matrix = np.vstack((matrices[i], matrices[j]))
        costs.append(get_cost(matrix))
        # print("Matrix: ", matrix)
        # cost = get_cost(matrix)
        # costs.append(costs)
        # print(f"Cost for angles {angles[i]} and {angles[j]}: {cost}")
        # break

