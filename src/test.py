import numpy as np

data_local = np.loadtxt('neural_input_local_sovgol.csv', delimiter=',')
local_X = data_local[0:, 1:-1]  # Input
print(local_X.shape)

data_global = np.loadtxt('neural_input_global_sovgol.csv', delimiter=',')
global_X = data_global[0:, 1:-1]  # Input
print(global_X.shape)