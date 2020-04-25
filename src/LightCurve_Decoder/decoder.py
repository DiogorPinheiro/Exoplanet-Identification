import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random
import matplotlib.pyplot as plt

from utilities import createChunkingPlot


def chunks(data, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(data), n):
        yield data[i:i + n]


if __name__ == "__main__":
    data_global = np.loadtxt('../data/global_movavg.csv', delimiter=',')
    global_X = data_global[0:, 1:-1]  # Input
    global_Y = data_global[0:, -1]  # Labels
    # Scale Data
    scaler_global = MinMaxScaler(feature_range=(0, 1))  # Scale Values
    global_X = scaler_global.fit_transform(global_X)

    chuncked_data = list(chunks(global_X[0], 50))

    #createChunkingPlot(global_X[0], 50)

    plt.show()
