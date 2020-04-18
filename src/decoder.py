import numpy as np
from sklearn.preprocessing import MinMaxScaler


def chunks(data, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(data), n):
        yield data[i:i + n]


if __name__ == "__main__":
    data_local = np.loadtxt('data/local_movavg.csv', delimiter=',')
    local_X = data_local[0:, 1:-1]  # Input
    local_Y = data_local[0:, -1]  # Labels
    # Scale Data
    scaler_local = MinMaxScaler(feature_range=(0, 1))  # Scale Values
    local_X = scaler_local.fit_transform(local_X)

    split_data = list(chunks(local_X[0], 5))
    # print(split_data)
    # Get loss values for all samples

    # For each input_length
    #       Choose three random chunks
    #       Remove those three samples from data
    #       Train Model
    #       Compare result with reference - Se for maior que percentagem a analisar -> Append to list
