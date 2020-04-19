import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random


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

    for lcurve in local_X:
        # Split Data Into Chunks of 5 Samples
        split_data = list(chunks(selected_curve, 5))
    blocks = []
    aux = []
    for i in range(3):

    print(random.choice(split_data))
    # Get loss values for all samples

    # For each input_length
    #       Choose three random chunks
    #       Replace those three samples from data with average values array and save those samples into list
    #       Predict value for light_curve and append to list

    # Compare values and get those that are under 1 and 2 std below average
