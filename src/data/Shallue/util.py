import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import csv
from sklearn.preprocessing import MinMaxScaler

# File Directories
GLOBAL_TEST = "separated/test_global_sovgol.csv"
LOCAL_TEST = "separated/test_local_sovgol.csv"
GLOBAL_TRAIN = "separated/train_global_sovgol.csv"
LOCAL_TRAIN = "separated/train_local_sovgol.csv"


def appendToFile(file, row):
    '''
        Append Row Of Kepid Associated Values To The Table

        Input: Row Of Values
        Output: None
    '''
    with open(file, 'a') as fd:
        writer = csv.writer(fd)
        writer.writerow(row)


def create_file(file_name, datax, datay):
    file = open(file_name, "w")
    file.close()

    for index, (x, y) in enumerate(zip(datax, datay)):
        aux = []
        for i in range(len(x)):
            aux.append(x[i])
        aux.append(y)
        appendToFile(file_name, aux)


if __name__ == "__main__":

    data_local = np.loadtxt('Local.csv', delimiter=',', skiprows=1)

    data_global = np.loadtxt(
        'Global.csv', delimiter=',', skiprows=1)

    local_X = data_local[0:, 0:-1]  # Input
    local_Y = data_local[0:, -1]  # Labels
    # Suffle Data (Only For Shallue Datasets)
    local_X, local_Y = shuffle(local_X, local_Y)

    # Separate Local Data
    X_train_local, X_test_local, y_train_local, y_test_local = train_test_split(
        local_X, local_Y, test_size=0.2, random_state=1)

    scaler_local = MinMaxScaler(feature_range=(0, 1))  # Scale Values
    train_local = scaler_local.fit_transform(X_train_local)
    test_local = scaler_local.transform(X_test_local)

    global_X = data_global[0:, 0:-1]  # Input
    global_Y = data_global[0:, -1]  # Labels
    # Suffle Data (Only For Shallue Datasets)
    global_X, global_Y = shuffle(global_X, global_Y)

    # Separate global Data
    X_train_global, X_test_global, y_train_global, y_test_global = train_test_split(
        global_X, global_Y, test_size=0.2, random_state=1)

    scaler_global = MinMaxScaler(feature_range=(0, 1))  # Scale Values
    train_global = scaler_global.fit_transform(X_train_global)
    test_global = scaler_global.transform(X_test_global)

    #create_file(GLOBAL_TEST, test_global, y_test_global)
    #create_file(GLOBAL_TRAIN, train_global, y_train_global)
    #create_file(LOCAL_TEST, test_local, y_test_local)
    #create_file(LOCAL_TRAIN, train_local, y_train_local)
    zeros = 0
    ones = 0

    for i in local_Y:
        if i == 0:
            zeros += 1
        else:
            ones += 1

    print("Zeros {} ; Ones {}".format(zeros, ones))
