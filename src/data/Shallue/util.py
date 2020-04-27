import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import csv
from sklearn.preprocessing import MinMaxScaler

# File Directories
GLOBAL_TEST = "separated/test_global2.csv"
LOCAL_TEST = "separated/test_local2.csv"
GLOBAL_TRAIN = "separated/train_global2.csv"
LOCAL_TRAIN = "separated/train_local2.csv"


def appendToFile(file, row):
    '''
        Append Row Of Kepid Associated Values To The Table

        Input: Row Of Values
        Output: None
    '''
    with open(file, 'a') as fd:
        writer = csv.writer(fd)
        writer.writerow(row)


def create_file(file_name, data):
    file = open(file_name, "w")
    file.close()

    for x in data:
        appendToFile(file_name, x)


if __name__ == "__main__":

    data_local = np.loadtxt('shallue_local.csv', delimiter=',', skiprows=1)

    data_global = np.loadtxt('shallue_global.csv', delimiter=',', skiprows=1)

    data_local = shuffle(data_local)
    data_global = shuffle(data_global)

    train_global, test_global = train_test_split(
        data_global, test_size=0.2, random_state=1)

    scaler_global = MinMaxScaler(feature_range=(0, 1))  # Scale Values
    train_global = scaler_global.fit_transform(train_global)
    test_global = scaler_global.transform(test_global)

    train_local, test_local = train_test_split(
        data_local, test_size=0.2, random_state=1)

    scaler_local = MinMaxScaler(feature_range=(0, 1))  # Scale Values
    train_local = scaler_local.fit_transform(train_local)
    test_local = scaler_local.transform(test_local)

    create_file(GLOBAL_TEST, test_global)
    create_file(GLOBAL_TRAIN, train_global)
    create_file(LOCAL_TEST, test_local)
    create_file(LOCAL_TRAIN, train_local)
