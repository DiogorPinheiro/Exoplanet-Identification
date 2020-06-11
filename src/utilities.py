import numpy as np
import csv
from keras import backend as K

# ------------------------- Evaluation Metrics ---------------------


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# ------------------------- List Manipulation -----------------------
def concatenate(gl, lo):
    '''
        Concatenates two lists.

        @param gl (np.array(float)): global values
        @param lo (np.array(float)): local values

        @return (np.array(float)): concatencated values of both views

    '''
    gl = gl.tolist()
    lo = lo.tolist()
    for index, a in enumerate(lo):
        for b in a:
            gl[index].append(b)

    return np.array(gl)


def joinLists(data1, data2):
    '''
        Joins two lists.

        @param data1 (np.array(float)): values of the first data array
        @param data2 (np.array(float)): values of the second data array

        @return (np.array(float)): joined values of both views

    '''
    for index, a in enumerate(data2):
        for b in a:
            data1[index].append(b)

    return np.array(data1)


def writeToFile(name, data):
    '''
        Write data to file.

        @param name (String): name of file
        @param data (np.array(float)): data to be written to file

    '''
    with open(name, "w") as fd:  # Write Header
        writer = csv.writer(fd)
        writer.writerow(data)
