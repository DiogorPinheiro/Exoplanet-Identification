import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split
from keras.models import load_model
from sklearn.utils import shuffle

from utilities import chunkVisualization, recall_m, f1_m, precision_m, auc_roc

# Model Directories
CNN_MODEL_DIRECTORY = '../models/CNN.h5'
ALEXNET_MODEL_DIRECTORY = '../models/alexnet.h5'
FNN_MODEL_DIRECTORY = '../models/FNN.h5'
LSTM_MODEL_DIRECTORY = '../models/lstm.h5'


def getModel(model_name):
    dependencies = {
        'f1_m': f1_m,
        'precision_m': precision_m,
        'recall_m': recall_m,
        'auc_roc': auc_roc
    }
    return load_model(model_name, custom_objects=dependencies)


def getScore(model, data_X, data_y):
    score = model.evaluate(data_X, data_y, verbose=0)
    print("%s: %.2f%% " %
          (model.metrics_names[1], score[1]*100, model.metrics_names[2]))


def checkPrediction(model, datax, datay, index):
    # Useful Indexex: 0,5,17,21,23,29,42
    prediction = model.predict_classes(datax)
    # for i in range(len(prediction)):
    #    print("X=%s, Predicted=%s" % (datay[i], prediction[i]))
    return(prediction[index])


def buildArray(mean_value, array_size):
    '''
        Builds an array with the light curve mean value

        input: mean_value (float) 
               array_size (int )
        output: array of floats
    '''
    aux = []
    for i in range(array_size):
        aux.append(mean_value)
    return aux


def replace_curve(indexes, data, mean_value):
    aux = []
    for i, d in enumerate(data):
        if i not in indexes:
            # Get array with the mean value
            aux.append(buildArray(mean_value, len(d)))
        else:
            aux.append(d)
    return aux


def setIntersection(data):
    '''
        Find the common elements in a list

        input: data (list of lists)
        output: list
    '''
    print(type(data))
    result = set(data[0])
    for s in data[1:]:
        result.intersection_update(s)
    return list(result)


def groupToPoints(data):
    '''
        Convert light curve divided by groups to a list of points

        input: data (list of lists of lists)
        output: list
    '''
    aux = []
    for v1, x in enumerate(data):
        for v2, y in enumerate(x):
            for v3, z in enumerate(y):
                aux.append(data[v1][v2][v3])
    return aux


if __name__ == "__main__":
    '''
        Access points -> data[0][0][0]
        Access groups -> data[0][0]
        Access chunks -> data[0]
    '''

    # Get Data
    data_global = np.loadtxt(
        '../data/Shallue/separated/global_test.csv', delimiter=',')
    data_global = shuffle(data_global)
    global_X = data_global[0:, 0:-1]  # Input
    global_Y = data_global[0:, -1]  # Labels

    global_X = np.expand_dims(
        global_X, axis=2)    # Shape data

    model = getModel(CNN_MODEL_DIRECTORY)

    test_lightCurve = global_X[0]

    mean_value = np.mean(test_lightCurve)   # Light Curve Mean Value
    print("Mean value = {}".format(mean_value))

    # Divide Data Into 41 Chunks
    chunks = np.array_split(test_lightCurve, 41)

    # Group indexes that will be used for search
    comb_values = [0, 1, 2, 3, 4]
    # Save the original prediction, it will serve as reference
    reference_pred = checkPrediction(model, global_X, global_Y, 0)
    for i in range(4):
        # Control number of groups in light curve
        divider_size = 5 * (i+1)

        combinations = list(itertools.combinations(comb_values, 3))

        search_groups = []
        print(i)
        for comb in combinations:
            data = replace_curve(comb, chunks, mean_value)
            # divide light curve into groups
            groups = np.array_split(chunks, divider_size)
            # Create light curve with new values
            new_curve = groupToPoints(data)
            global_X[0] = new_curve
            # Falta passar de groups para ter dados como no test_lightcurve
            pred = checkPrediction(model, global_X, global_Y, 0)
            if pred != reference_pred:
                search_groups.append(comb)

        comb_values = setIntersection(search_groups)
        print(comb_values)

    # getScore(CNN_MODEL_DIRECTORY, global_X, global_Y)

    # checkPrediction(model, global_X, global_Y, 0)

    # chunkVisualization(global_X[0], 50)
