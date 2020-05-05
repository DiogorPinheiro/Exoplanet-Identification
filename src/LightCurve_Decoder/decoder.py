import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split
from keras.models import load_model
from sklearn.utils import shuffle
from collections import Counter

from utilities import chunkVisualization, recall_m, f1_m, precision_m, auc_roc, saveChunkVisualization

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
    prediction = model.predict_classes(datax)
    # for i in range(len(prediction)):
    #    print("i = {} ; X={} ; Predicted={}".format(
    #        i, datay[i], prediction[i]))
    return(prediction[index])


def buildArray(mean_value, array):
    '''
        Builds an array with the light curve mean value

        input: mean_value (float)
               array (list of lists )
        output: array of floats
    '''
    aux = []
    for index, i in enumerate(array):
        for index2, i2 in enumerate(i):
            aux.append(mean_value)
    # Use numpy to reshape array
    aux = np.array(aux)
    aux = aux.reshape((aux.shape[0], 1))

    return list(aux)


def replace_curve(indexes, data, mean_value):
    aux = []

    for i, d in enumerate(data):
        if i not in indexes:
            arr = buildArray(mean_value, d)
            # Get array with the mean value
            aux.append(arr)
        else:
            aux.append(d)
    return aux


def setIntersection(data):
    '''
        Find the common elements in a list

        input: data (list of lists)
        output: list
    '''
    result = set(data[0])
    for s in data[1:]:
        result.intersection_update(s)
    return list(result)


def mostFrequentInSet(data):
    flattened_list = [elem for sublist in data for elem in sublist]
    print(flattened_list)
    print(max(set(flattened_list), key=flattened_list.count))


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


def defineWantedSet(indexes):
    '''
        Define the groups where the program needs to search

        input: indexes (list of int) -> indexes with possible target
        output: list of int
    '''
    aux = []
    for i in indexes:
        aux.append(i*2)
        aux.append((i*2)+1)
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
    # data_global = shuffle(data_global)
    global_X = data_global[0:, 0:-1]  # Input
    global_Y = data_global[0:, -1]  # Labels

    global_X = np.expand_dims(
        global_X, axis=2)    # Shape data

    model = getModel(CNN_MODEL_DIRECTORY)

    pos_index = 3088    # 3070, 3065, 3077, 3088, 3021, 2939
    test_lightCurve = global_X[pos_index]

    mean_value = np.mean(test_lightCurve)   # Light Curve Mean Value
    print("Mean value = {}".format(mean_value))

    # Divide Data Into 41 Chunks
    chunks = np.array_split(test_lightCurve, 40)

    # Group indexes that will be used for search
    comb_values = [0, 1, 2, 3, 4]
    # Save the original prediction, it will serve as reference
    reference_pred = checkPrediction(model, global_X, global_Y, pos_index)
    print("Reference Prediction = {}".format(reference_pred))

    divider_size = 5
    for i in range(4):
        # Control number of groups in light curve
        # 0 -> 5 groups ; 1 -> 10 groups ; 2 -> 20 groups ; 3 -> 41 groups

        combinations = list(itertools.combinations(comb_values, 3))
        # divide light curve into groups
        groups = np.array_split(chunks, divider_size)
        search_groups = []
        print("Iteration nº{}".format(i))
        for val, comb in enumerate(combinations):
            global_X_copy = global_X.copy()    # To avoid manipulating the original data

            data = replace_curve(comb, groups, mean_value)

            # Create light curve with new values
            new_curve = groupToPoints(data)
            new_curve = np.array(new_curve)
            global_X_copy[pos_index] = list(
                new_curve.reshape((new_curve.shape[0], 1)))

            #chunkVisualization(global_X_copy[pos_index], 400)
            # Save graph to file
            # save_filename = (
            #    'Images/{}_iter{}_comb{}'.format(pos_index, i, val))
            # Number of vertical lines in visualization (representing the groups)
            #vert_lines = int((len(test_lightCurve)-1)/divider_size)
            # saveChunkVisualization(
            #    global_X_copy[pos_index], vert_lines, save_filename)

            # Predict new light curve and compare with reference
            pred = checkPrediction(model, global_X_copy, global_Y, pos_index)
            print("Prediction = {}".format(pred))
            print(comb)
            if pred != reference_pred:
                search_groups.append(list(comb))
        mostFrequentInSet(search_groups)
        # Define the indexes that will be searched next
        if not search_groups:
            print("No Search Groups Found!")
            break
        else:
            comb_values = defineWantedSet(setIntersection(search_groups))
            print(comb_values)
            if i == 3:
                divider_size = (divider_size*2)+1
            else:
                divider_size = divider_size * 2

    # getScore(CNN_MODEL_DIRECTORY, global_X, global_Y)

    # chunkVisualization(global_X[0], 50)
