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
    for i in range(len(prediction)):
        print("X=%s, Predicted=%s" % (datay[i], prediction[i]))
    # print("X=%s, Predicted=%s" % (datay[index], prediction[index]))


if __name__ == "__main__":
    # Get Data
    data_global = np.loadtxt(
        '../data/Shallue/separated/global_test.csv', delimiter=',')
    data_global = shuffle(data_global)
    global_X = data_global[0:, 0:-1]  # Input
    global_Y = data_global[0:, -1]  # Labels

    global_X = np.expand_dims(
        global_X, axis=2)    # Shape data

    model = getModel(CNN_MODEL_DIRECTORY)

    chunked_points = np.array_split(global_X[0], 41)

    divider_size = 5
    chuncked_group = np.array_split(chunked_points, divider_size)

    # getScore(CNN_MODEL_DIRECTORY, global_X, global_Y)

    # checkPrediction(model, global_X, global_Y, 0)

    # combinations = list(itertools.combinations(chuncked_data, 3))

    # chunkVisualization(global_X[0], 50)
