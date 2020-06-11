import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.utils import shuffle

from utilities import f1_m, precision_m, recall_m

# Model Directories
CNN_MODEL_DIRECTORY = 'models/CNN.hdf5'
FNN_MODEL_DIRECTORY = 'models/FNN.hdf5'
LSTM_MODEL_DIRECTORY = 'models/LSTM.hdf5'
DUAL_CNN_MODEL_DIRECTORY = 'models/CNN_DUAL.hdf5'


def evaluate(model_name, data_X, data_y):
    '''
        Evaluate model using Accuracy, Loss, AUC, Precision, Recall and F1 metrics.

        @param model_name (String): directory where the model .hdf5 is saved
        @param data_X (np.ndarray): data used for evaluation
        @param data_y (np.ndarray): true labels of data

    '''
    # Model Dependencies
    dependencies = {
        'f1_m': f1_m,
        'precision_m': precision_m,
        'recall_m': recall_m,
        'auc_roc': tf.keras.metrics.AUC(),
        'num_classes': 1,
        'input_shape': (data_X.shape[1], 1)
    }

    # Load Model
    model = load_model(model_name, custom_objects=dependencies)

    # Evaluate
    score_loss = []
    score_acc = []
    score_f1 = []
    score_prec = []
    score_rec = []
    score_auc = []
    for i in range(50):             # 50 repetitions
        # Shuffle data to avoid memorization
        data_X_shuf, data_y_shuf = shuffle(data_X, data_y)

        # Evaluate Model
        score = model.evaluate(data_X_shuf, data_y_shuf, verbose=0)

        score_loss.append(score[0])
        score_acc.append(score[1])
        score_f1.append(score[2])
        score_prec.append(score[3])
        score_rec.append(score[4])
        score_auc.append(score[5])

    # Print Results
    print("\n------------------ Model : {} ---------------------".format(model_name))
    print("{}: {:0.2f} ".format(model.metrics_names[0], np.mean(score_loss)))
    print("%s: %.2f%% " %
          (model.metrics_names[1], np.mean(score_acc)))
    print("%s: %.2f%% " %
          (model.metrics_names[2], np.mean(score_f1)))
    print("%s: %.2f%% " %
          (model.metrics_names[3], np.mean(score_prec)))
    print("%s: %.2f%% " %
          (model.metrics_names[4], np.mean(score_rec)))
    print("%s: %.2f%% " %
          (model.metrics_names[5], np.mean(score_auc)))

    print("---------------------------------------------------")


def evaluateDual(model_name, global_X, global_y, local_X, local_y):
    '''
        Evaluate models with dual input using Accuracy, Loss, AUC, Precision, Recall and F1 metrics.

        @param model_name (String): directory where the model .hdf5 is saved
        @param global_X (np.ndarray): global view data used for evaluation
        @param global_y (np.ndarray): true labels of data
        @param local_X (np.ndarray): local view data used for evaluation
        @param local_y (np.ndarray): true labels of data

    '''
    # Model Dependencies
    dependencies = {
        'f1_m': f1_m,
        'precision_m': precision_m,
        'recall_m': recall_m,
        'auc_roc': tf.keras.metrics.AUC(),
        'num_classes': 1,
        'input_shape': (local_X.shape[1], global_X.shape[1], 1)
    }

    # Get Model
    model = load_model(model_name, custom_objects=dependencies)

    # Evaluate
    score_loss = []
    score_acc = []
    score_f1 = []
    score_prec = []
    score_rec = []
    score_auc = []
    for i in range(50):
        global_X_shuf, global_y_shuf, local_X_shuf, local_y_shuf = shuffle(
            global_X, global_y, local_X, local_y)

        # Evaluate Model
        score = model.evaluate(
            [local_X_shuf, global_X_shuf], global_y_shuf, verbose=0)

        score_loss.append(score[0])
        score_acc.append(score[1])
        score_f1.append(score[2])
        score_prec.append(score[3])
        score_rec.append(score[4])
        score_auc.append(score[5])

    # Print results
    print("\n------------------ Model : {} ---------------------".format(model_name))
    print("{}: {:0.2f} ".format(model.metrics_names[0], np.mean(score_loss)))
    print("%s: %.2f%% " %
          (model.metrics_names[1], np.mean(score_acc)))
    print("%s: %.2f%% " %
          (model.metrics_names[2], np.mean(score_f1)))
    print("%s: %.2f%% " %
          (model.metrics_names[3], np.mean(score_prec)))
    print("%s: %.2f%% " %
          (model.metrics_names[4], np.mean(score_rec)))
    print("%s: %.2f%% " %
          (model.metrics_names[5], np.mean(score_auc)))

    print("---------------------------------------------------")


if __name__ == "__main__":
    '''
        Read data and evaluate model.

    '''
    data_global = np.loadtxt(
        'data/Shallue/separated/global_test.csv', delimiter=',')
    # data_global = shuffle(data_global)
    global_X = data_global[0:, 0:-1]  # Input
    global_Y = data_global[0:, -1]  # Labels

    data_local = np.loadtxt(
        'data/Shallue/separated/local_test.csv', delimiter=',')
    # data_local = shuffle(data_local)
    local_X = data_local[0:, 0:-1]  # Input
    local_Y = data_local[0:, -1]  # Labels

    # Scale Data
    # scaler_global = MinMaxScaler(feature_range=(0, 1))  # Scale Values
    # global_X = scaler_global.fit_transform(global_X)

    # Separate global Data
    # X_train_global, X_test_global, y_train_global, y_test_global = train_test_split(
    #    global_X, global_Y, test_size=0.2, random_state=1)

    # X_test_global = np.expand_dims(
    #    X_test_global, axis=2)    # Shape data

    global_X = np.expand_dims(
        global_X, axis=2)

    local_X = np.expand_dims(
        local_X, axis=2)

    # evaluate(CNN_MODEL_DIRECTORY, local_X, local_Y)
    evaluateDual(DUAL_CNN_MODEL_DIRECTORY,
                 global_X, global_Y, local_X, local_Y)
