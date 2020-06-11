import numpy as np
import tensorflow as tf
from tensorflow import keras
from kerastuner import Hyperband
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from kerastuner.engine.hyperparameters import HyperParameters
import random

# File that contains the hypermodels
from hyperModels import CNNHyperModel, FNNHyperModel, LSTMHyperModel, DualCNNHyperModel

# Model directories
CNN_MODEL_DIRECTORY = 'models/CNN.hdf5'
FNN_MODEL_DIRECTORY = 'models/FNN.hdf5'
LSTM_MODEL_DIRECTORY = 'models/LSTM.hdf5'
DUAL_CNN_MODEL_DIRECTORY = 'models/CNN_DUAL.hdf5'


if __name__ == "__main__":
    '''
        Load Data, create model using Keras-Tuner and save model to file.

    '''
    # --------------------------- Load Data -------------------------------------------
    data_global = np.loadtxt(
        'global_train.csv', delimiter=',')          # Global View
    global_X = data_global[0:, 0:-1]  # Input
    global_Y = data_global[0:, -1]  # Labels

    #global_X, global_Y = shuffle(global_X, global_Y)
    X_train_global, X_test_global, y_train_global, y_test_global = train_test_split(
        global_X, global_Y, test_size=0.2, random_state=1)

    # Shape Data
    X_train_global_shaped = np.expand_dims(X_train_global, axis=2)
    X_test_global_shaped = np.expand_dims(X_test_global, axis=2)

    data_local = np.loadtxt(
        'local_train.csv', delimiter=',')           # Local View
    local_X = data_local[0:, 0:-1]  # Input
    local_Y = data_local[0:, -1]  # Labels

    #local_X, local_Y = shuffle(local_X, local_Y)
    X_train_local, X_test_local, y_train_local, y_test_local = train_test_split(
        local_X, local_Y, test_size=0.2, random_state=1)            # Split Data as 0.8 Training and 0.2 Testing

    # Shape Data
    X_train_local_shaped = np.expand_dims(X_train_local, axis=2)
    X_test_local_shaped = np.expand_dims(X_test_local, axis=2)

    input_shape_global = (X_test_global_shaped.shape[1], 1)
    input_shape_local = (X_test_local_shaped.shape[1], 1)

    # ------------------------------ Create Models ------------------------------------

    build_model = CNNHyperModel(input_shape=input_shape_local, num_classes=1)
    #build_model = LSTMHyperModel(input_shape=input_shape_global,num_classes=1)
    #build_model = DualCNNHyperModel(input_shape_local, input_shape_global, 1)
    #build_model = FNNHyperModel(num_classes=1)

    tuner = Hyperband(build_model, objective='val_loss',
                      max_epochs=40, project_name='num_'+str(random.randint(0, 29)), seed=random.randint(0, 30))
    tuner.search(X_train_global_shaped, y_train_global,
                 epochs=40, validation_data=(X_test_global_shaped, y_test_global), callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')])

    # Search for Dual CNN
    # tuner.search([X_train_local_shaped, X_train_global_shaped], y_train_global,
    #             epochs=40, validation_data=([X_test_local_shaped, X_test_global_shaped], y_test_global), callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')])

    # ----------------------------- Summarize and Save Model ------------------------

    best_model = tuner.get_best_models(1)[0]
    best_model.summary()
    best_param = tuner.get_best_hyperparameters(1)[0]
    print(best_param.values)

    best_model.save(CNN_MODEL_DIRECTORY)
