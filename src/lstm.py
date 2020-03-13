import numpy as np
from comet_ml import Experiment, Optimizer
import tensorflow as tf
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Input, concatenate, Flatten, Dropout, PReLU, BatchNormalization, Activation, LSTM
from keras.layers.convolutional import Conv1D
from keras.layers import MaxPooling1D
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler
import time as t
from sklearn.metrics import roc_auc_score, recall_score, precision_score


from dataFunctions import dataInfo

CSV_FILE = "/home/jcneves/Documents/Identifying-Exoplanets-Using-ML/src/q1_q17_dr24_tce_2020.01.28_08.52.13.csv"
DATA_DIRECTORY = "/home/jcneves/Documents/keplerData"

def auc_roc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value

def model_creator(train_X_global,ls_units, dense_units, dropout_d, dropout_l, learn_rate, momentum):
    input = Input(shape=(train_X_global.shape[1], 1))

    #model = LSTM(units=10,return_sequences=True)(input)
    model = LSTM(units=ls_units)(input)
    model = BatchNormalization()(model)
    model = Dropout(dropout_l)(model)
    model = Activation('relu')(model)

    model = Dense(units=dense_units)(model)
    model = BatchNormalization()(model)
    model = Dropout(dropout_d)(model)
    model = Activation('relu')(model)
    model = Dense(units=dense_units)(model)
    model = BatchNormalization()(model)
    model = Dropout(dropout_d)(model)
    model = Activation('relu')(model)

    out = Dense(1, activation='sigmoid')(model)

    model = Model(inputs=input, outputs=out)

    opt = optimizers.SGD(lr=learn_rate, decay=0.0001, momentum=momentum, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=opt,  metrics=['accuracy'])
    return model

def fits(train_X_global, train_Y_global, val_X_global, val_Y_global, test_X_global, test_Y_global, epochs, batch_size, ls_units, dense_units,dropout_d, dropout_l, learn_rate, momentum, train_X_local, train_Y_local, val_X_local, val_Y_local, test_X_local, test_Y_local ):
    model = model_creator(train_X_global,ls_units, dense_units, dropout_d, dropout_l, learn_rate, momentum )

    # Local or Global View
    model.fit(train_X_global, train_Y_global, batch_size=batch_size, epochs=epochs,
              validation_data=(val_X_global, val_Y_global),
              callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=1, mode='auto')])
    score = model.evaluate(test_X_global, test_Y_global, verbose=1)[1]

    # Local and Global View
    #model.fit([train_X_global,train_X_local], train_Y_global, batch_size=batch_size, epochs=epoch,validation_data=([val_X_global,val_X_local], val_Y_global), callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')])
    #score = model.evaluate([test_X_global,test_X_local], test_Y_global, verbose=0)[1]

    return score



def main():
    start = t.time()

    experiment = Experiment("hMRp4uInUqRHs0pHtHFTl6jUL")

    # Data For The Sequential 1D-CNN
    data_local = np.loadtxt('neural_input_local_sovgol.csv', delimiter=',')
    local_X = data_local[0:, 1:-1]  # Input
    local_Y = data_local[0:, -1]  # Labels
    scaler_local = MinMaxScaler(feature_range=(0, 1))   # Scale Values
    rescaled_local_X = scaler_local.fit_transform(local_X)

    data_global = np.loadtxt('neural_input_global_sovgol.csv', delimiter=',')
    global_X = data_global[0:, 1:-1]  # Input
    global_Y = data_global[0:, -1]  # Labels
    scaler_global = MinMaxScaler(feature_range=(0, 1))  # Scale Values
    rescaled_global_X = scaler_global.fit_transform(global_X)

    # Separate Data
    train_X_local, val_X_local, test_X_local = np.split(rescaled_local_X, [int(.8 * len(rescaled_local_X)), int(0.9 * len(rescaled_local_X))])  # Training = 80%, Validation = 10%, Test = 10%
    train_Y_local, val_Y_local, test_Y_local = np.split(local_Y, [int(.8 * len(local_Y)), int(0.9 * len(local_Y))])
    #print("Total: {} ; Training: {} ; Evaluation: {} ; Test: {}".format(len(local_X),len(train_X_local),len(val_X_local),len(test_X_local)))

    train_X_global, val_X_global, test_X_global = np.split(rescaled_global_X, [int(.8 * len(rescaled_global_X)), int(0.9 * len(rescaled_global_X))])  # Training = 80%, Validation = 10%, Test = 10%
    train_Y_global, val_Y_global, test_Y_global = np.split(global_Y, [int(.8 * len(global_Y)), int(0.9 * len(global_Y))])
    #print("Total: {} ; Training: {} ; Evaluation: {} ; Test: {}".format(len(global_X),len(train_X_global),len(val_X_global),len(test_X_global)))

    # Shape Data
    train_X_global = np.expand_dims(train_X_global, axis=2)
    val_X_global = np.expand_dims(val_X_global, axis=2)
    test_X_global = np.expand_dims(test_X_global, axis=2)
    train_X_local = np.expand_dims(train_X_local, axis=2)
    val_X_local = np.expand_dims(val_X_local, axis=2)
    test_X_local = np.expand_dims(test_X_local, axis=2)

    batch_size = 32
    epochs = 20
    ls_units = 5
    dense_units = 64
    dropout_d = 0.2
    dropout_l = 0.2
    learn_rate = 0.03
    momentum = 0.2
    params = {'batch_size': batch_size,
              'epochs': epochs,
              'ls_units': ls_units,
              'dense_units': dense_units,
              'dropout_d': dropout_d,
              'dropout_l': dropout_l,
              'learn_rate': learn_rate,
              'momentum': momentum,
              }

    config = {
        "algorithm": "bayes",
        "name": "Optimize LSTM Network",
        "spec": {"maxCombo": 0, "objective": "minimize", "metric": "loss"},
        "parameters": {
            "batch_size": {"type": "discrete", "values":[16,32]},
            "epochs": {"type": "integer", "min": 10, "max": 50},
            "ls_units": {"type": "discrete", "values":[2,5,10,15]},
            "dense_units": {"type": "discrete","values": [32,64,128]},
            "dropout_d": {"type": "float","scalingType":"uniform", "min": 0.0, "max": 0.5},
            "dropout_l": {"type": "float", "scalingType": "uniform", "min": 0.0, "max": 0.5},
            "learn_rate": {"type": "float","scalingType":"loguniform", "min": 0.01, "max": 0.5},
            "momentum": {"type": "float","scalingType":"uniform", "min":0.0, "max": 0.5},
        },
        "trials": 1,
    }



    opt = Optimizer(config, api_key="hMRp4uInUqRHs0pHtHFTl6jUL", project_name="lstm1")

    for experiment in opt.get_experiments():
        epochs = experiment.get_parameter("epochs")
        batch_size = experiment.get_parameter("batch_size")
        ls_units = experiment.get_parameter("ls_units")
        dense_units = experiment.get_parameter("dense_units")
        dropout_d = experiment.get_parameter("dropout_d")
        dropout_l = experiment.get_parameter("dropout_l")
        learn_rate = experiment.get_parameter("learn_rate")
        momentum = experiment.get_parameter("momentum")

        acc = fits(train_X_global, train_Y_global, val_X_global, val_Y_global, test_X_global, test_Y_global,
                  epochs, batch_size, ls_units, dense_units, dropout_d, dropout_l, learn_rate, momentum, train_X_local, train_Y_local, val_X_local, val_Y_local, test_X_local, test_Y_local )
        # Reverse the score for minimization
        experiment.log_metric("accuracy", acc)

    experiment.log_parameters(params)

    # Train And Evaluate Model
    #model = model_creator(train_X_global)
    #model.fit(train_X_global, train_Y_global, batch_size=16, epochs=43, validation_data=(val_X_global, val_Y_global), callbacks=[EarlyStopping(monitor='roc_auc', min_delta=0, patience=2, verbose=1, mode='max')])
    #score = model.evaluate(test_X_global, test_Y_global, verbose=0)[1]
    #print("Test Accuracy = {}".format(score))

main()