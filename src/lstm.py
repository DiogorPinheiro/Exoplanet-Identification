import numpy as np
from comet_ml import Experiment, Optimizer
import tensorflow as tf
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Input, concatenate, Flatten, Dropout, PReLU, BatchNormalization, Activation, LSTM
from keras.constraints import MaxNorm
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras import optimizers
from keras.layers import Bidirectional, TimeDistributed
from sklearn.preprocessing import MinMaxScaler
import time as t
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from keras import backend as K
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score

from training import f1_m, precision_m, recall_m, mainEvaluate, auc_roc
from utilities import writeToFile
from dataFunctions import dataInfo

CSV_FILE = "/home/jcneves/Documents/Identifying-Exoplanets-Using-ML/src/q1_q17_dr24_tce_2020.01.28_08.52.13.csv"
DATA_DIRECTORY = "/home/jcneves/Documents/keplerData"


def singleLSTModel(train_X_global, ls_units, dense_units, dropout_d, dropout_l, learn_rate, momentum):
    model = Sequential()
    model.add(LSTM(ls_units, activation='relu', return_sequences=True, recurrent_dropout=dropout_l,
                   unit_forget_bias=True, bias_initializer='zeros', input_shape=(train_X_global.shape[1], 1)))
    #model.add(Bidirectional(LSTM(ls_units, input_shape = (train_X_global.shape[1], 1))))
    model.add(BatchNormalization(center=False, scale=False))
    # model.add(PReLU())
    model.add(LSTM(ls_units, activation='relu', recurrent_dropout=dropout_l,
                   return_sequences=False, unit_forget_bias=True, bias_initializer='zeros'))
    model.add(BatchNormalization(center=False, scale=False))
    # model.add(PReLU())

    model.add(Dense(dense_units))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_d))
    model.add(PReLU())
    # model.add(Dense(dense_units))
    # model.add(BatchNormalization())
    # model.add(Dropout(dropout_d))
    # model.add(PReLU())

    model.add(Dense(1, activation='sigmoid'))

    opt = optimizers.SGD(lr=0.001*learn_rate, decay=1e-6,
                         momentum=momentum, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=opt,
                  metrics=['accuracy', f1_m, precision_m, recall_m, auc_roc])
    return model


def model_creator(train_X_global, ls_units, dense_units, dropout_d, dropout_l, learn_rate, momentum):
    input = Input(shape=(train_X_global.shape[1], 1))

    model = LSTM(units=ls_units, return_sequences=True,
                 unit_forget_bias=True, bias_initializer='zeros')(input)
    #model = LSTM(units=ls_units)(input)
    #model = BatchNormalization(center=False,scale=False)(model)
    model = PReLU()(model)
    model = Dropout(dropout_l)(model)
    model = LSTM(units=ls_units, unit_forget_bias=True,
                 bias_initializer='zeros', return_sequences=True)(model)
    #model = BatchNormalization(center=False,scale=False)(model)
    model = PReLU()(model)
    model = Dropout(dropout_l)(model)
    model = LSTM(units=ls_units, unit_forget_bias=True,
                 bias_initializer='zeros', return_sequences=True)(model)
    #model = BatchNormalization(center=False,scale=False)(model)
    model = PReLU()(model)
    model = Dropout(dropout_l)(model)

    model = Flatten()(model)

    model = Dense(units=dense_units)(model)
    #model = BatchNormalization()(model)
    model = PReLU()(model)
    model = Dropout(dropout_d)(model)
    model = Dense(units=dense_units)(model)
    #model = BatchNormalization()(model)
    model = PReLU()(model)
    model = Dropout(dropout_d)(model)

    out = Dense(1, activation='sigmoid')(model)

    model = Model(inputs=input, outputs=out)

    # opt = optimizers.SGD(lr=0.001*learn_rate, decay=0.0001,
    #                     momentum=momentum, nesterov=True)
    opt = optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='binary_crossentropy', optimizer=opt,
                  metrics=['accuracy', f1_m, precision_m, recall_m, auc_roc])
    return model


def fits(train_X_global, train_Y_global, test_X_global, test_Y_global, epochs, batch_size, ls_units, dense_units, dropout_d, dropout_l, learn_rate, momentum, train_X_local, train_Y_local,  test_X_local, test_Y_local):
    model = model_creator(train_X_global, ls_units, dense_units,
                          dropout_d, dropout_l, learn_rate, momentum)
    #model = singleLSTModel(train_X_global, ls_units, dense_units, dropout_l, dropout_d, learn_rate, momentum)
    # Local or Global View
    model.fit(train_X_global, train_Y_global, batch_size=batch_size, epochs=epochs,
              validation_data=(test_X_global, test_Y_global),
              callbacks=[EarlyStopping(monitor='val_auc_roc', min_delta=1, patience=10, verbose=1, mode='max')])
    score = model.evaluate(test_X_global, test_Y_global, verbose=1)[1]

    # Local and Global View
    #model.fit([train_X_global,train_X_local], train_Y_global, batch_size=batch_size, epochs=epoch,validation_data=([val_X_global,val_X_local], val_Y_global), callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')])
    #score = model.evaluate([test_X_global,test_X_local], test_Y_global, verbose=0)[1]

    return score


if __name__ == "__main__":
    start = t.time()

    #experiment = Experiment("hMRp4uInUqRHs0pHtHFTl6jUL")

    data_local = np.loadtxt(
        'data/Shallue/shallue_local.csv', delimiter=',', skiprows=1)
    local_X = data_local[0:, 1:-1]  # Input
    local_Y = data_local[0:, -1]  # Labels

    data_global = np.loadtxt(
        'data/Shallue/shallue_global.csv', delimiter=',', skiprows=1)
    global_X = data_global[0:, 1:-1]  # Input
    global_Y = data_global[0:, -1]  # Labels

    # Separate Local Data
    X_train_local, X_test_local, y_train_local, y_test_local = train_test_split(
        local_X, local_Y, test_size=0.2, random_state=1)

    # Scale Data
    scaler_local = MinMaxScaler(feature_range=(0, 1))  # Scale Values
    X_train_local = scaler_local.fit_transform(X_train_local)
    X_test_local = scaler_local.transform(X_test_local)

    global_X = data_global[0:, 1:-1]  # Input
    global_Y = data_global[0:, -1]  # Labels
    X_train_global, X_test_global, y_train_global, y_test_global = train_test_split(
        global_X, global_Y, test_size=0.2, random_state=1)

    scaler_global = MinMaxScaler(feature_range=(0, 1))  # Scale Values
    X_train_global = scaler_global.fit_transform(X_train_global)
    X_test_global = scaler_global.transform(X_test_global)

    # Shape Data
    X_train_global_shaped = np.expand_dims(X_train_global, axis=2)
    X_test_global_shaped = np.expand_dims(X_test_global, axis=2)
    X_train_local_shaped = np.expand_dims(X_train_local, axis=2)
    X_test_local_shaped = np.expand_dims(X_test_local, axis=2)

    # LSTM Neural Networks
    # model = singleLSTModel(X_train_global_shaped, 10, 64, 0.298,
    #                      0.298, 0.0016434, 0.25)

    # Evaluation
    split = 5
    epoch = 43
    batch = 32
    nb = 5

    # md, hist_lo = mainEvaluate('single-global', model, X_train_global_shaped, X_train_local_shaped, X_test_global_shaped,
    #                           X_test_local_shaped, y_train_global, y_test_global, nb, epoch, batch, split, 'functional')
    # md, hist_lo = mainEvaluate('single-global',model,X_train_global,X_train_local_shaped,X_test_global_shaped,X_test_local_shaped,y_train_global,y_test_global,nb,epoch,batch,split,'sequential')
    # md, hist_lo = mainEvaluate('single-local',model,X_train_global,X_train_local_shaped,X_test_global_shaped,X_test_local_shaped,y_train_global,y_test_global,nb,epoch,batch,split,'functional')
    # md, hist_lo = mainEvaluate('single-local',model,X_train_global,X_train_local_shaped,X_test_global_shaped,X_test_local_shaped,y_train_global,y_test_global,nb,epoch,batch,split,'sequential')
    # md, hist_lo = mainEvaluate('dual',model,X_train_global,X_train_local_shaped,X_test_global_shaped,X_test_local_shaped,y_train_global,y_test_global,nb,epoch,batch,split,'functional')
    # md, hist_lo = mainEvaluate('dual', model, X_train_global_shaped, X_train_local_shaped, X_test_global_shaped, X_test_local_shaped, y_train_global, y_test_global, nb, epoch, batch, split, 'sequential')
    '''
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
            "batch_size": {"type": "discrete", "values": [16, 32]},
            "epochs": {"type": "integer", "min": 10, "max": 50},
            "ls_units": {"type": "discrete", "values": [2, 5, 10, 15]},
            "dense_units": {"type": "discrete", "values": [32, 64, 128]},
            "dropout_d": {"type": "float", "scalingType": "uniform", "min": 0.0, "max": 0.5},
            "dropout_l": {"type": "float", "scalingType": "uniform", "min": 0.0, "max": 0.5},
            "learn_rate": {"type": "float", "scalingType": "loguniform", "min": 0.01, "max": 0.5},
            "momentum": {"type": "float", "scalingType": "uniform", "min": 0.0, "max": 0.5},
        },
        "trials": 1,
    }

    opt = Optimizer(config, api_key="hMRp4uInUqRHs0pHtHFTl6jUL",
                    project_name="lstm-draw1")

    for experiment in opt.get_experiments():
        epochs = experiment.get_parameter("epochs")
        batch_size = experiment.get_parameter("batch_size")
        ls_units = experiment.get_parameter("ls_units")
        dense_units = experiment.get_parameter("dense_units")
        dropout_d = experiment.get_parameter("dropout_d")
        dropout_l = experiment.get_parameter("dropout_l")
        learn_rate = experiment.get_parameter("learn_rate")
        momentum = experiment.get_parameter("momentum")

        #model = singleLSTModel(X_train_global_shaped, ls_units, dense_units, dropout_l, dropout_d, learn_rate, momentum)
        # md, hist_lo = mainEvaluate('single-global', model, X_train_global_shaped, X_train_local_shaped, X_test_global_shaped,
        #                       X_test_local_shaped, y_train_global, y_test_global, nb, epochs, batch_size, split, 'functional')
        acc = fits(X_train_global_shaped, y_train_global, X_test_global_shaped, y_test_global,
                   epochs, batch_size, ls_units, dense_units, dropout_d, dropout_l, learn_rate, momentum, X_train_local_shaped, y_train_local, X_test_local_shaped, y_test_local)
        # Reverse the score for minimization
        experiment.log_metric("loss", acc)

    experiment.log_parameters(params)
    '''
    # Train And Evaluate Model
    model = model_creator(X_train_global_shaped, 15,
                          128, 0.34, 0.09, 0.178, 0.49)
    model.fit(X_train_global_shaped, y_train_global, batch_size=32, epochs=43, validation_data=(X_test_global_shaped,
                                                                                                y_test_global), callbacks=[EarlyStopping(monitor='roc_auc', min_delta=0, patience=2, verbose=1, mode='max')])
    md, hist_lo, tens = mainEvaluate('single-global', model, X_train_global_shaped, X_train_local_shaped, X_test_global_shaped,
                                     X_test_local_shaped, y_train_global, y_test_global, 5, 32, 16, 5, 'functional', 'lstm.h5')  # print("Test Accuracy = {}".format(score))
    writeToFile("lstm_comp.csv", tens)
