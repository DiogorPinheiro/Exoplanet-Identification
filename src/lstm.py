import numpy as np
from comet_ml import Experiment, Optimizer
import tensorflow as tf
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Input, concatenate, Flatten, Dropout, PReLU, BatchNormalization, Activation, LSTM, CuDNNLSTM
from keras.initializers import Ones, Orthogonal, VarianceScaling, Zeros
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras import optimizers
from keras.layers import MaxPooling1D
from sklearn.preprocessing import MinMaxScaler
import time as t
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from keras import backend as K
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score

from evaluation import f1_m, precision_m, recall_m, mainEvaluate, auc_roc
from dataFunctions import dataInfo

CSV_FILE = "/home/jcneves/Documents/Identifying-Exoplanets-Using-ML/src/q1_q17_dr24_tce_2020.01.28_08.52.13.csv"
DATA_DIRECTORY = "/home/jcneves/Documents/keplerData"

def singleLSTModel(train_X_global, ls_units, dense_units, dropout_d, dropout_l, learn_rate, momentum):
    model = Sequential()
    model.add(LSTM(ls_units, input_shape = (train_X_global.shape[1], 1), return_sequences=True))
    #model.add(LSTM(ls_units, input_shape = (train_X_global.shape[1], 1)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_l))
    model.add(PReLU())
    model.add(LSTM(ls_units))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_l))
    model.add(PReLU())

    model.add(Dense(dense_units))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_d))
    model.add(PReLU())
    model.add(Dense(dense_units))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_d))
    model.add(PReLU())

    model.add(Dense(1,activation='sigmoid'))

    opt = optimizers.SGD(lr=learn_rate, decay=0.0001,
                         momentum=momentum, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=opt,
                  metrics=['accuracy', f1_m, precision_m, recall_m])
    return model

def model_creator(train_X_global, ls_units, dense_units, dropout_d, dropout_l, learn_rate, momentum):
    input = Input(shape=(train_X_global.shape[1], 1))

    model = LSTM(units=ls_units,return_sequences=True)(input)
    #model = LSTM(units=ls_units)(input)
    model = BatchNormalization()(model)
    model = Dropout(dropout_l)(model)
    model = PReLU()(model)
    model = LSTM(units=ls_units)(model)
    model = BatchNormalization()(model)
    model = Dropout(dropout_l)(model)
    model = PReLU()(model)

    #model=Flatten()(model)

    model = Dense(units=dense_units)(model)
    #model = VarianceScaling(scale=1.0,mode='fan_avg', distribution='uniform')(model)
    #model = Zeros()(model)
    print(model.summary())
    model = MaxPooling1D(pool_size=2, strides=2)(model)
    model = Dropout(dropout_d)(model)
    model = PReLU()(model)
    #model = Zeros()(model)
    #model = Dense(units=dense_units)(model)
    #model = Dropout(dropout_d)(model)
    #model = BatchNormalization()(model)
    #model = VarianceScaling(scale=1.0,mode='fan_avg', distribution='uniform')(model)
    #model = Zeros()(model)
    #model = PReLU()(model)
    #model = Flatten()(model)

    out = Dense(1, activation='sigmoid')(model)

    model = Model(inputs=input, outputs=out)

    opt = optimizers.SGD(lr=0.01*learn_rate, decay=0.0001,
                         momentum=momentum, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=opt,
                  metrics=['accuracy', f1_m, precision_m, recall_m])
    return model


def fits(train_X_global, train_Y_global, val_X_global, val_Y_global, test_X_global, test_Y_global, epochs, batch_size, ls_units, dense_units, dropout_d, dropout_l, learn_rate, momentum, train_X_local, train_Y_local, val_X_local, val_Y_local, test_X_local, test_Y_local):
    model = model_creator(train_X_global, ls_units, dense_units,
                          dropout_d, dropout_l, learn_rate, momentum)

    # Local or Global View
    model.fit(train_X_global, train_Y_global, batch_size=batch_size, epochs=epochs,
              validation_data=(val_X_global, val_Y_global),
              callbacks=[EarlyStopping(monitor='val_loss', min_delta=1, patience=10, verbose=1, mode='min')])
    score = model.evaluate(test_X_global, test_Y_global, verbose=1)[1]

    # Local and Global View
    #model.fit([train_X_global,train_X_local], train_Y_global, batch_size=batch_size, epochs=epoch,validation_data=([val_X_global,val_X_local], val_Y_global), callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')])
    #score = model.evaluate([test_X_global,test_X_local], test_Y_global, verbose=0)[1]

    return score


def main():
    start = t.time()

    #experiment = Experiment("hMRp4uInUqRHs0pHtHFTl6jUL")

    data_local = np.loadtxt('data/neural_input_local.csv', delimiter=',')
    local_X = data_local[0:, 1:-1]  # Input
    local_Y = data_local[0:, -1]  # Labels

    data_global = np.loadtxt('data/neural_input_global.csv', delimiter=',')
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
    model = singleLSTModel(X_train_global_shaped, 10, 64, 0.298,
                          0.298, 0.0016434, 0.25)

    # Evaluation
    split = 5
    epoch = 43
    batch = 16
    nb = 5

    md, hist_lo = mainEvaluate('single-global', model, X_train_global_shaped, X_train_local_shaped, X_test_global_shaped,
                               X_test_local_shaped, y_train_global, y_test_global, nb, epoch, batch, split, 'functional')
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
        experiment.log_metric("loss", acc)

    experiment.log_parameters(params)
    '''

    # Train And Evaluate Model
    #model = model_creator(train_X_global)
    #model.fit(train_X_global, train_Y_global, batch_size=16, epochs=43, validation_data=(val_X_global, val_Y_global), callbacks=[EarlyStopping(monitor='roc_auc', min_delta=0, patience=2, verbose=1, mode='max')])
    #score = model.evaluate(test_X_global, test_Y_global, verbose=0)[1]
    #print("Test Accuracy = {}".format(score))


main()
