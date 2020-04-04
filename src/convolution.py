import numpy as np
from comet_ml import Experiment, Optimizer
import tensorflow as tf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Input, concatenate, Flatten, Dropout, PReLU, BatchNormalization, Activation, GaussianNoise
from keras.layers.convolutional import Conv1D
from keras.layers import MaxPooling1D
from keras.models import Model
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler
import time as t
import pandas as pd
from keras import backend as K
import csv

from dataFunctions import dataInfo
from evaluation import f1_m, precision_m, recall_m, mainEvaluate, auc_roc
import provedModels as pm
from utilities import writeToFile, joinLists
from otherAlgorithms import *

CSV_FILE = "/home/jcneves/Documents/Identifying-Exoplanets-Using-ML/src/q1_q17_dr24_tce_2020.01.28_08.52.13.csv"
DATA_DIRECTORY = "/home/jcneves/Documents/keplerData"

# -------------------------------- Utilities ------------------------------------------------


def getKepids(table):
    '''
        Get All Kepids In The CSV File

        Output: List of Kepids Numbers (List of Int)
    '''
    return dataInfo.listKepids(table)


def getCSVData():
    '''
        Call dataCSV Function To Get The Data Of CSV File

        Output: Pandas Dataframe
    '''
    return dataInfo.dataCSV(CSV_FILE)

# ---------------------------------- Models ----------------------------------------------------


def bothViewsCNN(x_train_local, x_train_global, lay1_filters, l1_kernel_size, pool_size, strides, conv_dropout, lay2_filters, l2_kernel_size, dense_f, dense_dropout):
    # CNN Model
    print(x_train_local.shape)
    print(x_train_global.shape)
    inputLayer_local = Input(shape=(x_train_local.shape[1], 1))
    inputLayer_global = Input(shape=(x_train_global.shape[1], 1))

    conv_local = Conv1D(16, kernel_size=5, strides=1, padding='same', dilation_rate=1,
                        activation='relu', kernel_initializer='he_normal')
    conv_global = Conv1D(16,  kernel_size=5, strides=1, padding='same', dilation_rate=1,
                         activation='relu', kernel_initializer='he_normal')

    # Input1
    model1 = conv_global(inputLayer_global)  # Disjoint Conv Layer
    model1 = Conv1D(16,  kernel_size=5, strides=1, padding='same',
                    dilation_rate=1, activation='relu')(model1)
    model1 = MaxPooling1D(pool_size=5, strides=2, padding='valid')(model1)
    #model1 = Dropout(0.20)(model1)
    model1 = GaussianNoise(0.1)(model1)
    model1 = Conv1D(32,  kernel_size=5, strides=1, padding='same',
                    dilation_rate=1, activation='relu')(model1)  # Disjoint Conv Layer
    model1 = Conv1D(32,  kernel_size=5, strides=1, padding='same',
                    dilation_rate=1, activation='relu')(model1)
    model1 = MaxPooling1D(pool_size=5, strides=2, padding='valid')(model1)
    #model1 = Dropout(0.20)(model1)
    model1 = GaussianNoise(0.1)(model1)
    model1 = Conv1D(64,  kernel_size=5, strides=1, padding='same',
                    dilation_rate=1, activation='relu')(model1)  # Disjoint Conv Layer
    model1 = Conv1D(64,  kernel_size=5, strides=1, padding='same',
                    dilation_rate=1, activation='relu')(model1)
    model1 = MaxPooling1D(pool_size=5, strides=2, padding='valid')(model1)
    #model1 = Dropout(0.20)(model1)
    model1 = GaussianNoise(0.1)(model1)
    model1 = Conv1D(128,  kernel_size=5, strides=1, padding='same',
                    dilation_rate=1, activation='relu')(model1)  # Disjoint Conv Layer
    model1 = Conv1D(128,  kernel_size=5, strides=1, padding='same',
                    dilation_rate=1, activation='relu')(model1)
    model1 = MaxPooling1D(pool_size=5, strides=2, padding='valid')(model1)
    #model1 = Dropout(0.20)(model1)
    model1 = GaussianNoise(0.1)(model1)
    model1 = Conv1D(256,  kernel_size=5, strides=1, padding='same',
                    dilation_rate=1, activation='relu')(model1)  # Disjoint Conv Layer
    model1 = Conv1D(256,  kernel_size=5, strides=1, padding='same',
                    dilation_rate=1, activation='relu')(model1)
    model1 = MaxPooling1D(pool_size=5, strides=2, padding='valid')(model1)
    #model1 = Dropout(0.20)(model1)
    model1 = GaussianNoise(0.1)(model1)
    model1 = Flatten()(model1)

    # Input2
    model2 = conv_local(inputLayer_local)
    model2 = Conv1D(16,  kernel_size=5, strides=1, padding='same',
                    dilation_rate=1, activation='relu')(model2)  # Disjoint Conv Layer
    model2 = MaxPooling1D(pool_size=7, strides=2, padding='valid')(model2)
    #model2 = Dropout(0.20)(model2)
    model2 = GaussianNoise(0.1)(model2)
    model2 = Conv1D(32,  kernel_size=5, strides=1, padding='same',
                    dilation_rate=1, activation='relu')(model2)  # Disjoint Conv Layer
    model2 = Conv1D(32,  kernel_size=5, strides=1, padding='same',
                    dilation_rate=1, activation='relu')(model2)
    model2 = MaxPooling1D(pool_size=7, strides=2, padding='valid')(model2)
    #model2 = Dropout(0.20)(model2)
    model2 = GaussianNoise(0.1)(model2)
    model2 = Flatten()(model2)
    # Concatenation
    concatLayerQ = keras.layers.concatenate(
        [model1, model2], axis=1)  # Concatenate Layer
    #flatLayerQ = Flatten()(concatLayerQ)

    # Fully-Connected Layers
    denseLayerQ = Dense(512, activation='relu')(concatLayerQ)
    denseLayerQ = Dense(512, activation='relu')(denseLayerQ)
    denseLayerQ = BatchNormalization()(denseLayerQ)
    #denseLayerQ = GaussianNoise(0.1)(denseLayerQ)
    #denseLayerQ = Dropout(0.20)(denseLayerQ)
    denseLayerQ = Dense(512, activation='relu')(denseLayerQ)
    denseLayerQ = Dense(512, activation='relu')(denseLayerQ)
    denseLayerQ = BatchNormalization()(denseLayerQ)
    #denseLayerQ = GaussianNoise(0.1)(denseLayerQ)
    #denseLayerQ = Dropout(0.20)(denseLayerQ)

    outputLayer = Dense(1, activation='sigmoid')(denseLayerQ)  # Output Layer

    model = Model(inputs=[inputLayer_local,
                          inputLayer_global], outputs=outputLayer)

    #opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    opt = optimizers.Adam(learning_rate=10e-5, beta_1=0.9,
                          beta_2=0.999, amsgrad=False)
    model.compile(loss='binary_crossentropy', optimizer=opt,
                  metrics=['accuracy', auc_roc])

    return model


def functionalCNN(x_train_global):
    var = Input(shape=(x_train_global.shape[1], 1))
    model = Conv1D(filters=128, kernel_size=3, padding='same')(var)
    model = BatchNormalization()(model)
    model = MaxPooling1D(pool_size=2, strides=2)(model)
    model = Dropout(0.20)(model)
    model = Activation('relu')(model)
    model = Conv1D(filters=128, kernel_size=3, padding='same')(model)
    model = BatchNormalization()(model)
    model = MaxPooling1D(pool_size=2, strides=2)(model)
    model = Dropout(0.20)(model)
    model = Activation('relu')(model)
    model = Flatten()(model)
    model = Dense(64)(model)
    model = Dropout(0.20)(model)
    model = Activation('relu')(model)
    out = Dense(1, activation='sigmoid')(model)
    model = Model(inputs=var, outputs=out)

    opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=opt,
                  metrics=['accuracy', f1_m, precision_m, recall_m])

    return model


def triage(experiment, train_X_global, train_Y_global, val_X_global, val_Y_global, test_X_global, test_Y_global, epoch, batch_size, lay1_filters, l1_kernel_size, pool_size, strides, conv_dropout, lay2_filters, l2_kernel_size, dense_f, dense_dropout, x_train_global, train_X_local, train_Y_local, val_X_local, val_Y_local, test_X_local, test_Y_local, x_train_local):
    #model = seqModelCNN(lay1_filters,l1_kernel_size,pool_size,strides,conv_dropout,lay2_filters,l2_kernel_size,dense_f,dense_dropout,x_train_global)
    model = bothViewsCNN(train_X_global, train_X_local, lay1_filters, l1_kernel_size, pool_size,
                         strides, conv_dropout, lay2_filters, l2_kernel_size, dense_f, dense_dropout)
    '''   with experiment.train():
        history = model.fit(train_X_global, train_Y_global,
                            batch_size=batch_size,
                            epochs=epoch,
                            verbose=1,
                            validation_data=(val_X_global, val_Y_global),
                            callbacks=[
                                EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=3, verbose=1, mode='auto')])

    # will log metrics with the prefix 'test_'
    with experiment.test():
        loss, accuracy = model.evaluate(test_X_global, test_Y_global)
        metrics = {
            'loss': loss,
            'accuracy': accuracy
        }
        experiment.log_metrics(metrics)

    experiment.log_dataset_hash(train_X_global)
    '''
    # Local or Global View
    model.fit(train_X_global, train_Y_global, batch_size=batch_size, epochs=epoch, validation_data=(val_X_global,
                                                                                                    val_Y_global), callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')])
    score = model.evaluate(test_X_global, test_Y_global, verbose=0)[1]

    # Local and Global View
    #model.fit([train_X_global,train_X_local], train_Y_global, batch_size=batch_size, epochs=epoch,validation_data=([val_X_global,val_X_local], val_Y_global), callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')])
    #score = model.evaluate([test_X_global,test_X_local], test_Y_global, verbose=0)[1]

    return score


def seqModelCNN(lay1_filters, l1_kernel_size, pool_size, strides, conv_dropout, lay2_filters, l2_kernel_size, dense_f, dense_dropout, x_train):
    model = Sequential()
    model.add(Conv1D(filters=128, kernel_size=3, input_shape=(x_train.shape[1], 1),
                     padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(Dropout(0.20))
    model.add(PReLU())
    model.add(Conv1D(filters=128, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(Dropout(0.2))
    model.add(PReLU())
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Dropout(0.2))
    model.add(PReLU())
    model.add(Dense(1, activation='sigmoid'))

    opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=opt,
                  metrics=['accuracy', f1_m, precision_m, recall_m])

    return model


def main():
    start = t.time()

    #experiment = Experiment("hMRp4uInUqRHs0pHtHFTl6jUL")

    table = getCSVData().drop_duplicates()
    kepids = getKepids(table).drop_duplicates(
    ).reset_index(drop=True)  # List of Kepids
    # dataReader.createFluxDatabase(table,kepids,DATA_DIRECTORY)

    # Data For The Sequential 1D-CNN
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

    # Machine Learning Algorithms
    #model = knn()
    #model = svmachine()
    #model = feedForwardNN(X_train_global, X_train_local_shaped)

    # Deep Learning Models
    #model = seqModelCNN(0, 0, 0, 0, 0, 0, 0, 0, 0, X_train_global_shaped)
    #model = bothViewsCNN(X_train_global, X_train_local_shaped,0, 0, 0, 0, 0, 0, 0, 0, 0)
    #model = functionalCNN(X_train_global)

    # Proved Deep Learning Models
    #model, type = pm.vgg(X_train_global)
    #model, type = pm.alexNet(X_train_global)

    # Evaluation
    split = 5
    epoch = 32
    batch = 50
    nb = 5
    # md, hist_lo = mainEvaluate('single-global',model,X_train_global,X_train_local_shaped,X_test_global_shaped,X_test_local_shaped,y_train_global,y_test_global,nb,epoch,batch,split,'functional')
    # md, hist_lo = mainEvaluate('single-global',model,X_train_global,X_train_local_shaped,X_test_global_shaped,X_test_local_shaped,y_train_global,y_test_global,nb,epoch,batch,split,'sequential')
    # md, hist_lo = mainEvaluate('single-local',model,X_train_global,X_train_local_shaped,X_test_global_shaped,X_test_local_shaped,y_train_global,y_test_global,nb,epoch,batch,split,'functional')
    # md, hist_lo = mainEvaluate('single-local',model,X_train_global,X_train_local_shaped,X_test_global_shaped,X_test_local_shaped,y_train_global,y_test_global,nb,epoch,batch,split,'sequential')
    # md, hist_lo = mainEvaluate('dual',model,X_train_global,X_train_local_shaped,X_test_global_shaped,X_test_local_shaped,y_train_global,y_test_global,nb,epoch,batch,split,'functional')
    # md, hist_lo = mainEvaluate('dual', model, X_train_global_shaped, X_train_local_shaped, X_test_global_shaped, X_test_local_shaped, y_train_global, y_test_global, nb, epoch, batch, split, 'sequential')

    # Table Comparison
    ffnnlist = []
    cnnglobal = []
    cnndual = []

    #model = knn()
    # score = mainEvaluate('simple-local', model, X_train_global, X_train_local, X_test_global,
    #                     X_test_local, y_train_global, y_test_global, nb, epoch, batch, split, 'functional')
    #print("KNN : {}".format(score))
    #model = svmachine()
    # score = mainEvaluate('simple-local', model, X_train_global, X_train_local, X_test_global,
    #                     X_test_local, y_train_global, y_test_global, nb, epoch, batch, split, 'functional')
    #print("SVM : {}".format(score))

    #model = feedForwardNN(X_train_global, X_train_local)
    # md, hist_lo = mainEvaluate('dual-fnn', model, X_train_global_shaped, X_train_local_shaped, X_test_global_shaped,
    #                           X_test_local_shaped, y_train_global, y_test_global, nb, epoch, batch, split, 'functional')
    # ffnnlist.append(hist_lo)

    # model = bothViewsCNN(X_train_global_shaped, X_train_local_shaped,
    #                     0, 0, 0, 0, 0, 0, 0, 0, 0)
    # md, hist_lo = mainEvaluate('dual', model, X_train_global_shaped, X_train_local_shaped, X_test_global_shaped,
    #                           X_test_local_shaped, y_train_global, y_test_global, nb, epoch, batch, split, 'functional')
    # cnndual.append(hist_lo)

    #model = seqModelCNN(0, 0, 0, 0, 0, 0, 0, 0, 0, X_train_global_shaped)
    # md, hist_lo = mainEvaluate('single-local', model, X_train_global_shaped, X_train_local_shaped, X_test_global_shaped,
    #                           X_test_local_shaped, y_train_global, y_test_global, nb, epoch, batch, split, 'sequential')
    # cnnglobal.append(hist_lo)

    #l1 = joinLists(ffnnlist, cnnglobal)
    #l1 = joinLists(l1, cnndual)
    # writeToFile(l1)

    model, seq = pm.alexNet(X_train_local_shaped)
    md, hist_lo = mainEvaluate('single-local', model, X_train_global_shaped, X_train_local_shaped, X_test_global_shaped,
                               X_test_local_shaped, y_train_global, y_test_global, nb, epoch, batch, split, seq)


'''
    batch_size = 128
    epochs = 20
    lay1_filters=128
    lay2_filters = 128
    l1_kernel_size = 3
    l2_kernel_size = 3
    pool_size = 2
    conv_dropout=0.2
    dense_dropout=0.2
    strides = 2
    dense_f=64
    optimizer = 'adam'
    activation = 'relu'
    params = {'batch_size': batch_size,
              'epochs': epochs,
              'lay1_filters':lay1_filters,
              'lay2_filters': lay2_filters,
              'layer1_type': 'Dense',
              'conv_dropout':conv_dropout,
              'dense_dropout': dense_dropout,
              'l1_kernel_size':l1_kernel_size,
              'l2_kernel_size': l2_kernel_size,
              'dense_f':dense_f,
              'strides':strides,
              'pool_size':pool_size,
              'layer1_activation': activation,
              'optimizer': optimizer
              }


    #model = seqModelCNN(lay1_filters,l1_kernel_size,pool_size,strides,conv_dropout,lay2_filters,l2_kernel_size,dense_f,dense_dropout,x_train_global)

    #opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    train_X_global = np.expand_dims(train_X_global, axis=2)
    val_X_global = np.expand_dims(val_X_global, axis=2)
    test_X_global = np.expand_dims(test_X_global, axis=2)
    #model.fit(train_X_global, train_Y_global)
    train_X_local = np.expand_dims(train_X_local, axis=2)
    val_X_local = np.expand_dims(val_X_local, axis=2)
    test_X_local = np.expand_dims(test_X_local, axis=2)

    config = {
        "algorithm": "bayes",
        "name": "Optimize CNN Network",
        "spec": {"maxCombo": 3, "objective": "minimize", "metric": "loss"},
        "parameters": {
            "batch_size": {"type": "integer", "min": 20, "max": 254},
            "epochs":{"type": "integer", "min": 10, "max": 40},
            "lay1_filters": {"type": "integer", "min": 32, "max": 254},
            "lay2_filters": {"type": "integer", "min": 32, "max": 254},
            "l1_kernel_size": {"type": "integer", "min": 1, "max": 5},
            "l2_kernel_size": {"type": "integer", "min": 1, "max": 5},
            "pool_size": {"type": "integer", "min": 2, "max": 4},
            "conv_dropout": {"type": "float",  "min": 0.01, "max": 0.3},
            "dense_dropout": {"type": "float",  "min": 0.01, "max": 0.3},
            "strides": {"type": "integer", "min": 2, "max": 4},
            "dense_f": {"type": "integer", "min": 8, "max": 254},
        },
        "trials": 1,
    }

    opt = Optimizer(config, api_key="hMRp4uInUqRHs0pHtHFTl6jUL", project_name="cnn-doubleinput")

    for experiment in opt.get_experiments():
        epochs = experiment.get_parameter("epochs")
        batch_size = experiment.get_parameter("batch_size")
        lay1_filters = experiment.get_parameter("lay1_filters")
        l1_kernel_size = experiment.get_parameter("l1_kernel_size")
        l2_kernel_size = experiment.get_parameter("l2_kernel_size")
        pool_size = experiment.get_parameter("pool_size")
        strides = experiment.get_parameter("strides")
        conv_dropout = experiment.get_parameter("conv_dropout")
        lay2_filters = experiment.get_parameter("lay2_filters")
        dense_f = experiment.get_parameter("dense_f")
        dense_dropout = experiment.get_parameter("dense_dropout")

        acc = fit(experiment, train_X_global, train_Y_global, val_X_global, val_Y_global, test_X_global, test_Y_global, epochs, batch_size,lay1_filters,l1_kernel_size,pool_size,strides,conv_dropout,lay2_filters,l2_kernel_size,dense_f,dense_dropout,x_train_global, train_X_local, train_Y_local, val_X_local, val_Y_local, test_X_local, test_Y_local,x_train_local )
        # Reverse the score for minimization
        experiment.log_metric("accuracy", acc)


    experiment.log_parameters(params)

    #training(model, train_X_global, train_Y_global, val_X_global, val_Y_global, nb_cv = 5, batch_size = 10, nb_epochs = 2)

    end = t.time()
    #print(end - start)
'''
main()
