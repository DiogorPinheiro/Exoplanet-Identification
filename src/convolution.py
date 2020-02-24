import numpy as np
import pickle
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Input, concatenate, Flatten
from keras.layers.convolutional import Conv1D
from keras.layers import MaxPooling1D
from keras.models import Model
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler

from dataFunctions import dataInfo

CSV_FILE = "/home/jcneves/Documents/Identifying-Exoplanets-Using-ML/src/q1_q17_dr24_tce_2020.01.28_08.52.13.csv"
DATA_DIRECTORY = "/home/jcneves/Documents/keplerData"

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

def normalizeData(data):
    '''
        Read CSV File And Normalize Each Column (Except Kepid and Label)

        Replaces Values In CSV File With Their Normalized Version
    '''

    normalized_results=[]

    i=1
    for col in data:
        x_array = np.array(col)
        #print("Array:{}".format(x_array))
        result = preprocessing.normalize([x_array])
        #print(val)
        normalized_results.append(result)


    return normalized_results

def main():
    table = getCSVData().drop_duplicates()
    kepids = getKepids(table).drop_duplicates().reset_index(drop=True)  # List of Kepids
    #dataReader.createFluxDatabase(table,kepids,DATA_DIRECTORY)

    # Data For The Sequential 1D-CNN
    data_local = np.loadtxt('neural_input_local.csv', delimiter=',')
    local_X = data_local[0:, 1:-1]  # Input
    local_Y = data_local[0:, -1]  # Labels
    scaler_local = MinMaxScaler(feature_range=(0, 1))   # Scale Values
    rescaled_local_X = scaler_local.fit_transform(local_X)

    data_global = np.loadtxt('neural_input_global.csv', delimiter=',')
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
    #print(train_X_local.shape)
    x_train_local = train_X_local.reshape(train_X_local.shape[0], 1, train_X_local.shape[1])
    #print(x_train_local.shape)

    #print(train_X_global.shape)
    x_train_global = train_X_global.reshape(train_X_global.shape[0], 1, train_X_global.shape[1])
    #print(x_train_global.shape)

    # CNN Model
    inputLayer_local = Input(shape=x_train_local.shape)
    inputLayer_global = Input(shape=x_train_global.shape)

    conv_local = Conv1D(201, 10, strides=1, input_shape=x_train_local.shape, padding='same', dilation_rate=1, activation='relu')
    conv_global = Conv1D(2001, 10,  strides=1, input_shape=x_train_local.shape, padding='same', dilation_rate=1, activation='relu')

    convQ1 = conv_local(inputLayer_local)
    poolLayerQ1 = MaxPooling1D(pool_size=5, strides=1, padding='valid')(convQ1)
    convQ2 = conv_global(inputLayer_global)
    poolLayerQ2 = MaxPooling1D(pool_size=5, strides=1, padding='valid')(convQ2)

    concatLayerQ = concatenate([inputLayer_local, inputLayer_global], axis=1)
    flatLayerQ = Flatten()(concatLayerQ)
    denseLayerQ = Dense(10, activation='relu')(flatLayerQ)

    outputLayer = Dense(2, activation='sigmoid')(denseLayerQ)

    model = Model(inputs=[inputLayer_local, inputLayer_global], outputs=outputLayer)
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    model.fit(x_train, y_train,  epochs=20, batch_size=128)
    score = model.evaluate(x_test, y_test, batch_size=128)

main()