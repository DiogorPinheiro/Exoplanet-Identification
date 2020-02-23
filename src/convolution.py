import numpy as np
import pickle
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.convolutional import Conv1D
from keras.layers import GlobalMaxPooling1D
import lightkurve as lk
import pandas as pd

from dataFunctions import dataCleaner
from dataFunctions import dataInfo
from dataFunctions import dataAnalyzer
from dataFunctions import dataReader

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

def rawCNN(n_timesteps, n_features):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
    model.add(Conv1D(filters=64, activation='sigmoid'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1, activation='sigmoid'))  # Output Layer - > Sigmoid Function



def main():
    table = getCSVData().drop_duplicates()
    #kepids = getKepids(table)  # List of Kepids
    #dataReader.createFluxDatabase(table,kepids,DATA_DIRECTORY)

    # Raw Data For The Sequential 1D-CNN
    #data = pickle.load(open( "concateneted_flux.p", "rb" ))
    #data = np.asarray(data)

    # Get Labels
    #data_y = np.loadtxt('dataset.csv', delimiter=',', skiprows=1)
    #Y = data_y[0:, -1]  # Labels
    #normalized_data=normalizeData(data)
    #print(len(data))
    #print(len(normalized_data))

    #for col in data:
    #    print(len(col))
    df = pd.read_csv("neural_input_global.csv", sep=",")
    df.drop_duplicates(subset=None, inplace=True)
    df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
    # Write the my_dataframe.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)results to a different file
    df.to_csv("neural_input_global.csv")

    #train_X, val_X, test_X = np.split(normalized_data, [int(.8 * len(normalized_data)), int(0.9 * len(normalized_data))])  # Training = 80%, Validation = 10%, Test = 10%
    #train_Y, val_Y, test_Y = np.split(Y, [int(.8 * len(Y)), int(0.9 * len(Y))])

main()