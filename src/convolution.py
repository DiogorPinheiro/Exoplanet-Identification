import numpy as np
import pickle
import lightkurve as lk
import csv
from sklearn import preprocessing

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


def main():
    table = getCSVData().drop_duplicates()
    kepids = getKepids(table)  # List of Kepids

    dataReader.createFluxDatabase(table,kepids,DATA_DIRECTORY)
    # Create function for a functional 1D-CNN

main()