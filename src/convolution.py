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

def getConcatenatedLightCurve(flux,time):
    '''
        Concatenate all Light Curves associated with the kepid and normalize values

        Input: kepid number
        Output: normalized concatenated light curve
    '''
    for i in flux:  # Same scale for all segments
        i /= np.median(i)
    out_flux = np.concatenate(flux)
    out_time = np.concatenate(time)
    normalized_flux = dataCleaner.movingAverage(out_flux, 15)
    lc = lk.LightCurve(out_time, normalized_flux)
    ret = lc.normalize()
    return ret.flux, out_time

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

main()