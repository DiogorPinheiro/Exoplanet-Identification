import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightkurve as lk

from dataFunctions import dataCleaner
from dataFunctions import dataInfo
from dataFunctions import dataAnalyzer
from dataFunctions import dataReader

CSV_FILE = "/home/jcneves/Documents/Identifying-Exoplanets-Using-ML/data/q1_q17_dr24_tce_2020.01.28_08.52.13.csv"
DATA_DIRECTORY = "/home/jcneves/Documents/keplerData"

def getCSVData():
    return dataInfo.dataCSV(CSV_FILE)

def getKepids(table):
    return dataInfo.listKepids(table)
    
def getLabel(table,kepid):
    return dataInfo.labelFinder(table,kepid)   

def createFeaturesTable(table,kepid):
    '''
        1. Read and Convert Data
        2. Clean and Normalize Light Curves
        3. Get All Features
            . Analyze Data
            . Label
        4. Return Row of Features
    '''
    filenames = dataReader.filenameWarehouse(kepid,DATA_DIRECTORY)
    time, flux = dataReader.fitsConverter(filenames)
    normalized_flux = dataCleaner.movingAverage(flux[0],20)
    lc=lk.LightCurve(time[0],normalized_flux)
    ret = lc.normalize()
    normalized_flux = ret.flux
    
    global_mean=dataInfo.getGlobalMean(normalized_flux)
    global_median=dataInfo.getGlobalMedian(normalized_flux)
    globa_std=dataInfo.getGlobalSTD(normalized_flux)
    
    strongPeaks, mediumPeaks, weakPeaks = dataInfo.categorizedPoints(normalized_flux)
    num_strongPeaks=len(strongPeaks)
    max_strongPeaks=max(strongPeaks)
    mean_strongPeaks=np.mean(strongPeaks)
    std_strongPeaks=np.std(strongPeaks)
    
    num_mediumPeaks=len(mediumPeaks)
    max_mediumPeaks=max(mediumPeaks)
    mean_mediumPeaks=np.mean(mediumPeaks)
    std_mediumPeaks=np.std(mediumPeaks)
    
    num_weakPeaks=len(weakPeaks)
    max_weakPeaks=max(weakPeaks)
    mean_weakPeaks=np.mean(weakPeaks)
    std_weakPeaks=np.std(weakPeaks)
    
    label=getLabel(table,kepid)
    
    
    
    
    
def appendToFile(row):
    with open('dataset.csv','a') as fd:
        fd.write(row)   

def main():
    # Create table
    with open("dataset.csv", "w") as dataset:
       pass
    table=getCSVData()
    # List of Kepids
    kepids=getKepids(table)
    
    #for id in kepids:
    #    row = createFeaturesTable(id)
    #    appendToFile(row)