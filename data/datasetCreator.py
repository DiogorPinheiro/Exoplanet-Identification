import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightkurve as lk
import csv
from sklearn import preprocessing

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


def getLabel(table, kepid):
    return dataInfo.labelFinder(table, kepid)


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

def createFeaturesTable(table, kepid, normalized_flux):
    '''
        1. Read and Convert Data
        2. Clean and Normalize Light Curves
        3. Get All Features
            . Analyze Data
            . Label
        4. Return Row of Features
        
        Input: CSV Table, Kepid Number, Normalized Flux Associated To The Kepid
        Output: Row Of Features 
    '''
    row = []
    
    global_mean = dataInfo.getGlobalMean(normalized_flux)
    global_median = dataInfo.getGlobalMedian(normalized_flux)
    globa_std = dataInfo.getGlobalSTD(normalized_flux)

    strongPeaks, indStrong , mediumPeaks, indMedium, weakPeaks, indWeak, nonPeaks, indNon = dataInfo.categorizedPoints(
        normalized_flux)

    num_strongPeaks = len(strongPeaks)
    if len(strongPeaks):
        max_strongPeaks = max(strongPeaks)
        mean_strongPeaks = np.mean(strongPeaks)
        std_strongPeaks = np.std(strongPeaks)
        width_strongPeaks = dataInfo.getPeaksWidth(normalized_flux,indStrong)
        mean_StrongWidth = np.mean(width_strongPeaks)
        std_StrongWidth = np.std(width_strongPeaks)

    num_mediumPeaks = len(mediumPeaks)
    if len(mediumPeaks):
        max_mediumPeaks = max(mediumPeaks)
        mean_mediumPeaks = np.mean(mediumPeaks)
        std_mediumPeaks = np.std(mediumPeaks)
        width_mediumPeaks = dataInfo.getPeaksWidth(normalized_flux,indMedium)
        mean_MediumWidth = np.mean(width_mediumPeaks)
        std_MediumWidth = np.std(width_mediumPeaks)

    num_weakPeaks = len(weakPeaks)
    if len(weakPeaks):
        max_weakPeaks = max(weakPeaks)
        mean_weakPeaks = np.mean(weakPeaks)
        std_weakPeaks = np.std(weakPeaks)
        width_weakPeaks = dataInfo.getPeaksWidth(normalized_flux,indWeak)
        mean_WeakWidth = np.mean(width_weakPeaks)
        std_WeakWidth = np.std(width_weakPeaks)

    if len(nonPeaks):
        num_nonPeaks = len(nonPeaks)
        mean_nonPeaks = np.mean(nonPeaks)
        std_nonPeaks = np.std(nonPeaks)
        width_nonPeaks = dataInfo.getPeaksWidth(normalized_flux,indNon)
        mean_NonPeaks = np.mean(width_nonPeaks)
        std_NonPeaks = np.std(width_nonPeaks)

    overall_mean, overall_std = dataInfo.overallPeakData(
        strongPeaks, mediumPeaks, weakPeaks)
    
    label = getLabel(table, kepid)
    
    row.append(kepid)
    row.append(num_strongPeaks)
    row.append(num_mediumPeaks)
    row.append(num_weakPeaks)
    row.append(global_mean)
    row.append(global_median)
    row.append(globa_std)
    #row.append(continuous_strongPeaks)
    #row.append(continuous_mediumPeaks)
    row.append(mean_StrongWidth)
    row.append(mean_MediumWidth)
    row.append(std_StrongWidth)
    row.append(std_MediumWidth)
    row.append(max_strongPeaks)
    row.append(overall_mean)
    row.append(overall_std)
    row.append(mean_NonPeaks)
    row.append(std_NonPeaks)
    
    # Pairwise Product Of All Values
    values=[num_strongPeaks,num_mediumPeaks,num_weakPeaks,global_mean,global_median,globa_std,mean_StrongWidth,mean_MediumWidth,
            std_StrongWidth,std_MediumWidth,max_strongPeaks,overall_mean,overall_std,mean_NonPeaks,std_nonPeaks]

    for i in range(0,len(values)-1):
        for j in range (i,len(values)):
            if i != j:
                res = values[i]*values[j]
                row.append(res)
            else:
                continue
    #print("Length:{}; values:{}".format(len(row),len(values)))           
    return row

def appendToFile(row):
    '''
        Append Row Of Kepid Associated Values To The Table
        
        Input: Row Of Values
        Output: None
    '''
    with open('dataset.csv', 'a') as fd:
        writer=csv.writer(fd)
        writer.writerow(row)

def meanRemoval():
    '''
        Normalize All Features Using Mean Removal
        
        Mean Removal :  x_norm = (x - mean) / std
        
        std: Standard Deviation 
    '''
    dataset = pd.read_csv('dataset.csv', delimiter=',',header=0)
    #print(dataset)
    for i in range(1,len(dataset.columns)):
        data = dataset.iloc[:,i]
        mean = np.mean(data)
        std = np.std(data)
        #print("mean={};std={}".format(mean,std))
        for j in range(0,len(dataset)):
            if std != 0:
                value = (dataset.iloc[j,i] - mean) //std
                #print("({},{});value:{}".format(i,j,value))
                dataset.iloc[j,i] = value
    
    dataset.to_csv('dataset.csv')


def main():
    # Create table
    fields=['kepid','NumStrongPeaks','NumMediumPeaks','NumWeakPeaks','GlobalMean','GlobalMedian','GlobalSTD','MeanStrongPeakWidth','STDStrongPeakWidth','MeanMediumPeakWidth','STDMediumPeakWidth',
            'MaxMagnitude','MeanOverallPoints','STDOverallPoints','NonPeakWidthMean','NonPeakWidthSTD']
    for i in range(105):    # Create Headers For Pairwise Values
        fields.append('Pairwise {}'.format(i+1))
    with open("dataset.csv", "w") as fd:
        writer=csv.writer(fd)
        writer.writerow(fields)
    table = getCSVData()
    # List of Kepids
    kepids = getKepids(table)
    #kepids=[11442793,4458082,5602588]

    for id in kepids:
        filenames = dataReader.filenameWarehouse(id, DATA_DIRECTORY)
        time, flux = dataReader.fitsConverter(filenames)
        normalized_flux,out_time = getConcatenatedLightCurve(flux,time)
        row = createFeaturesTable(table,id,normalized_flux)
        #print(row)
        appendToFile(row) 
      
    meanRemoval()
    # Remove Duplicate
main()