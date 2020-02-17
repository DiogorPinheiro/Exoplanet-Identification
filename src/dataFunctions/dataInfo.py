import numpy as np
import scipy.signal as signal
import pandas as pd

# Calculate Global Mean


def getGlobalMean(data):
    return np.mean(data)
# Calculate Global Median


def getGlobalMedian(data):
    return np.median(data)
# Calculate Global Standard Deviation


def getGlobalSTD(data):
    return np.std(data)
# Calculate Maximum Peak (Minimum Value)


def getMaxPeakPerFile(data):
    return max(data)
# Get Strong Peaks


def getStrongPeaks(data):
    # Use signal invertion for detection
    indexes, prop = signal.find_peaks(-data, -np.mean(data)+(2*np.std(data)), distance=100 )
    #print("std:{} ; 2*std:{} ".format( -np.std(data), -2*np.std(data)))
    #print("2+std:{} ; std:{} ; mean:{}".format( -np.mean(data)+(2*np.std(data)), -np.mean(data)+(np.std(data)), -np.mean(data) ))
    #print(prop['peak_heights'])
    #print(len(indexes))
    inv = []
    for i in prop['peak_heights']:  # Invert Signal to positive values
        inv.append(i)
    return inv, indexes
# Get Medium Peaks


def getMediumPeaks(data):
    # Below 1 STD but above 2 STD
    indexes, prop = signal.find_peaks(
        -data, -np.mean(data)+(np.std(data)), -np.mean(data)+(2*np.std(data)))
    inv = []
    for i in prop['peak_heights']:  # Invert Signal to positive values
        inv.append(i)
    return inv, indexes
# Get Weak Peaks


def getWeakPeaks(data):
    # Below mean and above 1 STD
    indexes, prop = signal.find_peaks(-data,
                                      (-np.mean(data), -np.mean(data)+(np.std(data))))
    inv = []
    for i in prop['peak_heights']:  # Invert Signal to positive values
        inv.append(i)
    return inv, indexes
# Get Points For Each Category


def getNonPeaks(data):
    # Above mean
    indexes, prop = signal.find_peaks(data, np.mean(data))
    return prop['peak_heights'], indexes

def getPeaksWidth(data,indexes):
    a = signal.peak_widths(data, indexes,rel_height=0.90)
    return a[0]

def categorizedPoints(data):
    strongPeaks, indStrong = getStrongPeaks(data)
    mediumPeaks, indMedium = getMediumPeaks(data)
    weakPeaks, indWeak = getWeakPeaks(data)
    nonPeaks, indNon = getNonPeaks(data)
    return strongPeaks, indStrong, mediumPeaks, indMedium, weakPeaks, indWeak, nonPeaks, indNon
# Get kepid Label


def labelFinder(table, kepid):
    # print(findColumnNumber(table,kepid))
    for i, r in table.iterrows():
        if r['kepid'] == kepid:
            return r['av_training_set']
# get CSV data


def dataCSV(filename):
    return pd.read_csv(filename, skiprows=15, usecols=[0, 8])
# Listar Kepids


def listKepids(file):
    print(file.iloc[:,0])
    return file.iloc[:,0]
# Overall Peak Points Data


def overallPeakData(strong, medium, weak):
    aux = []
    aux.append(strong)
    aux.append(medium)
    aux.append(weak)
    aux= np.concatenate(aux)
    return np.mean(aux), np.std(aux)
