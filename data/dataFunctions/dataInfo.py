import numpy as np
import scipy.signal as signal
import pandas as pd

# Calculate Global Mean


def getGlobalMean(data):
    mean_data = []
    for f in data:
        mean_data.append(np.mean(f))
    return mean_data
# Calculate Global Median


def getGlobalMedian(data):
    median_data = []
    for f in data:
        median_data.append(np.median(f))
    return median_data
# Calculate Global Standard Deviation


def getGlobalSTD(data):
    std_data = []
    for f in data:
        std_data.append(np.std(f))
    return std_data
# Calculate Maximum Peak (Minimum Value)


def getMaxPeakPerFile(data):
    max_peak = []
    for f in data:
        max_peak.append(min(f))
# Get Strong Peaks


def getStrongPeaks(data):
    # Use signal invertion for detection
    indexes, prop = signal.find_peaks(-data, -(np.mean(data)-(2*np.std(data))))
    inv = []
    for i in prop['peak_heights']:  # Invert Signal to positive values
        inv.append(i)
    return inv
# Get Medium Peaks


def getMediumPeaks(data):
    # Below 1 STD but above 2 STD
    indexes, prop = signal.find_peaks(
        -data, (-(np.mean(data)-(np.std(data))), -(np.mean(data)-(2*np.std(data)))))
    inv = []
    for i in prop['peak_heights']:  # Invert Signal to positive values
        inv.append(i)
    return inv
# Get Weak Peaks


def getWeakPeaks(data):
    # Below mean and above 1 STD
    indexes, prop = signal.find_peaks(-data,
                                      (-np.mean(data), -(np.mean(data)-(np.std(data)))))
    inv = []
    for i in prop['peak_heights']:  # Invert Signal to positive values
        inv.append(i)
    return inv
# Get Points For Each Category


def getNonPeaks(data):
    # Above mean
    indexes, prop = signal.find_peaks(data, np.mean(data))
    return prop['peak_heights']


def categorizedPoints(data):
    strongPeaks = getStrongPeaks(data)
    mediumPeaks = getMediumPeaks(data)
    weakPeaks = getWeakPeaks(data)
    nonPeaks = getNonPeaks(data)
    return strongPeaks, mediumPeaks, weakPeaks, nonPeaks
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


def listKepids(filename):
    table = pd.read_csv(filename, skiprows=15, usecols=[0])
# Overall Peak Points Data


def overallPeakData(strong, medium, weak):
    aux = []
    aux.append(strong)
    aux.append(medium)
    aux.append(weak)
    return np.mean(aux), np.std(aux)
