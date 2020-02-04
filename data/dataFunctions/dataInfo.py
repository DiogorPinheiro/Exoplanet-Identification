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
    max_peak=[]
    for f in data:
        max_peak.append(min(f))
# Get Strong Peaks
def getStrongPeaks(data):
    indexes,prop=signal.find_peaks(-data,-(np.mean(data)-(2*np.std(data)))) # Use signal invertion for detection
    inv=[]
    for i in prop['peak_heights']:  # Invert Signal to positive values
        inv.append(i*(-1))
    return inv
# Get Medium Peaks      - TO CHANGE
def getMediumPeaks(data):
    # Below 1 STD but above 2 STD
    indexes,peaks=signal.find_peaks(data,height=(np.mean(data-(2*np.std(data))),np.mean(data-np.std(data))))
    return peaks['peak_heights']
# Get Weak Peaks    - TO CHANGE
def getWeakPeaks(data):
    # Below mean and above 1 STD
    indexes,peaks=signal.find_peaks(data,height=(np.mean(data-(np.std(data))), np.mean(data)))
    return peaks['peak_heights']
# Get Points For Each Category
def categorizedPoints(data):
    strongPeaks=getStrongPeaks(data)
    mediumPeaks=getMediumPeaks(data)
    weakPeaks=getWeakPeaks(data)
    return strongPeaks,mediumPeaks,weakPeaks
# Get kepid Label
def labelFinder(table,kepid):
    #print(findColumnNumber(table,kepid))
    for i, r in table.iterrows():
        if r['kepid'] == kepid:
            return r['av_training_set']
#get CSV data
def dataCSV(filename):
    return pd.read_csv(filename, skiprows=15 , usecols = [0,8])

        
def listKepids(filename):
    table = pd.read_csv(filename, skiprows=15 , usecols = [0])