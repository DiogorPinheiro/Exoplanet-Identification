import numpy as np
import scipy.signal as signal

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


