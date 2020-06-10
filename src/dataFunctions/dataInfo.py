import numpy as np
import scipy.signal as signal
import pandas as pd


def getGlobalMean(data):
    '''
        Get global mean of data.

        @param data (np.array[float]): data values of the light curve

        @return (float): mean value

    '''
    return np.mean(data)


def getGlobalMedian(data):
    '''
        Get global median of data.

        @param data (np.array[float]): data values of the light curve

        @return (float): median value

    '''
    return np.median(data)


def getGlobalSTD(data):
    '''
        Get global standard deviation of data.

        @param data (np.array[float]): data values of the light curve

        @return (float): standard deviation value

    '''
    return np.std(data)


def getMaxPeakPerFile(data):
    '''
        Find maximum value of data.

        @param data (np.array[float]): data values of the light curve

        @return (float): maximum value found

    '''
    return max(data)


def getStrongPeaks(data):
    '''
        Find strong peaks in data, that is values smaller than two standard deviations below the mean.

        @param data (np.array[float]): data values of the light curve

        @return inv(list[float]): strong peak values
        @return indexes(list[float]): strong peak indexes

    '''
    # Use signal invertion for detection
    indexes, prop = signal.find_peaks(-data, -
                                      np.mean(data)+(2*np.std(data)), distance=100)
    # print("std:{} ; 2*std:{} ".format( -np.std(data), -2*np.std(data)))
    # print("2+std:{} ; std:{} ; mean:{}".format( -np.mean(data)+(2*np.std(data)), -np.mean(data)+(np.std(data)), -np.mean(data) ))
    # print(prop['peak_heights'])
    # print(len(indexes))
    inv = []
    for i in prop['peak_heights']:  # Invert Signal to positive values
        inv.append(i)
    return inv, indexes


def getMediumPeaks(data):
    '''
        Find median peaks in data, that is values smaller than one standard deviation below the mean.

        @param data (np.array[float]): data values of the light curve

        @return inv(list[float]): medium peak values
        @return indexes(list[int]): medium peak indexes

    '''
    # Below 1 STD but above 2 STD
    indexes, prop = signal.find_peaks(
        -data, -np.mean(data)+(np.std(data)), -np.mean(data)+(2*np.std(data)))
    inv = []
    for i in prop['peak_heights']:  # Invert Signal to positive values
        inv.append(i)
    return inv, indexes
# Get Weak Peaks


def getWeakPeaks(data):
    '''
        Find weak peaks in data, that is values smaller than the mean but above the first standard deviations.

        @param data (np.array[float]): data values of the light curve

        @return inv(list[float]): weak peak values
        @return indexes(list[int]): weak peak indexes

    '''
    # Below mean and above 1 STD
    indexes, prop = signal.find_peaks(-data,
                                      (-np.mean(data), -np.mean(data)+(np.std(data))))
    inv = []
    for i in prop['peak_heights']:  # Invert Signal to positive values
        inv.append(i)
    return inv, indexes
# Get Points For Each Category


def getNonPeaks(data):
    '''
        Find non peaks in data, that peaks above the mean.

        @param data (np.array[float]): data values of the light curve

        @return inv(list[float]): non peak values
        @return indexes(list[int]): non peak indexes

    '''
    indexes, prop = signal.find_peaks(data, np.mean(data))
    return prop['peak_heights'], indexes


def getPeaksWidth(data, indexes):
    '''
        Find the width of the peaks.

        @param data (np.array[float]): data values of the light curve
        @param indexes [list[int]]: index position of the peaks to be analyzed

        @return (float): peak width

    '''
    a = signal.peak_widths(data, indexes, rel_height=0.90)
    return a[0]


def categorizedPoints(data):
    '''
        Get all the peaks for all categories (strong, medium, weak and non).

        @param data (np.array[float]): data values of the light curve

        @return StrongPeaks(list[float]), mediumPeaks(list[float]), weakPeaks(list[float]), nonPeaks(list[float]) : peak values for all 4 cateogories
        @return indStrong(list[int]), indMedium(list[int]), indWeak(list[int]), indNon(list[int]): peak indexes for all 4 categories

    '''
    strongPeaks, indStrong = getStrongPeaks(data)
    mediumPeaks, indMedium = getMediumPeaks(data)
    weakPeaks, indWeak = getWeakPeaks(data)
    nonPeaks, indNon = getNonPeaks(data)
    return strongPeaks, indStrong, mediumPeaks, indMedium, weakPeaks, indWeak, nonPeaks, indNon


def labelFinder(table, kepid, index):
    '''
        Find label of kepid in table.

        @param table (Panda Dataframe): TCE table content
        @param kepid (int): ID of target star
        @param index (int): index of wanted row in table

        @return (int): if the label is found, returns the label

    '''
    # print(findColumnNumber(table,kepid))
    for i, r in table.iterrows():
        if (r['kepid'] == kepid) and (i == index):
            return r['av_training_set']
        else:
            continue


def dataCSV(filename):
    '''
        Read CSV table content.

        @param filename (String): path of file to be read

        @return (Panda Dataframe): content of file

    '''
    return pd.read_csv(filename, skiprows=15, usecols=[0, 2, 4, 6, 8])


def listKepids(file):
    '''
        List all kepids present in the file.

        @param file (Pandas Dataframe): content of TCE table

        @return (Pandas Dataframe): column corresponding to the kepids
    '''

    # print(file.iloc[:, 0])
    return file.iloc[:, 0]


def overallPeakData(strong, medium, weak):
    '''
        Aggregate all peaks data.

        @param strong(list[float]): values of strong peaks found in the light curve
        @param medium(list[float]): values of medium peaks found in the light curve
        @param weak(list[float]): values of weak peaks found in the light curve

        @return (float): mean of all peaks
        @return (float): standard deviation of all peaks

    '''
    aux = []
    aux.append(strong)
    aux.append(medium)
    aux.append(weak)
    aux = np.concatenate(aux)
    return np.mean(aux), np.std(aux)


def getTCEPeriod(table, kepid, index):
    '''
        Get the TCE period for a given row in the TCE table.

        @param table (Panda Dataframe): TCE table content
        @param kepid (int): ID of target star
        @param index (int): index of wanted row in table

        @return (float): returns the TCE period if the row with the given index exists

    '''
    for i, r in table.iterrows():
        if (r['kepid'] == kepid) and (i == index):
            return r['tce_period']


def getTCEDuration(table, kepid, index):
     '''
        Get the TCE duration for a given row in the TCE table.

        @param table (Panda Dataframe): TCE table content
        @param kepid (int): ID of target star
        @param index (int): index of wanted row in table

        @return (float): returns the TCE duration if the row with the given index exists

    '''
    for i, r in table.iterrows():
        if (r['kepid'] == kepid) and (i == index):
            return r['tce_duration']


def getTCETransitMidpoint(table, kepid, index):
     '''
        Get the TCE transit mid-point for a given row in the TCE table.

        @param table (Panda Dataframe): TCE table content
        @param kepid (int): ID of target star
        @param index (int): index of wanted row in table

        @return (float): return the TCE transit mid-point if the row with the given index exists

    '''
    for i, r in table.iterrows():
        if (r['kepid'] == kepid) and (i == index):
            return r['tce_time0bk']
