import numpy as np
import pandas as pd
import lightkurve as lk
import csv
from sklearn import preprocessing

#from dataFunctions import dataInfo
#from dataFunctions import dataReader
import dataInfo as di
#import dataReader as dr

CSV_FILE = "../data/q1_q17_dr24_tce_2020.01.28_08.52.13.csv"


def getCSVData():
    '''
        Call dataCSV function to get the data of CSV file.

        @return (Pandas Dataframe) : Dataframe with data read from the CSV

    '''
    return di.dataCSV(CSV_FILE)


def getKepids(table):
    '''
        Get all kepids in the CSV file.

        @return (list[int]): List of kepids numbers 

    '''
    return di.listKepids(table)


def getLabel(table, kepid):
    '''
        Get the kepid label from the TCE table.

        @param table (Pandas Dataframe): content from csv table
        @param kepid (int): ID of target star

        @return (boolean):  1 if Label Is PC (Confirmed Planet) 
                            0 if AFP (Astrophysical False Positive) or NTP (Nontransiting Phenomenon)

    '''
    label = di.labelFinder(table, kepid)
    if label == 'PC':
        return 1
    else:
        return 0


def labelCatcher(table, kepid):
    '''
        Get label associated with the kepid from the TCE table.

        @param table (Pandas Dataframe): content from csv table
        @param kepid (int): ID of target star

       @return (String): label/class of the kepid 

    '''
    return di.labelFinder(table, kepid)


# Label added in the second version
def createFeaturesTable(table, kepid, normalized_flux, label):
    '''
        1. Read and convert data
        2. Clean and normalize light curves
        3. Get all features
            . Analyze data
            . Label
        4. Return row of features.

        @param table (Pandas Dataframe): contents of TCE table CSV Table
        @param kepid (int): kepid number
        @param normalized_flux (np.array[float]): Normalized flux associated to the kepid

        @return (np.array[float]): Light curve's features 
    '''
    row = []

    global_mean = di.getGlobalMean(normalized_flux)
    global_median = di.getGlobalMedian(normalized_flux)
    globa_std = di.getGlobalSTD(normalized_flux)

    strongPeaks, indStrong, mediumPeaks, indMedium, weakPeaks, indWeak, nonPeaks, indNon = di.categorizedPoints(
        normalized_flux)

    num_strongPeaks = len(strongPeaks)
    if len(strongPeaks):
        max_strongPeaks = max(strongPeaks)
        mean_strongPeaks = np.mean(strongPeaks)
        std_strongPeaks = np.std(strongPeaks)
        width_strongPeaks = di.getPeaksWidth(normalized_flux, indStrong)
        mean_StrongWidth = np.mean(width_strongPeaks)
        std_StrongWidth = np.std(width_strongPeaks)
        max_strongPeaksPercentage = (
            (max_strongPeaks-global_mean)/abs(global_mean))*100  # Percentage Change

    num_mediumPeaks = len(mediumPeaks)
    if len(mediumPeaks):
        max_mediumPeaks = max(mediumPeaks)
        mean_mediumPeaks = np.mean(mediumPeaks)
        std_mediumPeaks = np.std(mediumPeaks)
        width_mediumPeaks = di.getPeaksWidth(normalized_flux, indMedium)
        mean_MediumWidth = np.mean(width_mediumPeaks)
        std_MediumWidth = np.std(width_mediumPeaks)

    num_weakPeaks = len(weakPeaks)
    if len(weakPeaks):
        max_weakPeaks = max(weakPeaks)
        mean_weakPeaks = np.mean(weakPeaks)
        std_weakPeaks = np.std(weakPeaks)
        width_weakPeaks = di.getPeaksWidth(normalized_flux, indWeak)
        mean_WeakWidth = np.mean(width_weakPeaks)
        std_WeakWidth = np.std(width_weakPeaks)

    if len(nonPeaks):
        num_nonPeaks = len(nonPeaks)
        mean_nonPeaks = np.mean(nonPeaks)
        std_nonPeaks = np.std(nonPeaks)
        width_nonPeaks = di.getPeaksWidth(normalized_flux, indNon)
        mean_NonPeaksW = np.mean(width_nonPeaks)
        std_NonPeaksW = np.std(width_nonPeaks)

    overall_mean, overall_std = di.overallPeakData(
        strongPeaks, mediumPeaks, weakPeaks)

    # label = getLabel(table, kepid) # Used in the first version

    row.append(kepid)
    row.append(num_strongPeaks)
    row.append(num_mediumPeaks)
    row.append(num_weakPeaks)
    row.append(global_mean)
    row.append(global_median)
    row.append(globa_std)
    # row.append(continuous_strongPeaks)
    # row.append(continuous_mediumPeaks)
    if len(strongPeaks):
        row.append(mean_StrongWidth)
    else:
        mean_StrongWidth = 0
        row.append(mean_StrongWidth)
    if len(mediumPeaks):
        row.append(mean_MediumWidth)
    else:
        mean_MediumWidth = 0
        row.append(mean_MediumWidth)
    if len(strongPeaks):
        row.append(std_StrongWidth)
    else:
        std_StrongWidth = 0
        row.append(std_StrongWidth)
    if len(mediumPeaks):
        row.append(std_MediumWidth)
    else:
        std_MediumWidth = 0
        row.append(std_MediumWidth)
    if len(strongPeaks):
        row.append(max_strongPeaks)
        row.append(max_strongPeaksPercentage)
    else:
        max_strongPeaks = -1
        row.append(max_strongPeaks)
        max_strongPeaksPercentage = -1
        row.append(max_strongPeaksPercentage)
    row.append(overall_mean)
    row.append(overall_std)
    if len(nonPeaks):
        row.append(mean_NonPeaksW)
        row.append(std_NonPeaksW)
    else:
        mean_NonPeaksW = (-3)
        std_NonPeaksW = (-3)
        row.append(mean_NonPeaksW)
        row.append(std_NonPeaksW)

    # Pairwise Product Of All Values
    values = [num_strongPeaks, num_mediumPeaks, num_weakPeaks, global_mean, global_median, globa_std, mean_StrongWidth, mean_MediumWidth,
              std_StrongWidth, std_MediumWidth, max_strongPeaks, max_strongPeaksPercentage, overall_mean, overall_std, mean_NonPeaksW, std_NonPeaksW]

    for i in range(0, len(values)-1):
        for j in range(i, len(values)):
            if i != j:
                res = values[i]*values[j]
                row.append(res)
            else:
                continue

    row.append(label)
    #print("Length:{}; values:{}".format(len(row),len(values)))

    return row


def appendToFile(row):
    '''
        Append row of kepid associated values to the table.

        @param row (np.array[float]): Row of features extracted from a light curve

    '''
    with open('dataset_teste3.csv', 'a') as fd:
        writer = csv.writer(fd)
        writer.writerow(row)


def normalizeTable():
    '''
        Read CSV file and normalize each column (except kepid and label).

        Replaces values in CSV file with their normalized version.

    '''

    df = pd.read_csv("dataset_teste2.csv", delimiter=',',
                     header=0, index_col=False)

    i = 1
    for col in df.columns[1:(len(df.columns)-1)]:
        x_array = np.array(df[col])
        # print("Array:{}".format(x_array))
        val = preprocessing.normalize([x_array])
        # print(val)
        for j in range(0, len(df)):
            # print("({},{});value:{}".format(i,j,val[0,j]))
            df.iloc[j, i] = val[0, j]
        i += 1

    df.to_csv('dataset_teste3.csv', index=False)


if __name__ == "__main__":
    '''
        Steps:
        1. Write Column Names To The CSV File
        2. Get All Kepids Numbers
        3. For Each Kepid : Read Light Curve, Concatenate All Light Curves Associated With Kepid, Create Row Of Features, Append Row To CSV File
        4. Normalize Each Column Of The CSV File
    '''
    # Create Column Names
    fields = ['kepid', 'NumStrongPeaks', 'NumMediumPeaks', 'NumWeakPeaks', 'GlobalMean', 'GlobalMedian', 'GlobalSTD', 'MeanStrongPeakWidth', 'STDStrongPeakWidth', 'MeanMediumPeakWidth', 'STDMediumPeakWidth',
              'MaxMagnitude', 'MaxMagnitudePercentage', 'MeanOverallPoints', 'STDOverallPoints', 'NonPeakWidthMean', 'NonPeakWidthSTD']
    for i in range(120):    # Create Headers For Pairwise Values
        fields.append('Pairwise {}'.format(i+1))
    fields.append('label')
    with open("dataset_teste3.csv", "w") as fd:  # Write Header
        writer = csv.writer(fd)
        writer.writerow(fields)

    #table = getCSVData().drop_duplicates()

    # kepids = getKepids(table)  # List of Kepids
    # print(kepids)

    # for id in kepids:
    #    lab = labelCatcher(table, id)
    #    if lab != 'UNK':    # Avoid Unknown Light Curves
    #        filenames = dr.filenameWarehouse(id, DATA_DIRECTORY)    # Get Full Path Of All Light Curves
    #        time, flux = dr.fitsConverter(filenames)        # Read Light Curves And Obtain Time And Flux
    #        normalized_flux,out_time = dr.getConcatenatedLightCurve(flux,time) # Concatenate All Light Curves
    #        row = createFeaturesTable(table,id,normalized_flux) # Create Row Of Features
    #        appendToFile(row)   # Append Row To CSV File

    data_local = np.loadtxt('../data/local_movavg.csv', delimiter=',')
    ke = data_local[:, 0]
    labels = data_local[:, -1]
    data_local = data_local[:, 1:-1]

    for index, id in enumerate(ke):
        row = createFeaturesTable([], id, data_local[index], labels[index])
        appendToFile(row)

    # normalizeTable()    # Normalize All Columns In The CSV File
