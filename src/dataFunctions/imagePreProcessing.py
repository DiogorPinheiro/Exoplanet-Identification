import numpy as np
import csv

import dataInfo as di

csvmac = "/Users/diogopinheiro/Documents/Engenharia Informática/3º Ano/2º Semestre/Projeto/Identifying-Exoplanets-Using-ML/src/q1_q17_dr24_tce_2020.01.28_08.52.13.csv"
#CSV_FILE = "/home/jcneves/Documents/Identifying-Exoplanets-Using-ML/src/q1_q17_dr24_tce_2020.01.28_08.52.13.csv"


def getCSVData():
    '''
        Call dataCSV Function To Get The Data Of CSV File

        Output: Pandas Dataframe
    '''
    return di.dataCSV(csvmac)


def getLabel(table, kepid, index):
    '''
        Get The Kepid Label From The TCE Table

        Input: Table (Pandas Dataframe) and Kepid (int)
        Output: 1 if Label Is PC (Confirmed Planet) or 0 if AFP (Astrophysical False Positive) 
                or NTP (Nontransiting Phenomenon)
    '''
    label = di.labelFinder(table, kepid, index)
    if label == 'PC':
        return 1
    elif label == "UNK":
        return 2
    else:
        return 0


def writeFile(file, row):
    with open(file, 'a') as fd:
        writer = csv.writer(fd)
        writer.writerow(row)


def createImageNameLabelDataset():
    table = getCSVData()
    file = open("../data/KepidsWithoutDuplicates.csv", "w")

    kepids = di.listKepids(table)  # List of Kepids (Avoid Duplicates)
    i = 0
    #kepids = kepids[:10]
    for i, k in enumerate(kepids):
        print("{}/{}".format(i, len(kepids)))
        label = getLabel(table, k, i)
        if label != 2:
            aux = [k, label]
            writeFile("../data/KepidsWithoutDuplicates.csv", aux)
        i += 1


def main():
    createImageNameLabelDataset()


main()
