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


def getLabel(table, kepid):
    '''
        Get The Kepid Label From The TCE Table

        Input: Table (Pandas Dataframe) and Kepid (int)
        Output: 1 if Label Is PC (Confirmed Planet) or 0 if AFP (Astrophysical False Positive) 
                or NTP (Nontransiting Phenomenon)
    '''
    label = di.labelFinder(table, kepid)
    if label == 'PC':
        return 1
    else:
        return 0


def writeFile(file, row):
    with open(file, 'a') as fd:
        writer = csv.writer(fd)
        writer.writerow(row)


def createImageNameLabelDataset():
    table = getCSVData().drop_duplicates()

    file = open("withoutDuplicateKepids.csv", "w")

    kepids = (di.listKepids(table)).drop_duplicates().reset_index(
        drop=True)  # List of Kepids (Avoid Duplicates)
    i = 0
    for k in kepids:
        print("{}/{}".format(i, len(kepids)))
        label = getLabel(table, k)

        aux = [k, label]
        writeFile("withoutDuplicateKepids.csv", aux)


def main():
    createImageNameLabelDataset()


main()
