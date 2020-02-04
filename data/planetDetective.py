import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

def createFeaturesTable(kepid):
    '''
        1. Read and Convert Data
        2. Clean and Normalize Light Curve
        3. Get All Features
            . Data Analyzes
            . Label
        4. Return Row of Features
    '''
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
    
    appendToFile(row)