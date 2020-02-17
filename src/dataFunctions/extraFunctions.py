import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightkurve as lk
import csv

def meanRemoval():
    '''
        Normalize All Features Using Mean Removal
        
        Mean Removal :  x_norm = (x - mean) / std
        
        std: Standard Deviation 
    '''
    dataset = pd.read_csv('dataset.csv', delimiter=',',header=0)
    #print(dataset)
    for i in range(1,len(dataset.columns)-1):
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
    