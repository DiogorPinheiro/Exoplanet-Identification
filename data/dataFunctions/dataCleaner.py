import numpy as np
import prox_tv as ptv

def moving_average(N,window):
    '''
        Calculate the moving average of the provided numpy array list
        
        input: array and window size used for the moving average 
        output: array with the corresponding moving average 
    '''
    values=[]
    for f in N:
        #print(f)
        #print(np.convolve(f, np.ones((window,))/window, mode='same'))
        values.append(np.convolve(f, np.ones((window))/window, mode='same'))
    #print(values)
    return values

def percentageChange(data,average):
    percentage_change_array = []
    for i, (t,f) in enumerate(zip(data,average)):
        value = (f-t)/t
        #print(value)
        percentage_change_array.append(value*100)
    #print(percentage_change_array)   
    return percentage_change_array

def timeVariation(data):
    return ptv.tv1_1d(data,20)
