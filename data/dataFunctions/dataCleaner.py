import numpy as np
import prox_tv as ptv

def moving_average(N,window):
    '''
        Calculate the moving average of the provided numpy array list
        
        input: array and window size used for the moving average 
        output: array with the corresponding moving average 
    '''
    middle=int(window/2)+1 # Assuming that the window will always be an odd number
    res_array=np.convolve(N, np.ones((window,))/window, mode='same')
    return res_array[middle]
    
def percentageChange(data):
    percentage_change_array = []
    for i, t in enumerate(data):
        #print("data={};average={}\n".format(t,f))
        value = (t-moving_average(data,15))/moving_average(data,15)
        #print("value={}\n".format(value))
        #print(value)
        percentage_change_array.append(value*100)
    #print(percentage_change_array)   
    return percentage_change_array

def timeVariation(data):
    return ptv.tv1_1d(data,20)
