import numpy as np
import prox_tv as ptv

def movingAverage(N, window):
    '''
        Calculate The Moving Average Of The Provided Numpy Array List

        Input: Array And Window Size Used For The Moving Average
        Output: Array With The Corresponding Moving Average
    '''
    N_padded = np.pad(N, (window//2, window-1-window//2), mode='edge')

    return np.convolve(N_padded, np.ones((window,))/window, mode='valid')


def percentageChange(data, average):
    '''
        Calculate The Percentage Change Of An Array

        Input: data (Flux Array) ; average - Array Of Values Obtained In The movingAverage Function
        Output: Array With The Percentage Change Values
    '''
    percentage_change_array = []
    for i, (t, f) in enumerate(zip(data, average)):
        value = (float(f)-t)/abs(t)
        # print("data={};average={};Result={}\n".format(t,f,value))
        percentage_change_array.append(value)
    # print(percentage_change_array)
    return percentage_change_array


def timeVariation(data):
    '''
        Signal Low Frequency Oscilation Removal Using Time Variation Method

        Input: Data (Flux Array)
        Output: Data Without Noise
    '''
    return ptv.tv1_1d(data, 20)


def normalization(data, mean, std):
    '''
        Normalize Data Using Mean Removal Method
            Mean Removal: New_Value = (Old_Value - Mean) / Standard_Deviation

        Input: data (Flux Array) ; mean (Int) ; std (float)- Standard Deviation
        Output: Normalized Flux Array
    '''
    aux = []
    for i in data:
        aux.append((i-mean)//std)
    return aux


def meanDivide(data, mean):
    '''
        Divide Values By Their Mean

        Input: data(Flux Array), mean (int)
        Output: Array Of Resulting Values
    '''
    aux = []
    for i in data:
        aux.append(i//mean)
    return aux


def shiftDown(data):
    '''
        Shift All Values By 1

        Input: Flux Array
        Output: Array Of Resulting Values
    '''
    aux = []
    for i in data:
        aux.append(i - 1)
    return aux

