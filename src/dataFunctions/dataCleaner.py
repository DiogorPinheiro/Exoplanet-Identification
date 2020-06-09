import numpy as np
import prox_tv as ptv
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


def movingAverage(N, window):
    '''
        Calculate the moving average of the provided numpy array list.

        @param N (np.array[float]): Values to be averaged
        @param window (int): Window size to be used to calculate the average 

        @return (array[float]): Array With The Corresponding Moving Average
    '''
    N_padded = np.pad(N, (window//2, window-1-window//2), mode='edge')

    return np.convolve(N_padded, np.ones((window,))/window, mode='valid')


def percentageChange(data, average):
    '''
        Calculate the percentage change of an array.

        @param data (np.array(float)): flux values of the light curve 
        @param average (np.array(float): Array of values obtained in the moving average function

        @return (np.array(float)): Array with the percentage change values
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
        Signal low frequency oscilation removal using time variation method.

        @param data (np.array(float): flux values of a light curve
        @return (np.array(float)): flux values without noise
    '''
    return ptv.tv1_1d(data, 20)


def normalization(data, mean, std):
    '''
        Normalize Data Using Mean Removal Method using ->    Mean Removal: New_Value = (Old_Value - Mean) / Standard_Deviation.

        @param data (np.array(float): flux values of a light curve
        @param mean (float): mean value of data 
        @param std (float): standard Deviation

        @return (np.array(float)): Normalized Flux Array
    '''
    aux = []
    for i in data:
        aux.append((i-mean)//std)
    return aux


def meanDivide(data, mean):
    '''
        Divide values by their mean.

        @param data (np.array[float]): flux values of a light curve
        @param mean (int): mean value of data

        @return (array[float]): array of resulting values
    '''
    aux = []
    for i in data:
        aux.append(i//mean)
    return aux


def shiftDown(data):
    '''
        Shift all values by 1.

        @param data (np.array[float]): flux values of a light curve
        @return (array[float]): array of resulting values
    '''
    aux = []
    for i in data:
        aux.append(i - 1)
    return aux


def sovitzGolay(data):
    '''
        Apply Sovitsky-Golay filter to data

        @param data (np.array[float]): flux values of a light curve
        @return (array[float]): array of resulting values
    '''
    return savgol_filter(data, 51, 3)
