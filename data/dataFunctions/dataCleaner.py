import numpy as np
import prox_tv as ptv


def movingAverage(N, window):
    '''
        Calculate the moving average of the provided numpy array list

        input: array and window size used for the moving average 
        output: array with the corresponding moving average 
    '''
    N_padded = np.pad(N, (window//2, window-1-window//2), mode='edge')

    return np.convolve(N_padded, np.ones((window,))/window, mode='valid')


def percentageChange(data, average):
    percentage_change_array = []
    for i, (t, f) in enumerate(zip(data, average)):
        value = (float(f)-t)/abs(t)
        # print("data={};average={};Result={}\n".format(t,f,value))
        percentage_change_array.append(value)
    # print(percentage_change_array)
    return percentage_change_array


def timeVariation(data):
    return ptv.tv1_1d(data, 20)


def normalization(data, mean, std):
    aux = []
    for i in data:
        aux.append((i-mean)//std)
    return aux


def meanDivide(data, mean):
    aux = []
    for i in data:
        aux.append(i//mean)
    return aux


def shiftDown(data):
    aux = []
    for i in data:
        aux.append(i - 1)
    return aux
