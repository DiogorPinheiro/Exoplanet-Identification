import numpy as np

def moving_average(N,window):
    '''
        Calculate the moving average of the provided numpy array list
        
        input: list of arrays and window size used for the moving average 
        output: list of arrays with the corresponding moving average 
    '''
    return np.convolve(N, np.ones((window,))/window, mode='valid')
