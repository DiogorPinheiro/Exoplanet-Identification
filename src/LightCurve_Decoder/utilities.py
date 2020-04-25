import matplotlib.pyplot as plt


def createChunkingPlot(data, chunk_size):
    '''
        Plot light curve 'data' separated in chunks (represented by vertical lines)

        input: data -> Light Curve 
            chunk_size -> Number of points contained in each single chunk (except the last one, which has chunk_size+1 points)
    '''
    x = list(range(len(data)))
    plt.plot(x, data, '.', color='red')
    for i in range(0, len(data), chunk_size):
        plt.axvline(x=i)
    plt.show()
