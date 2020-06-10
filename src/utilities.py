import numpy as np
import csv


def concatenate(gl, lo):
    '''
        Concatenates two lists.

        @param gl (np.array(float)): global values
        @param lo (np.array(float)): local values

        @return (np.array(float)): concatencated values of both views

    '''
    gl = gl.tolist()
    lo = lo.tolist()
    for index, a in enumerate(lo):
        for b in a:
            gl[index].append(b)

    return np.array(gl)


def joinLists(data1, data2):
    '''
        Joins two lists.

        @param data1 (np.array(float)): values of the first data array
        @param data2 (np.array(float)): values of the second data array

        @return (np.array(float)): joined values of both views

    '''
    for index, a in enumerate(data2):
        for b in a:
            data1[index].append(b)

    return np.array(data1)


def writeToFile(name, data):
    '''
        Write data to file.

        @param name (String): name of file
        @param data (np.array(float)): data to be written to file

    '''
    with open(name, "w") as fd:  # Write Header
        writer = csv.writer(fd)
        writer.writerow(data)
