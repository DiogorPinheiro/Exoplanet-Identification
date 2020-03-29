import numpy as np
import csv


def concatenate(gl, lo):
    gl = gl.tolist()
    lo = lo.tolist()
    for index, a in enumerate(lo):
        for b in a:
            gl[index].append(b)

    return np.array(gl)


def joinLists(data1, data2):
    for index, a in enumerate(data2):
        for b in a:
            data1[index].append(b)

    return np.array(data1)


def writeToFile(data):
    with open("comparison_table.csv", "w") as fd:  # Write Header
        writer = csv.writer(fd)
        writer.writerow(data)
