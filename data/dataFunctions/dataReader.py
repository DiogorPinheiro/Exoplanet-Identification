from astropy.io import fits
import numpy as np
import os

LONG_QUARTER_PREFIXES = {'0': ['2009131105131'],
                         '1': ['2009166043257'],
                         '2': ['2009259160929'],
                         '3': ['2009350155506'],
                         '4': ['2010078095331', '2010009091648'],
                         '5': ['2010174085026'],
                         '6': ['2010265121752'],
                         '7': ['2010355172524'],
                         '8': ['2011073133259'],
                         '9': ['2011177032512'],
                         '10': ['2011271113734'],
                         '11': ['2012004120508'],
                         '12': ['2012088054726'],
                         '13': ['2012179063303'],
                         '14': ['2012277125453'],
                         '15': ['2013011073258'],
                         '16': ['2013098041711'],
                         '17': ['2013131215648']}


def filenameWarehouse(kepid, dir):
    '''
    Get all filenames (their absolute path) into a list in order to be read 

    input : kepid(ID of the target star), dir (directory of the file)
    output : list of filenames

    '''
    prefix = "LONG_QUARTER_PREFIXES"
    aggregate = []
    for i in range(18):  # For each one of quarter prefixes
        for pref in prefix[i]:
            base_name = "kplr{}-{}_lls.fits".format(kepid, pref)  # File format
            filename = os.path.join(dir, base_name)
            # Check if file actually exists (sometimes there is a missing quarter)
            if os.path.exists(filename):
                aggregate.append(filename)

    return aggregate


# Function to read data from file
def fitsConverter(aggregate):
    '''
    Read the HDUList of each file and store the quarter and data to be used (PDCSAP_FLUX and Time)

    input: List of filenames (their absolute path)
    output: Array of time values, array of PDCSAP_FLUX values and their respective quarters name

    '''
    brightness = []  # PDCSAP_FLUX
    time = []
    quarter = []
    with fits.open(aggregate, mode="readonly") as hdulist:
        # hdulist.info()
        quarter = hdulist['Primary'].header['Quarter']
        time.append(hdulist[1].data['time'])
        quarter.append(hdulist[1].data['PDCSAP_FLUX'])

    return time, brightness
