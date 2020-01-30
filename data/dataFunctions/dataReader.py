from astropy.io import fits
import numpy as np
import os
from pathlib import Path

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
    Get all filenames (their absolute path) into a list  

    input : kepid(ID of the target star), dir (directory of the file)
    output : list of paths for every file contained in the directory provided

    '''
    prefix = LONG_QUARTER_PREFIXES
    aggregate = []
    kepid = "{:09d}".format(int(kepid)) # Padding the kepID to a 9 digit number
    dir=os.path.join(dir,kepid[0:4],kepid)  # Create directory following the format used
    
    for i in range(18):  # For all quarter prefixes
        for pref in prefix[str(i)]: # For every value in the dictionaire
            base_name = "kplr{}-{}_llc.fits".format(kepid, pref)  # File format
            filename = os.path.join(dir, base_name) # Create absolute path for the file
            print(filename)
            # Check if file actually exists (sometimes there is a missing quarter)
            if Path(filename).exists():
                aggregate.append(filename)
   
    return aggregate


# Function to read data from file
def fitsConverter(aggregate):
    '''
    Read the HDUList of each file and store the quarter and data to be used (PDCSAP_FLUX and Time)

    input: List of filenames (their absolute path)
    output: Array of time values and an array of PDCSAP_FLUX values

    '''
    brightness = []  # PDCSAP_FLUX
    time = []
    quarter = []
    for f in aggregate:
        with fits.open(f, mode="readonly") as hdulist:
            # hdulist.info()
            quarter.append(hdulist['Primary'].header['Quarter'])
            time.append(hdulist[1].data['time'])
            brightness.append(hdulist[1].data['PDCSAP_FLUX'])

    for i, (t,bright) in enumerate(zip(time,brightness)):   # Avoid NaN values 
        aux = np.logical_and(np.isfinite(t),np.isfinite(bright))    # Check if it is a number of a NaN, return bool
        time[i]=t[aux]  # Remove those that have NaN (we are manipulating numpy arrays, so this was the only way found)
        brightness[i]=bright[aux]
  
    return time, brightness
