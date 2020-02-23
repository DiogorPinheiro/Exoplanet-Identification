from astropy.io import fits
import numpy as np
import os
from pathlib import Path
from dataFunctions import dataInfo
from dataFunctions import dataCleaner
import lightkurve as lk

import pickle

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
    # Padding the kepID to a 9 digit number
    kepid = "{:09d}".format(int(kepid))
    # Create directory following the format used
    dir = os.path.join(dir, kepid[0:4], kepid)
    # print(kepid)
    for i in range(18):  # For all quarter prefixes
        for pref in prefix[str(i)]:  # For every value in the dictionaire
            base_name = "kplr{}-{}_llc.fits".format(kepid, pref)  # File format
            # Create absolute path for the file
            filename = os.path.join(dir, base_name)
            # print(filename)
            # Check if file actually exists (sometimes there is a missing quarter)
            if Path(filename).exists():
                aggregate.append(filename)
    # print(aggregate)
    return aggregate


# Function to read data from file
def fitsConverter(aggregate):
    '''
    Read the HDUList of each file and store the quarter and data to be used (PDCSAP_FLUX and Time)

    Input: List of filenames (their absolute path)
    Output: Array of time values and an array of PDCSAP_FLUX values

    '''
    brightness = []  # PDCSAP_FLUX
    time = []
    for f in aggregate:
        with fits.open(f, mode="readonly") as hdulist:
            # hdulist.info()
            time.append(hdulist[1].data['time'])
            brightness.append(hdulist[1].data['PDCSAP_FLUX'])

    for i, (t, bright) in enumerate(zip(time, brightness)):   # Avoid NaN values
        # Check if it is a number of a NaN, return bool
        aux = np.logical_and(np.isfinite(t), np.isfinite(bright))
        # Remove those that have NaN (we are manipulating numpy arrays, so this was the only way found)
        time[i] = t[aux]
        brightness[i] = bright[aux]

    # for f in brightness:
    #    f -= 2*np.median(f)
    #    f *= (-1)
    return time, brightness

def getConcatenatedLightCurve(flux,time):
    '''
        Concatenate all Light Curves associated with the kepid and normalize values

        Input: kepid number
        Output: normalized concatenated light curve
    '''
    for i in flux:  # Same scale for all segments
        i /= np.median(i)
    out_flux = np.concatenate(flux)
    out_time = np.concatenate(time)
    normalized_flux = dataCleaner.movingAverage(out_flux, 15)
    lc = lk.LightCurve(out_time, normalized_flux)
    ret = lc.normalize()
    return ret.flux, out_time

def createFluxDatabase(table,kepids,DATA_DIRECTORY):
    '''
        Create A Pickle File That Saves All Flux Values Associated With Each KepID (Concatenated Light Curves)

        Input: table = TCE Table ;
               kepids : Array Of All KepID Values Of The TCE Table
               DATA_DIRECTORY  : String Will Full Path To Data Directory
    '''
    flux_values = []
    for id in kepids:
        lab = dataInfo.labelFinder(table, id)
        if lab != 'UNK':  # Avoid Unknown Light Curves
            filenames = filenameWarehouse(id, DATA_DIRECTORY)  # Get Full Path Of All Light Curves
            time, flux = fitsConverter(filenames)  # Read Light Curves And Obtain Time And Flux
            normalized_flux, out_time = getConcatenatedLightCurve(flux, time)  # Concatenate All Light Curves
            flux_values.append(normalized_flux)

    pickle.dump(flux_values, open("concateneted_flux.p", "wb"))

def getConcatenatedLightCurve(flux,time):
    '''
        Concatenate all Light Curves associated with the kepid and normalize values

        Input: kepid number
        Output: normalized concatenated light curve
    '''
    for i in flux:  # Same scale for all segments
        i /= np.median(i)
    out_flux = np.concatenate(flux)
    out_time = np.concatenate(time)
    normalized_flux = dataCleaner.movingAverage(out_flux, 15)
    lc = lk.LightCurve(out_time, normalized_flux)
    ret = lc.normalize()
    return ret.flux, out_time