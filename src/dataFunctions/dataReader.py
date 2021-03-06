from astropy.io import fits
import numpy as np
import os
from pathlib import Path
import lightkurve as lk
import pickle

import dataInfo as di
import dataCleaner as dc


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
        Get all filenames (their absolute path) into a list.  

        @param kepid (int): ID of the target star
        @param dir (String): database directory

        @return (list[String]): paths for every file contained in the directory provided

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
    Read the HDUList of each file and store the quarter and data to be used (PDCSAP_FLUX and Time).

    @param aggregate (list[String]): filenames (their absolute path)
    @return (array[float]): time values 
    @return (array[float]): PDCSAP_FLUX values

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


def getConcatenatedLightCurve(flux, time):
    '''
        Concatenate all Light Curves associated with the kepid and normalize values.

        @param flux (array[float]): flux values of the light curve
        @param time (array[float]): time values of the light curve

        @return (array[float]): normalized floax values
        @return (Array[float]): time values

    '''
    for i in flux:  # Same scale for all segments
        i /= np.median(i)
    out_flux = np.concatenate(flux)
    out_time = np.concatenate(time)
    normalized_flux = dc.movingAverage(out_flux, 15)
    lc = lk.LightCurve(out_time, normalized_flux)
    ret = lc.normalize()
    return ret.flux, out_time


def createFluxDatabase(table, kepids, DATA_DIRECTORY):
    '''
        Create a pickle file that saves all flux values associated with each kepid (concatenated light curves).

        @param table (array[array[float]]): TCE table 
        @param kepids (array[int]): Array of all kepid values of the TCE table
        @param DATA_DIRECTORY (String) : String with full path to data directory

    '''
    flux_values = []
    for id in kepids:
        lab = di.labelFinder(table, id)
        if lab != 'UNK':  # Avoid Unknown Light Curves
            # Get Full Path Of All Light Curves
            filenames = filenameWarehouse(id, DATA_DIRECTORY)
            # Read Light Curves And Obtain Time And Flux
            time, flux = fitsConverter(filenames)
            normalized_flux, out_time = getConcatenatedLightCurve(
                flux, time)  # Concatenate All Light Curves
            flux_values.append(normalized_flux)

    pickle.dump(flux_values, open("concateneted_flux.p", "wb"))


def getConcatenatedLightCurve(flux, time):
    '''
        Concatenate all light curves associated with the kepid and normalize values.

        @param flux (array[float]): flux values of the light curve
        @param time (array[float]): time values of the light curve

        @return (array[float]): normalized floax values
        @return (Array[float]): time values

    '''
    for i in flux:  # Same scale for all segments
        i /= np.median(i)
    out_flux = np.concatenate(flux)
    out_time = np.concatenate(time)
    normalized_flux = dc.movingAverage(out_flux, 15)
    lc = lk.LightCurve(out_time, normalized_flux)
    ret = lc.normalize()
    return ret.flux, out_time
