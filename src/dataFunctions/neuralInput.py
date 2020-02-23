import numpy as np
import pickle
from sklearn import preprocessing
import lightkurve as lk
import pandas as pd
import csv

from dataFunctions import dataCleaner
from dataFunctions import dataInfo
from dataFunctions import dataAnalyzer
from dataFunctions import dataReader

CSV_FILE = "/home/jcneves/Documents/Identifying-Exoplanets-Using-ML/src/q1_q17_dr24_tce_2020.01.28_08.52.13.csv"
DATA_DIRECTORY = "/home/jcneves/Documents/keplerData"
GLOBAL_CSV = "../neural_input_global.csv"
LOCAL_CSV = "../neural_input_local.csv"

def getCSVData():
    '''
        Call dataCSV Function To Get The Data Of CSV File

        Output: Pandas Dataframe
    '''
    return dataInfo.dataCSV(CSV_FILE)

def globaView(curve):
    lc_global = curve.bin(bins=2001, method='median').normalize() - 1   # Fixed Length 2001 Bins
    lc_global = (lc_global / np.abs(lc_global.flux.min())) * 2.0 + 1
    #print(lc_global.flux.shape)
    #lc_global.scatter()
    return lc_global

def localView(curve,f_dur):
    phase_mask = (curve.phase > -4 * f_dur) & (curve.phase < 4.0 * f_dur)
    lc_zoom = curve[phase_mask]
    lc_local = lc_zoom.bin(bins=201, method='median').normalize() - 1   # Fixed Length 201 Bins
    lc_local = (lc_local / np.abs(lc_local.flux.min())) * 2.0 + 1
    #print(lc_local.flux.shape)
    # lc_global.scatter()
    return lc_local


def getLabel(table, kepid):
    '''
        Get The Kepid Label From The TCE Table

        Input: Table (Pandas Dataframe) and Kepid (int)
        Output: 1 if Label Is PC (Confirmed Planet) or 0 if AFP (Astrophysical False Positive)
                or NTP (Nontransiting Phenomenon)
    '''
    label = dataInfo.labelFinder(table, kepid)
    if label == 'PC':
        return 1
    else:
        return 0

def main():
    table = getCSVData().drop_duplicates()
    kepids = (dataInfo.listKepids(table)).drop_duplicates().reset_index(drop=True) # List of Kepids (Avoid Duplicates)
    #dataReader.createFluxDatabase(table,kepids,DATA_DIRECTORY)

    global_flux=[]
    local_flux=[]

    for kep in kepids:
        filenames = dataReader.filenameWarehouse(kep, DATA_DIRECTORY)  # Get Full Path Of All Light Curves
        time, flux = dataReader.fitsConverter(filenames)  # Read Light Curves And Obtain Time And Flux
        out_flux, out_time = dataReader.getConcatenatedLightCurve(flux, time)  # Concatenate All Light Curves

        # Get Info About Host Star
        period = dataInfo.getTCEPeriod(table, kep)
        duration = dataInfo.getTCEDuration(table,kep)
        tO = dataInfo.getTCETransitMidpoint(table,kep)
        label = getLabel(table, kep)

        # Signal Pre-processing
        lc = lk.LightCurve(out_time,out_flux)
        lc_clean = lc.remove_outliers(sigma=20, sigma_upper=4)    # Clean Outliers
        temp_fold = lc_clean.fold(period, t0=tO)
        fractional_duration = (duration / 24.0) / period
        phase_mask = np.abs(temp_fold.phase) < (fractional_duration * 1.5)
        transit_mask = np.in1d(lc_clean.time, temp_fold.time_original[phase_mask])  #  Mask The Transit To Avoid Self-subtraction When Flattening The Signal

        lc_flat, trend_lc = lc_clean.flatten(return_trend=True, mask=transit_mask)  # Flatten The Mask While While Interpolating The Signal Points

        lc_fold = lc_flat.fold(period, t0=tO)   # Fold The Signal

        global_view = globaView(lc_fold)  # Global View
        gl_phase = global_view.phase
        gl_bri = global_view.flux

        global_info = []
        if not(np.isnan(gl_bri).any()): # Avoid Rows With nan values
            global_info.append(kep)
            for i in gl_bri:
                global_info.append(i)
            global_info.append(label)
            global_flux.append(global_info)


        local_view = localView(lc_fold,fractional_duration) # Local View
        lc_phase = local_view.phase
        lc_bri = local_view.flux
        local_info = []
        if not (np.isnan(lc_bri).any()):
            local_info.append(kep)
            for j in lc_bri:
                local_info.append(j)
            local_info.append(label)
            local_flux.append(local_info)

    with open(GLOBAL_CSV, 'w') as fd:
        for row in global_flux:
            writer = csv.writer(fd)
            writer.writerow(row)

    with open(LOCAL_CSV, 'w') as fd:
        for row in local_flux:
            writer = csv.writer(fd)
            writer.writerow(row)

main()