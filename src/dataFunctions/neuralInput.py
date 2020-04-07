import numpy as np
import pickle
from sklearn import preprocessing
import lightkurve as lk
import pandas as pd
import csv
import time as t
import matplotlib.pyplot as plt

# from dataFunctions import dataCleaner
# from dataFunctions import dataInfo
# from dataFunctions import dataAnalyzer
# from dataFunctions import dataReader

import dataInfo as di
import dataCleaner as dc
# import dataReader as dr

CSV_FILE = "/home/jcneves/Documents/Identifying-Exoplanets-Using-ML/src/q1_q17_dr24_tce_2020.01.28_08.52.13.csv"
#csvmac = "/Users/diogopinheiro/Documents/Engenharia Informática/3º Ano/2º Semestre/Projeto/Identifying-Exoplanets-Using-ML/src/q1_q17_dr24_tce_2020.01.28_08.52.13.csv"
DATA_DIRECTORY = "/home/jcneves/Documents/keplerData"
GLOBAL_CSV = "../neural_input_global.csv"
LOCAL_CSV = "../neural_input_local.csv"
GLOBAL_CSV_SOVGOL = "../neural_input_global_sovgol.csv"
LOCAL_CSV_SOVGOL = "../neural_input_local_sovgol.csv"
GLOBAL_CSV_TIMEV = "../neural_input_global_timev.csv"
LOCAL_CSV_TIMEV = "../neural_input_local_timev.csv"
GLOBAL_CSV_MOVAVG = "../neural_input_global_movavg.csv"
LOCAL_CSV_MOVAVG = "../neural_input_local_movavg.csv"


def getCSVData():
    '''
        Call dataCSV Function To Get The Data Of CSV File

        Output: Pandas Dataframe
    '''
    return di.dataCSV(CSV_FILE)


def globaView(curve):
    # Fixed Length 2001 Bins
    lc_global = curve.bin(bins=2001, method='median').normalize() - 1
    lc_global = (lc_global / np.abs(lc_global.flux.min())) * 2.0 + 1
    # print(lc_global.flux.shape)
    # lc_global.scatter()
    return lc_global


def localView(curve, f_dur):
    phase_mask = (curve.phase > -4 * f_dur) & (curve.phase < 4.0 * f_dur)
    lc_zoom = curve[phase_mask]
    if lc_zoom:
        # Fixed Length 201 Bins
        lc_local = lc_zoom.bin(bins=201, method='median').normalize() - 1
        lc_local = (lc_local / np.abs(lc_local.flux.min())) * 2.0 + 1
        # print(lc_local.flux.shape)
        # lc_global.scatter()
        return lc_local
    else:
        return False


def getLabel(table, kepid, index):
    '''
        Get The Kepid Label From The TCE Table

        Input: Table (Pandas Dataframe) and Kepid (int)
        Output: 1 if Label Is PC (Confirmed Planet) or 0 if AFP (Astrophysical False Positive)
                or NTP (Nontransiting Phenomenon)
    '''
    label = di.labelFinder(table, kepid, index)
    if label == 'PC':
        return 1
    elif label == 'UNK':
        return 2
    else:
        return 0


def writeFile(file, row):
    with open(file, 'a') as fd:
        writer = csv.writer(fd)
        writer.writerow(row)


def main():
    start = t.time()
    table = getCSVData().drop_duplicates()
    kepids = di.listKepids(table)  # List of Kepids
    # dataReader.createFluxDatabase(table,kepids,DATA_DIRECTORY)

    file = open("global_movavg_new.csv", "w")
    file2 = open("local_movavg_new.csv", "w")
    file.close()
    file2.close()

    global_flux = []
    local_flux = []
    lc_raw_keeper = []
    #kepids = [1162345, 1162345]
    kepids = kepids[:10]
    count = 0
    for i, kep in enumerate(kepids):
        # filenames = dr.filenameWarehouse(kep, dir_mac)  # Get Full Path Of All Light Curves
        # Read Light Curves And Obtain Time And Flux
        # time, flux = dr.fitsConverter(filenames)
        # out_flux, out_time = dr.getConcatenatedLightCurve(flux, time)  # Concatenate All Light Curves
        # out_flux = dc.sovitzGolay(out_flux)
        count += 1
        name = 'KIC {}'.format(kep)
        print(name)
        print("{}/{}".format(count, len(kepids)))

        lcfs = lk.search_lightcurvefile(
            name, mission='kepler', cadence='long', quarter=[1, 2, 3, 4, 5, 6, 7])
        if lcfs:    # Avoid empty objects
            lcfs = lcfs[:7].download_all()

            try:    # Avoid file with more than 1 target
                # Combine all PDCSAP_FLUX extensions into a single lightcurve object
                lc_raw = lcfs.PDCSAP_FLUX.stitch()
                lc_raw_keeper.append(lc_raw)
                # Clean outliers that are above mean level
                lc_clean = lc_raw.remove_outliers(sigma=20, sigma_upper=4)
            except:
                continue

            # Get Info About Host Star
            period = di.getTCEPeriod(table, kep, i)    # Period
            duration = di.getTCEDuration(table, kep, i)    # Transit Duration
            tO = di.getTCETransitMidpoint(table, kep, i)   # Transit Epoch
            label = getLabel(table, kep, i)

        else:   # If lcfs == 'NoneType'
            continue

        if label != 2:
            # Signal Pre-processing
            # lc = lk.LightCurve(out_time, out_flux)0
            # lc_clean = lc.remove_outliers( sigma=20, sigma_upper=4)    # Clean Outliers
            temp_fold = lc_clean.fold(period, t0=tO)
            fractional_duration = (duration / 24.0) / period
            phase_mask = np.abs(temp_fold.phase) < (fractional_duration * 1.5)
            # Mask The Transit To Avoid Self-subtraction When Flattening The Signal
            transit_mask = np.in1d(
                lc_clean.time, temp_fold.time_original[phase_mask])

            # Flatten The Mask While While Interpolating The Signal Points
            lc_flat, trend_lc = lc_clean.flatten(
                return_trend=True, mask=transit_mask)

            lc_fold = lc_flat.fold(period,  t0=tO)   # Fold The Signal
            global_view = globaView(lc_fold)  # Global View
            gl_phase = global_view.phase
            gl_bri = global_view.flux
            gl_bri = dc.movingAverage(gl_bri, 15)
            np.nan_to_num(gl_bri, nan=np.mean(gl_bri))
            global_info = []
            global_info.append(kep)
            for i in gl_bri:
                global_info.append(i)

            local_view = localView(lc_fold, fractional_duration)  # Local View
            if local_view == False:
                continue
            else:
                lc_phase = local_view.phase
                lc_bri = local_view.flux
                lc_bri = dc.movingAverage(lc_bri, 15)

                np.nan_to_num(lc_bri, nan=np.mean(lc_bri))
                local_info = []
                local_info.append(kep)
                for j in lc_bri:
                    local_info.append(j)

        # Avoid Time Series With Just Nan Values and Force Correspondance Between Views
            if not (np.isnan(lc_bri).any() or np.isnan(gl_bri).any()):
                # fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
                #ax.plot(lc_phase, lc_bri)
                #name = '../Images/Views/Local_' + str(kep) + ".png"
                # fig.savefig(name)   # save the figure to file
                # plt.close(fig)    # close the figure window

                # fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
                #ax.plot(gl_phase, gl_bri)
                #name = '../Images/Views/Global_' + str(kep) + ".png"
                # fig.savefig(name)   # save the figure to file
                # plt.close(fig)    # close the figure window

                local_info.append(label)
                writeFile("local_movavg_new.csv", local_info)
                global_info.append(label)
                writeFile("global_movavg_new.csv", global_info)

    # with open(GLOBAL_CSV, 'w') as fd:
    #    for row in global_flux:
    #        writer = csv.writer(fd)
    #        writer.writerow(row)

    # with open(LOCAL_CSV, 'w') as fd:
    #    for row in local_flux:
    #        writer = csv.writer(fd)
    #        writer.writerow(row)

    # pickle.dump(lc_raw_keeper, open("lc_raw.p", "wb"))

    end = t.time()
    print(end - start)


main()
