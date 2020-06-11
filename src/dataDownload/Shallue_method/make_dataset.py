# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path
import multiprocessing
import numpy as np
import pandas as pd
import preprocess

OUTPUT_DIR = "../../data/Shallue/"
TCE_TABLE_DIR = "./../data/q1_q17_dr24_tce_2020.01.28_08.52.13.csv"
KEPLER_DATA_DIR = "/Users/diogopinheiro/Documents/Engenharia Informática/3º Ano/2º Semestre/Projeto/data"

# Code adapted from https://github.com/dinismf/exoplanet_classification_thesis


def generate_tce_data(tce_table):

    # Initialise dataframes to populate with processed data
    flattened_fluxes_df = pd.DataFrame()
    folded_fluxes_df = pd.DataFrame()
    globalbinned_fluxes_df = pd.DataFrame()
    localbinned_fluxes_df = pd.DataFrame()

    # Processing metrics
    num_tces = len(tce_table)
    processed_count = 0
    failed_count = 0

    # Iterate over every TCE in the table
    for _, tce in tce_table.iterrows():

        try:
            # Process the TCE and retrieve the processed data.
            flattened_flux, folded_flux, global_view, local_view = process_tce(
                tce)

            # Append processed flux light curves for each TCE to output dataframes.
            flattened_fluxes_df = flattened_fluxes_df.append(
                pd.Series(flattened_flux), ignore_index=True)
            folded_fluxes_df = folded_fluxes_df.append(
                pd.Series(folded_flux), ignore_index=True)
            globalbinned_fluxes_df = globalbinned_fluxes_df.append(
                pd.Series(global_view), ignore_index=True)
            localbinned_fluxes_df = localbinned_fluxes_df.append(
                pd.Series(local_view), ignore_index=True)

            print('Kepler ID: {} processed'.format(tce.kepid))
            print("Processed Percentage: ",
                  ((processed_count + failed_count) / num_tces) * 100, "%")
            processed_count += 1

        except:
            print('Kepler ID: {} failed'.format(tce.kepid))
            failed_count += 1

    return flattened_fluxes_df, folded_fluxes_df, globalbinned_fluxes_df, localbinned_fluxes_df


def process_tce(tce):
    """Processes the light curve for a Kepler TCE and returns processed data

    Args:
      tce: Row of the input TCE table.

    Returns:
      Processed TCE data at each stage (flattening, folding, binning).

    Raises:
      IOError: If the light curve files for this Kepler ID cannot be found.
    """
    # Read and process the light curve.
    time, flattened_flux = preprocess.read_and_process_light_curve(
        tce.kepid, KEPLER_DATA_DIR)

    time, folded_flux = preprocess.phase_fold_and_sort_light_curve(
        time, flattened_flux, tce.tce_period, tce.tce_time0bk)

    # Generate the local and global views.
    local_view = preprocess.local_view(time, folded_flux, tce.tce_period,
                                       tce.tce_duration, num_bins=201, bin_width_factor=0.16, num_durations=4)
    global_view = preprocess.global_view(
        time, folded_flux, tce.tce_period, num_bins=2001, bin_width_factor=1 / 2001)

    return local_view, global_view


if __name__ == '__main__':
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    # Load TCE Table
    tce_table = pd.read_csv(TCE_TABLE_DIR, comment='#')

    tce_table["tce_duration"] /= 24  # Convert hours to days.

    # Name of the target column and labels to use as training labels.
    _LABEL_COLUMN = "av_training_set"
    _ALLOWED_LABELS = {"PC", "AFP", "NTP"}

    # Discard other labels from TCE table other than the allowed labels.
    allowed_tces = tce_table[_LABEL_COLUMN].apply(
        lambda l: l in _ALLOWED_LABELS)
    tce_table = tce_table[allowed_tces]
    num_tces = len(tce_table)

    # Randomly shuffle the TCE table.
    np.random.seed(123)
    tce_table = tce_table.iloc[np.random.permutation(num_tces)]

    # Process the TCE tables
    localBin, globalBin = generate_tce_data(tce_table)

    # Store as .csv and serialized
    globalBin.to_csv(
        OUTPUT_DIR + 'global.csv', na_rep='nan', index=False)
    localBin.to_csv(
        OUTPUT_DIR + 'local.csv', na_rep='nan', index=False)
