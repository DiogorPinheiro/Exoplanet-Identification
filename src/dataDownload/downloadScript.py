'''
This file creates a script (.sh) to download all files associated with the kepID of each row on the table
of the csv file obtained at https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=q1_q17_dr24_tce

Output: downloader.sh

Additional Note: To execute the download, insert the command ./downloader.sh in the terminal

'''

import csv
import os
import sys

_COMMAND = ("wget -q -nH --cut-dirs=6 -r -l0 -N -np -erobots=off "
            "-R 'index*' -A _llc.fits")
_SOURCE = "http://archive.stsci.edu/pub/kepler/lightcurves"
Directory = "/Volumes/SAMSUNG/test"
CSV_FILE = "/Users/diogopinheiro/Documents/q1_q17_dr24_tce_2020.01.28_08.52.13.csv"
OUTPUT = "downloader.sh"


if __name__ == "__main__":
    # Get Kepler IDs
    kepids = set()
    with open(CSV_FILE) as file:
        reader = csv.DictReader(row for row in file if not row.startswith("#"))
        for row in reader:
            kepids.add(row["kepid"])

    # Create script to download files
    with open(OUTPUT, "w") as f:
        f.write("#!/bin/sh\n")
        for i, kepid in enumerate(kepids):
            kepid = "{0:09d}".format(int(kepid))  # Pad with zeros.
            subdir = "{}/{}".format(kepid[0:4], kepid)  # Get sub-directory
            download_dir = os.path.join(Directory, subdir)
            # Create URL by joining the base command with the sub-directory
            url = "{}/{}/".format(_SOURCE, subdir)
            f.write("{} -P {} {}\n".format(_COMMAND, download_dir, url))

        f.write("End of Download")

    # Make the download script executable.
    os.chmod(OUTPUT, 0o744)
