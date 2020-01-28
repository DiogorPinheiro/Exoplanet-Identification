# Filename containing the CSV file of TCEs in the training set.
TEMP_PATH="/Users/diogopinheiro/Documents/q1_q17_dr24_tce_2020.01.27_02.23.52.csv"
TCE_CSV_FILE="$TEMP_PATH"

# Directory to download Kepler light curves into.
KEPLER_DATA_DIR="/Volumes/SAMSUNG/KeplerData"

# Generate a bash script that downloads the Kepler light curves in the training set.
python downloadScript.py \
  --kepler_csv_file=${TCE_CSV_FILE} \
  --download_dir=${KEPLER_DATA_DIR}

./get_kepler.sh