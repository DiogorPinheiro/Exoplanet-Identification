# Development Process

## Know What Data to Use
* Long Cadence Light Curves from Kepler
    * Obtained in FITS (Flexible Image Transport System) format;
    * 18 quarters (Q0 to Q17)

## The Data Set
* Should include all posible labels : PC (planet candidate), AFP (astrophysical false positive), NTP (non-transiting phenomenon), UNK (unknown);
* Important Features: 
* Get the .csv file from https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=q1_q17_dr24_tce 
* Download the corresponding light curves from https://archive.stsci.edu/pub/kepler/lightcurves/tarfiles/Exoplanet_KOI/         ~90GB
* If needed for TensorFlow, generate TFRecord files 

## Training and Test Set
* 80% Training and 20% Test (?)
* Cross-Validation no training set

## Algorithms
* kNN
* Logistic Regression
* SVM
* Neural Networks (?)