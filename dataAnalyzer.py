from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt

'''
    Graphic Display Of Light Curves 
    
    Output:
        SAP_FLUX - Simple Aperture Photometry Fluctuation
        PDCSAP_FLUX - Pre-Search Data Conditioning SAP Fluctuation
'''
#filename = "./data/mastDownload/Kepler/kplr011446443_sc_Q113313330333033302/kplr011446443-2009131110544_slc.fits"
#filename = "/Users/diogopinheiro/Downloads/public_Q0_long_1/kplr000757218-2009131105131_llc.fits"
filename = "/Users/diogopinheiro/Downloads/005271608/kplr005271608-2010174085026_llc.fits"
#filename = "/Users/diogopinheiro/Downloads/005271608/kplr005271608-2010078095331_llc.fits"

fits.info(filename)

# with fits.open(filename) as hdulist:
#    header1 = hdulist[1].header
# print(repr(header1[0:24]))

# rename file for simplification purposes

with fits.open(filename, mode="readonly") as hdulist:
    # Read in the "BJDREF" which is the time offset of the time array.
    bjdrefi = hdulist[1].header['BJDREFI']
    bjdreff = hdulist[1].header['BJDREFF']

    # Read in the columns of data.
    times = hdulist[1].data['time']
    sap_fluxes = hdulist[1].data['SAP_FLUX']
    pdcsap_fluxes = hdulist[1].data['PDCSAP_FLUX']

# Convert the time array to full BJD by adding the offset back in.
'''
    BJD - Julian Days
    BKJD - Kepler Barycentric Julian Day
'''
bjds = times + bjdrefi + bjdreff


plt.figure(figsize=(9, 4))

# Plot the time, uncorrected and corrected fluxes.
plt.plot(bjds, sap_fluxes, '-k', label='SAP Flux')
plt.plot(bjds, pdcsap_fluxes, '-b', label='PDCSAP Flux')

plt.title('Kepler Light Curve')
plt.legend()
plt.xlabel('Time (days)')
plt.ylabel('Flux (electrons/second)')
plt.show()
