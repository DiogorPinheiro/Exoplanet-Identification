import matplotlib.pyplot as plt
import numpy as np
import dataAnalyzer as da
import dataReader as dr

# Laptop
#fname_PC = "/Users/diogopinheiro/Documents/Engenharia Informática/3º Ano/2º Semestre/Projeto/testData/0044/004458082/kplr004458082-2009259160929_llc.fits"
#fname_AFP = "/Users/diogopinheiro/Documents/Engenharia Informática/3º Ano/2º Semestre/Projeto/testData/0036/003649426/kplr003649426-2009166043257_llc.fits"
#fname_NTP = "/Users/diogopinheiro/Documents/Engenharia Informática/3º Ano/2º Semestre/Projeto/testData/0037/003735629/kplr003735629-2011073133259_llc.fits"

#Computer
fname_PC = "/home/jcneves/Documents/testData/0044/004458082/kplr004458082-2009259160929_llc.fits"
fname_NTP = "/home/jcneves/Documents/testData/0036/003649426/kplr003649426-2009166043257_llc.fits"
fname_AFP = "/home/jcneves/Documents/testData/0037/003735629/kplr003735629-2011073133259_llc.fits"

# ------------------------------------------- Plot Graphs -----------------------------------------------
dataFiles = [fname_PC, fname_NTP, fname_AFP]
#da.graphSinglePlot(fname_AFP)
#da.graphComparisonPlot(dataFiles)

# ------------------------------------------- FITS File Manipulation ------------------------------------
dir = "/home/jcneves/Documents/testData"
kepID = 4458082

filenames = dr.filenameWarehouse(kepID,dir)
#print(len(filenames))
#print("\n".join(filenames))

time, flux = dr.fitsConverter(filenames)
#print(flux)

for f in flux:
    f /= np.median(f)

#print(flux)   
plt.plot(np.concatenate(time),np.concatenate(flux),'-r')
plt.show()
