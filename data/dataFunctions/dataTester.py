import matplotlib.pyplot as plt
import numpy as np
import dataAnalyzer as da
import dataReader as dr
import dataCleaner as dc
import dataInfo as di
from scipy.ndimage.filters import gaussian_filter
import scipy.signal as signal
import prox_tv as ptv
from sklearn import preprocessing

# Laptop
#fname_PC = "/Users/diogopinheiro/Documents/Engenharia Informática/3º Ano/2º Semestre/Projeto/testData/0044/004458082/kplr004458082-2009259160929_llc.fits"
#fname_AFP = "/Users/diogopinheiro/Documents/Engenharia Informática/3º Ano/2º Semestre/Projeto/testData/0036/003649426/kplr003649426-2009166043257_llc.fits"
#fname_NTP = "/Users/diogopinheiro/Documents/Engenharia Informática/3º Ano/2º Semestre/Projeto/testData/0037/003735629/kplr003735629-2011073133259_llc.fits"

#Computer
fname_PC = "/home/jcneves/Documents/testData/0114/011442793/kplr011442793-2009166043257_llc.fits"
fname_NTP = "/home/jcneves/Documents/testData/0036/003649426/kplr003649426-2009166043257_llc.fits"
fname_AFP = "/home/jcneves/Documents/testData/0037/003735629/kplr003735629-2011073133259_llc.fits"

# ------------------------------------------- Plot Graphs -----------------------------------------------
dataFiles = [fname_PC, fname_NTP, fname_AFP]
#da.graphSinglePlot(fname_PC)
#da.graphComparisonPlot(dataFiles)

# ------------------------------------------- FITS File Manipulation ------------------------------------
dir = "/home/jcneves/Documents/keplerData"
kepID = 11442793

filenames = dr.filenameWarehouse(kepID,dir)
time, flux = dr.fitsConverter(filenames)
#std=np.std(flux[0])
#mean=np.mean(flux[0])
thresh_values=[]
#print(mean-2*std)
count=0




#da.plotThresholdComparison(kepID,dir)
#
# da.graphFullLightCurve(time,flux)

# ------------------------------------------ PC Plotting ------------------------------------------------

kepids_PC=[11442793,4458082,5602588,5695615,5796675,6264750,1161345,2019477,2439243,2854698,2854914,2857607,2975770,2985587,
       2985767, 2989404, 2019477, 2307415, 2556650, 2693736, 2715135, 2831055, 1870398, 2441495, 2557816, 2696703, 
        2832589, 1871056, 2141783, 2558163]
kepids_NTP=[892667, 1292087, 1574792, 2019227, 2693097, 2722074, 2974858, 2714955, 2988768, 1867630, 2578811, 2578891,
            2854948, 1295289, 2308603, 2282763, 2308957, 2309568, 2696938, 1576558, 2558488, 2579814,2835657, 1430207,
            2297728, 2568971, 2846358, 2860930, 1717528, 2158190]
kepids_AFP=[1162345, 892772, 1026957, 1160891, 1162150, 1162345, 1573174, 1575690, 1575873, 2167600, 2306740, 2307199, 2439211,
            2554853, 2578072, 2578077, 2693092, 2714932, 2714947, 2714954, 2988145, 2855603, 2856756, 2857323, 2857722,
            2860114, 2307199, 2860851, 2969628, 2969638, 2554867, 2693450, 2985398, 1294670, 2167890,2307206, 2440757, 2441151,
            2578404, 2693450, 2694337, 2715053, 2715113, 2715119, 2830919, 1295546, 1296164, 1576115, 1576144, 1869607,
            1870849, 2021440, 2307533, 2441728, 2578901, 2695030, 2715135, 2715228,2715245,2831251]
#da.plotSeveralGraphs(kepids_NTP,dir,15,2)  

# ------------------------------------------ Cleaner -----------------------------------------------------
#print(len(flux))
test=[[1,2,3,4],[2,3,4,5]]
fs=[]
#print(len(flux))
#for f in flux:  # Same scale for all segments  
#        value = dc.moving_average(f,15)
#        fs=dc.percentageChange(f,value)
f1 = ptv.tv1_1d(flux[0],20)

#plt.plot(time[0],flux[0],'k')
#plt.plot(time[0],f1,'-g')
print(min(f1))
std=np.std(f1)
mean=np.mean(f1)
indexes,_=signal.find_peaks(f1,height=(0,mean-(2*std)))
print(indexes)

mean_array=[]
mean_twostd=[]
mean_onestd=[]
for m in range(len(time[0])): 
        mean_array.append(mean)
for m in range(len(time[0])): 
        mean_onestd.append(mean-std)
for m in range(len(time[0])): 
        mean_twostd.append(mean-(2*std))        

plt.plot(time[0],flux[0],'c')     
plt.plot(time[0],f1,'m')   # Light Curve 
plt.plot(time[0],mean_array,'k')    # Mean 
plt.plot(time[0],mean_onestd,'g')   # One Standard Deviation Below The Mean
plt.plot(time[0],mean_twostd,'r')   # Two Standard Deviations Below The Mean
plt.show()

#plt.show()