from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
import dataReader as dr
import dataCleaner as dc
import prox_tv as ptv
import lightkurve as lk
import scipy.signal as signal
from lightkurve import SFFCorrector

def fileDetail(filename):
    '''
        Provide detailed info on the HDUList contained in the FITS file
    '''
    with fits.open(filename) as hdulist:
        header1 = hdulist[1].header
    print(repr(header1[0:24]))


def graphSinglePlot(filename):
    '''
    Plot light curve of the file provided


    BJD - Julian Days
    BKJD - Kepler Barycentric Julian Day
    '''
    with fits.open(filename, mode="readonly") as hdulist:
        # Read in the "BJDREF" which is the time offset of the time array.
        bjdrefi = hdulist[1].header['BJDREFI']
        bjdreff = hdulist[1].header['BJDREFF']

        # Read in the columns of data.
        times = hdulist[1].data['time']
        pdcsap_fluxes = hdulist[1].data['PDCSAP_FLUX']

    # Convert the time array to full BJD by adding the offset back in.
    bjds = times + bjdrefi + \
        bjdreff    # https://archive.stsci.edu/kepler/manuals/archive_manual.pdf#page=13

    plt.figure(figsize=(9, 4))
    plt.plot(times, pdcsap_fluxes, '-b', label='PDCSAP Flux')
    plt.title('Kepler Light Curve')
    plt.legend()
    plt.xlabel('Time (days)')
    plt.ylabel('Flux (electrons/second)')
    plt.show()


def graphComparisonPlot(filename):
    time = []
    flux = []
    for file in filename:
        with fits.open(file, mode="readonly") as hdulist:
            time.append(hdulist[1].data['time'])
            flux.append(hdulist[1].data['PDCSAP_FLUX'])

    fig, axs = plt.subplots(3)
    axs[0].plot(time[0], flux[0])
    axs[0].set_title('PC (Confirmed Planet)')
    axs[1].plot(time[1], flux[1], 'tab:orange')
    axs[1].set_title('NTP (Non-Transiting Phenomenon)')
    axs[2].plot(time[2], flux[2], 'tab:green')
    axs[2].set_title('AFP (Astrophysical False Positive)')

    for ax in axs.flat:
        ax.set(xlabel='Time (days)', ylabel='Flux ')
        ax.label_outer()
    plt.show()

def graphFullLightCurve(time,flux):
    for i in flux:  # Same scale for all segments  
        i /= np.median(i)

    plt.plot(np.concatenate(time),np.concatenate(flux),'-r')
    plt.yticks([]) 
    plt.show()
    
def showFileNames(filenames):
    print("Number of Files : {}".format(len(filenames)))
    print("\n".join(filenames))
    
def plotSeveralGraphs(kepids,dir,Nrows,Ncolumns):
    fig = plt.figure()
    for i, kep in enumerate(kepids):
        filenames = dr.filenameWarehouse(kep,dir)
        time, flux = dr.fitsConverter(filenames)
        ax = fig.add_subplot(Nrows,Ncolumns,(i+1))
        for f in flux:  # Same scale for all segments  
            f /= np.median(f)
        #print(max(flux,key=lambda x: x[1]))
        ax.plot(np.concatenate(time),np.concatenate(flux))
        #plt.yticks([])  
        #plt.ylim(0.975,1.01)    #PC
        plt.ylim(0.96,1.01)
    #plt.gca().set_aspect('equal',adjustable='box')   
    plt.show()  
    
def plotThresholdComparison(kepid,dir):
    filenames = dr.filenameWarehouse(kepid,dir)
    time, flux = dr.fitsConverter(filenames)
    std=np.std(flux[0])
    mean=np.mean(flux[0])
    
    mean_array=[]
    mean_twostd=[]
    mean_onestd=[]
    for m in range(len(time[0])): 
        mean_array.append(mean)
    for m in range(len(time[0])): 
        mean_onestd.append(mean-std)
    for m in range(len(time[0])): 
        mean_twostd.append(mean-(2*std))        
     
    plt.plot(time[0],flux[0],'b')   # Light Curve      
    plt.plot(time[0],mean_array,'k')    # Mean 
    plt.plot(time[0],mean_onestd,'g')   # One Standard Deviation Below The Mean
    plt.plot(time[0],mean_twostd,'r')   # Two Standard Deviations Below The Mean
    plt.show()
    
    
def graphThresholdExamples(kepids,dir):
    fig, axs = plt.subplots(3)

    for i,files in enumerate(kepids):
        filename = dr.filenameWarehouse(files,dir)
        t,f = dr.fitsConverter(filename)
        f1=dc.movingAverage(f[0],15)
        #f1 = ptv.tv1_1d(f[0],30)
        lc=lk.LightCurve(t[0],f1)
        ret = lc.normalize()
        f1 = ret.flux
        #f1=dc.percentageChange(f1,2)
        std=np.std(f1)
        mean=np.mean(f1)
        mean_array=[]
        mean_twostd=[]
        mean_onestd=[]
        for m in range(len(t[0])): 
            mean_array.append(mean)
        for m in range(len(t[0])): 
            mean_onestd.append(mean-std)
        for m in range(len(t[0])): 
            mean_twostd.append(mean-(2*std))   
        
        axs[i].plot(t[0], f1)
        axs[i].plot(t[0],mean_array,'k')    # Mean 
        axs[i].plot(t[0],mean_onestd,'g')   # One Standard Deviation Below The Mean
        axs[i].plot(t[0],mean_twostd,'r')   # Two Standard Deviations Below The Mean
    

    for ax in axs.flat:
        ax.set(xlabel='Time (days)')
        ax.label_outer()
    plt.show()
