
import dataAnalyzer as da

fname_PC = "/Users/diogopinheiro/Documents/Engenharia Informática/3º Ano/2º Semestre/Projeto/testData/0044/004458082/kplr004458082-2009259160929_llc.fits"
fname_AFP = "/Users/diogopinheiro/Documents/Engenharia Informática/3º Ano/2º Semestre/Projeto/testData/0036/003649426/kplr003649426-2009166043257_llc.fits"
fname_NTP = "/Users/diogopinheiro/Documents/Engenharia Informática/3º Ano/2º Semestre/Projeto/testData/0037/003735629/kplr003735629-2011073133259_llc.fits"


dataFiles = [fname_PC, fname_NTP, fname_AFP]
# da.graphSinglePlot(fname_AFP)
da.graphComparisonPlot(dataFiles)
