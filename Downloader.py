# Program to download from the MAST the light curves of Kepler stars using LightKurve
# Example for the stars Kepler-74 and Kepler-487
import lightkurve as lk
import numpy as np

targetstarLK = ["Kepler-74","Kepler-487"]

for i in np.arange(len(targetstarLK)):
    search_result=lk.search_lightcurve(targetstarLK[i], author='Kepler', cadence="short") 
    ifile=len(search_result)  
    lc_collection = search_result[:ifile].download_all() 
    lc=lc_collection[0].normalize().remove_nans()         # We generate a concatenated light curve using all those in the collection.  
    for curva in lc_collection[1:]:                       # and normalized to 1. We remove the NaNs too
        lc = lc.append(curva.normalize().remove_nans())   
    lcflux=lc.flux.value 
    lctime=lc.time.value 
    np.savetxt(targetstarLK[i],np.c_[lctime,lcflux.unmasked])  # We save them in ascii files
    print("Done! ",targetstarLK[i])