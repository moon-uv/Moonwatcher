#########################################################################
# Moonwatcher program
# Authors: sort.rnd(F. J. Ballesteros, I. Marti-Vidal, A. Fernández-Soto)
# Serial version, suitable for Jupyter notebooks
# 1D version, exploring only in lunar orbital periods
# Example for Kepler-487 b
#########################################################################

# (0) SETTING UP THE ENVIRONMENT

# Importing the usual libraries 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Importing Polynomial and polyfit to fit a polynomial
from numpy.polynomial import Polynomial
from numpy.polynomial.polynomial import polyfit

# Importing shift from scipy, which allows an "np.roll" by less than one unit by interpolating, and lombscargle
from scipy.ndimage import shift
from scipy.optimize import curve_fit
from astropy.timeseries import LombScargle

# Defining some variables and functions:

# To convert from astronomical units to km and vice versa
ua_to_km=149597870.7
km_to_ua=1./149597870.7

# Interval between two consecutive instants in the light curve
tlapse=58.84876/(60.*60.*24.) # we work with short cadence

# Function to compute the Hill radius in au; 
# Mstar in solar masses; Mplanet in Jupiter masses, R_orb orbital radius of the planet in au:
def R_Hill(Mstar,Mplanet,R_orb):
    R_out=R_orb*(Mplanet/(3.*Mstar*1047.348644))**(1./3.)
    return R_out

# Function to compute the mass of the central object from the orbital radius and period of its satellite;
# Output in Jupiter masses; R_or_lu in au; T_or_lu in days:
def M_orb(R_or_lu,T_or_lu):
    M_pla=1047.348644*(R_or_lu**3/(T_or_lu/365.25)**2)
    return M_pla

# Function to compute the period of the moon from its orbital radius and the planet's mass;
# Output in days; R_or_lu in au; M_pla in Jupiter masses:
def T_luna(R_or_lu,M_pla):
    T_or_lu=365.25*((1047.348644*(R_or_lu**3/M_pla))**(0.5))
    return T_or_lu

# Function to compute the orbital radius of the moon from its period and the planet's mass;
# Output in au; T_or_lu in days; M_pla in Jupiter masses:
def R_luna(T_or_lu,M_pla):
    R_or_lu=((((T_or_lu/365.25)**2)/1047.348644)*M_pla)**(1./3.)
    return R_or_lu 

def sine_func(x, amplitude, freq, pha):
    return amplitude * np.sin((2.*np.pi*freq * x) + pha)


# (1) WE DEFINE THE PLANET WE WILL WORK WITH, SET SOME PARAMETERS AND READ THE LIGHT CURVE

planet_name = 'Kepler-487 b'
orden=5  # polynomial order of the continuum fit. Default 5
time_samp = 3. # Time interval in minutes with which we explore the possible lunar periods

df = pd.read_csv("./data/exoplanet_data.tsv", sep="\t",index_col=['kepler name'])

planet_data = df.loc[planet_name,:]['light curve file name']

data = np.genfromtxt("./data/"+planet_data)


# (2) READING THE PLANET'S DATA

lctime=data[:,0]
lcflux=data[:,1]

t0=lctime[0]
tlast=lctime[-1]

print('first instant of the light curve (BKJD)',t0)
print('last instant of the light curve  (BKJD)',tlast)

tmin = df.loc[planet_name,:]['t0 first transit (KBJD)']
periodo = df.loc[planet_name,:]['orbital period (days)']

while tmin < t0:
    tmin=tmin + periodo

tdur = df.loc[planet_name,:]['transit duration (hrs)']/24.
Rp = df.loc[planet_name,:]['planet size (Earth radii)']*0.00004247
R_orb = df.loc[planet_name,:]['orbital semi major axis (au)']
starmass = df.loc[planet_name,:]['stellar mass (Solar mass)']
mass = df.loc[planet_name,:]['Planet_Mass_MM_(M_Jup)']

# The orbital radius of the exomoon will lie between Rp and R_Hill, that is, at the very least it will be 
# the size of the planet (grazing its atmosphere) and at most the radius of the Hill sphere (beyond that it would escape).
# It is easy to show that the period of a moon at the edge of the Hill radius is equal to planet_period/sqrt(3).

# These radial limits translate (thanks to Kepler's 3rd law) into the following limits for the periods:
Tlumin=T_luna(Rp,mass)                             
Tlumax=periodo/np.sqrt(3.)

# We will do the sampling in the time variable because we have seen that the radial one is quite insensitive
# and for a given period, fairly large changes in the search radius nevertheless give similar results

# Number of time and phase samples we take; in this example we sample every time_samp minuts and every 5º in phase
Tluint=int((Tlumax-Tlumin)*(1440.)/time_samp)
Fasint=int(360./5.) 

print('Period (days)=',periodo)
print('Orbital radius (au)=',R_orb,' (km)=',R_orb*ua_to_km)
print('Mass (Mj)=',mass)
print('Planet size (radius, au)',Rp,' (diameter, km)',2.*Rp*ua_to_km)
print('Star mass',starmass)
print('Midpoint (KBJD)=',tmin)
print('Transit duration (days)=',tdur,' (hours)=',tdur*24.)
print('Exploration range:',Tlumin,Tlumax,'days; samplings:',Tluint)


# (3) HERE WE SLICE THE LIGHT CURVE INTO A MATRIX OF INDIVIDUAL TRANSITS THAT WE STORE IN "curvas"

# R_orb is the orbital radius of the exoplanet. From one transit to the next it will travel 2*pi*R_orb
# R_Hill is the size of the Hill sphere for that planet. The Moon can only be inside this region.
# Therefore in each transit we are only interested in a fraction (2*R_Hill)/(2*pi*R_orb)=R_Hill/(pi*R_orb) of the light curve 
# To be on the safe side, we will keep three times that, hence R_Hill/R_orb.
i3hill=int(((R_Hill(starmass,mass,R_orb)/R_orb) * periodo) / tlapse) # length in time bins of 3 times the Hill radius
i5dur=int(5.*tdur/tlapse) # length in time bins of 5 times the duration of a transit
if i3hill>=i5dur:
    ilen=i3hill # length in time bins of the region of interest of a transit
else:
    ilen=i5dur

if ilen % 2 ==0:
    ilen=ilen+1    # We force it to be odd so that there is a central bin
xtimes=(np.arange(ilen) - int(ilen/2))*tlapse  # conversion of the above to times (in days), centered on '0'

ntrans=int((tlast-t0)/periodo)
print('Number of a priori transits',ntrans)

# We create a matrix to store the light curves of each transit. Each transit has "ilen" time pixels.
# curvas[i,j,0] stores the time corresponding to time pixel j 
# curvas[i,j,1] stores the flux corresponding to time pixel j

curvas=np.zeros([ntrans,ilen,2])-np.nan # We initialize it so that, where there is no flux data, the NaN flag remains


feos=[]   # Stores the transits where there is no data
sigmacurva=[]  # Estimates the standard deviation of the curve outside the transit

for itran in np.arange(ntrans):       # Loops over all the transits

    # in the index idx we identify the time values corresponding to the nth transit (itran) in the data.
    idx=(lctime>itran*periodo+(tmin+xtimes[0])-(tlapse/2.))&(lctime<itran*periodo+(tmin+xtimes[-1])+(tlapse/2.))

    # We temporarily store in lctran_t,lctran_f the times and fluxes of the nth transit
    lctran_t=lctime[idx]
    lctran_f=lcflux[idx]
    lctran_f2=lcflux[idx] #dummy copy of the above from which we will delete the transit region, in order to fit the rest to a polynomial


    # We are going to flatten the background of the light curve by fitting a polynomial of order "orden"
    
    if len(lctran_t)>0:      # This indicates that there is Kepler data for that transit -otherwise it would be 0
        idx2=(lctran_t>((itran*periodo)+tmin-tdur/2.))&(lctran_t<((itran*periodo)+tmin+tdur/2.))  # transit region (tdur=transit duration)
        lctran_f2[idx2]=np.nan # we "erase" the transit zone in lctran_f2 by turning it into NaNs
        idx3=np.isfinite(lctran_t) & np.isfinite(lctran_f2) # identifies the indices of lctran_f2 with values that are not NaN
        fito=Polynomial(polyfit(lctran_t[idx3],lctran_f2[idx3],orden)) # function that fits that region to a polynomial of order "orden"
        yfit=fito(lctran_t) # stores the result of the fit

        lctran_f_corr=lctran_f/yfit  # we correct the whole lctran_f (note, not lctran_f2) with the result of the fit
        sigmacurva.append(np.nanstd(lctran_f2/yfit))
        
        # We store the corrected curve in the transit matrix "curvas"
        for ii in np.arange(len(lctran_t)):

            idx=abs(xtimes-(lctran_t[ii]-(itran*periodo+tmin)))<tlapse/2.   # Detects where the current point fits best
            if (len(np.where(idx == 1)[0]))==1:                              # (if it does)
                id0=int(np.where(idx == 1)[0])                                  # Converts it to a numeric value

                curvas[itran,id0,0]=lctran_t[ii]      # times
                curvas[itran,id0,1]=lctran_f_corr[ii] # corrected fluxes
        plt.xlabel(itran)
        plt.plot(lctran_t,lctran_f)
        plt.plot(lctran_t,lctran_f_corr)
        plt.show()
    else:
        feos.append(itran)



# (4) WE REMOVE UGLY TRANSITS AND SHAVE THE DATA

# Add manually here any additional transit you want to remove and uncoment:
#feos=feos+[10,65,73,74,80]   #for example include the transits that look bad


# Transits with weird things get deleted:
for tranfeo in feos:
    curvas[tranfeo,:,1]=np.nan


print('stdev of the light curve',np.nanmean(sigmacurva))

sigmalevel=3.    #change here if you want another "shaving intensity"

# Our shaving reference is the average transit, inside the transit, and "1" outside the transit
transito_medio=np.nanmean(curvas[:,:,1],0)[:]
idx=abs(xtimes)>tdur/2.
transito_medio[idx]=1.

# Values that deviate by more than "sigmalevel" times the standard deviation get deleted:
for itran in np.arange(ntrans):
    idx=abs(curvas[itran,:,1]-transito_medio)>(sigmalevel*np.nanmean(sigmacurva))
    curvas[itran,idx,1]=np.nan


# (5) WE DO A SHIFT OF LESS THAN ONE UNIT, INTERPOLATING
# This is essential to align the transits correctly. With the above we have aligned them with an
# uncertainty of +-tlapse/2, tlapse being the size of a time bin (approx. 1 minute in short cadence,
# half an hour in long cadence), but that is not fine enough and would produce a false exomoon signal.
# We are going to recenter them so that the central bin coincides exactly with the center of the transit

tiempos=np.zeros([ntrans,ilen]) # here come the "good" times with respect to which we are going to realign the transits
for itran in np.arange(ntrans):
    for ipix in np.arange(ilen):
        tiempos[itran,ipix]=itran*periodo+tmin+xtimes[ipix]

fluxinter=np.zeros([ntrans,ilen]) # We will store here the fluxes of the curves from the previous block, once "shifted" and interpolated
fluxinter2=np.zeros([ntrans,ilen]) # Same, but correcting for possible wobble (TTV) due to gravitational tugs from another body (see further below)

condatos=np.zeros(ntrans)
for itran in np.arange(ntrans):
    tshft=0.
    for ipix in np.arange(ilen):
        if np.isfinite(curvas[itran,ipix,1]):  # If the first values of the curve were NaN it will ignore them until it reaches the 1st numeric value
            tshft=(curvas[itran,ipix,0]-tiempos[itran,ipix])/tlapse # Computes the amount of shift (fraction of a bin) needed to realign this transit
            condatos[itran]=1
            break                                                    # It only needs to do this for one value; as soon as it has it, we exit the loop
    fluxinter[itran]=shift(curvas[itran,:,1],tshft,mode='grid-wrap',order=1)  # Does a non-integer shift (actually a roll) of the fluxes, interpolating linearly (order=1)

# We compute the mean curve; if we do not correct for wobble, this is the one we will subtract. Otherwise, we will use it as a reference to recenter the
# transits and then recompute it again

curvaprom=np.nanmean(fluxinter,0)[:]  
plt.plot(xtimes,curvaprom)
plt.show()


# (6) WE SUBTRACT THE AVERAGE TRANSIT FROM THE INDIVIDUAL TRANSITS, EITHER RECENTERING BEFOREHAND OR NOT

# We compute the offsets of each transit with respect to the average transit:
xfind=np.arange(-250.,250.,0.1)
offset = np.zeros(ntrans)

for itran in np.arange(ntrans):
    ttemp = np.nan_to_num(fluxinter[itran],nan=curvaprom)  #where values are missing we substitute those of the mean curve - this improves the result of the correlation
    ycorr = np.correlate((1.-ttemp),(1.-curvaprom),mode='same') / np.nansum((1.-curvaprom)**2)
    xcorr = np.array(range(0,ycorr.shape[0])) - ycorr.size/2. 
            
    #xcorrmask=np.ma.masked_where(np.logical_or((abs(xcorr) > 200),(abs(xcorr) < 5)), xcorr)
    xcorrmask=np.ma.masked_where(abs(xcorr) > 200,xcorr)
    
    xcm=np.ma.compressed(np.ma.masked_where(np.ma.getmask(xcorrmask),xcorr))
    ycm=np.ma.compressed(np.ma.masked_where(np.ma.getmask(xcorrmask),ycorr))

    ycm_polyfit = np.polyfit(xcm,ycm,10)
    ycm_cont = np.poly1d(ycm_polyfit)
    ycm_fit = ycm_cont(xcm)
    
    xmax2 = xfind[np.argmax(ycm_cont(xfind))] 
    offset[itran]=(xmax2+0.5)  # vector that stores the offsets (in bins) of each transit
idx=abs(offset)>200
offset[idx]=0  # if the offset is too big is due to a bad correlation artifact. We equal them to 0.

plt.plot(np.arange(len(offset)),offset*tlapse*(60.*60.*24.),'*')
plt.xlabel('Transit number')
plt.ylabel('Offsets in seconds')
plt.show()


offsets=offset[condatos>0]*tlapse*(60.*60.*24.)   # Passing from bins to seconds
tiempos=periodo*np.arange(len(offset))[condatos>0]

frequency, power = LombScargle(tiempos,offsets).autopower(minimum_frequency=1/Tlumax,maximum_frequency=1/Tlumin)  # Calculating the Lomb-Scargle periodogram of the offsets

fig = plt.figure(figsize=(20, 6), dpi=100)
plt.xlabel("Orbital period (d)")

plt.plot(1/frequency,power)
plt.xlim(Tlumin,Tlumax)
plt.ylim(0.,np.mean(power)*10.)
plt.show()

invfreq=1/frequency
infre=invfreq[power==np.max(power[invfreq>periodo/3.])][0]
tiemposphased=tiempos%infre

# Trying a fit to a sinusoid of the folded offsets:

try:
    popt, pcov = curve_fit(sine_func, tiemposphased, offsets,bounds=([-np.inf,1/infre,0.],[np.inf,1.00000001/infre,np.pi]))
except:
    print('Petó el ajuste')
else:
    plt.plot(tiemposphased,offsets,'*')
    plt.plot(tiemposphased,sine_func(tiemposphased,*popt),'*')
    plt.xlabel('times phased to '+str(infre)+' days')
    plt.ylabel('Offsets en segundos')
    plt.show()
    print(f"Fitted parameters: amplitude={popt[0]:.2f}, frequency={popt[1]:.2f}")
finally:
    print(periodo/infre)

# Earlier we computed the offsets with respect to fluxinter; now, to avoid re-interpolating again what was already interpolated,
# we are going to recompute the corrections from curvas and recenter from the fluxes in this array

offset2 = np.zeros(ntrans)

for itran in np.arange(ntrans):  
    ttemp = np.nan_to_num(curvas[itran,:,1],nan=curvaprom)  #where values are missing we substitute those of the mean curve, as before
    ycorr = np.correlate((1.-ttemp),(1.-curvaprom),mode='same') / np.nansum((1.-curvaprom)**2)
    xcorr = np.array(range(0,ycorr.shape[0])) - ycorr.size/2. 
            
    #xcorrmask=np.ma.masked_where(np.logical_or((abs(xcorr) > 200),(abs(xcorr) < 5)), xcorr)
    xcorrmask=np.ma.masked_where(abs(xcorr) > 200,xcorr)
    
    xcm=np.ma.compressed(np.ma.masked_where(np.ma.getmask(xcorrmask),xcorr))
    ycm=np.ma.compressed(np.ma.masked_where(np.ma.getmask(xcorrmask),ycorr))

    ycm_polyfit = np.polyfit(xcm,ycm,10)
    ycm_cont = np.poly1d(ycm_polyfit)
    ycm_fit = ycm_cont(xcm)
    
    xmax2 = xfind[np.argmax(ycm_cont(xfind))] 
    offset2[itran]=(xmax2+0.5)  # vector that stores the offsets of each transit
    
for itran in np.arange(ntrans):
    fluxinter2[itran]=shift(curvas[itran,:,1],-offset2[itran],mode='grid-wrap',order=1)  # We recenter the transit
curvaprom2=np.nanmean(fluxinter2,0)[:]  # we recompute the definitive mean curve once the transits have been recentered
plt.plot(xtimes,curvaprom2)
plt.show()
    

# "curvasin" is the collection of transits with the average planetary transit subtracted;
# if any residue remains there, it may be the footprint of an exomoon
# curvasin[itran,0] stores the transits from which the planet's transit has been subtracted
# without correcting for potential TTVs, whereas curvasin[itran,1` stores the transits from which
#  the planet's transit has been subtracted after correcting for potential TTVs. 
curvasin=np.zeros([ntrans,2,ilen])              
for itran in np.arange(ntrans):                 # Removing the average transit 
    curvasin[itran,0]=fluxinter[itran]-(curvaprom-1.)
    curvasin[itran,1]=fluxinter2[itran]-(curvaprom2-1.)


# (7) WE COMPUTE THE STACKS FOR ALL POSSIBLE ORBITAL RADII, ORBITAL PERIODS AND PHASES OF THE PUTATIVE EXOMOON

# We define some values so we don't have to compute them each time inside the loop (and we save a few multiplications):
dospi=2*np.pi
periodo_dospi_R_orb=periodo/(dospi*R_orb)

# We are going to loop assuming that the moon orbits the planet with a certain period "Tlu" (which by Kepler implies
# an orbital distance "Rlu") and, starting at a certain phase "fase", we calculate which position of the light curve
# corresponds to those values in each transit, realign each transit centering it on that hypothetical value, and then
# we stack all the curves, averaging them and storing them in stack(Tlu,fase).
# As a control (contratest), we perform the operation but assuming a wrong position for the planet
# (the stacks will be computed as if the putative exomoon orbited around a point where there is no planet).
# This is the most expensive loop of all; it may take several days to run.

stack=np.zeros([Tluint,Fasint,ilen])     # Here we will store the stacks (remember: "ilen" is the length in bins of a transit)
stackstd=np.zeros([Tluint,Fasint,ilen])  # And here their corresponding standard deviation (to later compute the chi2)

stackcorr=np.zeros([Tluint,Fasint,ilen])     # Same but for the recentered transits
stackcorrstd=np.zeros([Tluint,Fasint,ilen])  # And here their corresponding standard deviation (to later compute the chi2)

stackoff=np.zeros([Tluint,Fasint,ilen])  # Ditto for the off-source on-transit
stackoffstd=np.zeros([Tluint,Fasint,ilen])

ideltashift=round(ilen*2./5.)  #how many bins we shift the off-source (in this example, 2/5 the length of the transit)


# Here we do the provisional stack; 
# acumulador[i,j,0] stores the uncorrected curve, acumulador[i,j,1] stores the TTV-corrected curve, 
# and acumulador[i,j,2] the off-source 

acumulador=np.zeros([ntrans,ilen,3])     

itac=-1
for Tlu in np.linspace(Tlumin,Tlumax,Tluint):   # Loop over the orbital period of the putative moon, in days
    itac=itac+1                                 # A counter for which period iteration we are on
    dospi_periodo_Tlu=dospi*periodo/Tlu         # We define this so as not to have to compute it each time inside the next loop
    Rlu=R_luna(Tlu,mass)                        # Computes which radius the current period corresponds to, according to Kepler's 3rd law
    print(Tluint,itac)
    ifas=-1
    for fase in np.linspace(0.,dospi,Fasint,endpoint=False):    # Loop over the possible initial phase
        ifas=ifas+1                                             # A counter for which phase iteration we are on

        acumulador=0.*acumulador  # Before starting the loop over phases, we reset the accumulator

        for itran in np.arange(ntrans):    # Loop over all transits

            # For those values of Tlu, Rlu and phase, we compute the time offset between
            # the planet and the moon for each transit:
            Tdelta=Rlu*np.cos(fase+(itran*dospi_periodo_Tlu))*periodo_dospi_R_orb

            idelta=round(Tdelta/tlapse)   # Passing time to bins

            acumulador[itran,:,0]=np.roll(curvasin[itran,0],idelta)   # we permute the light curve by that many bins and store it in acumulador
            acumulador[itran,:,1]=np.roll(curvasin[itran,1],idelta)  
            acumulador[itran,:,2]=np.roll(curvasin[itran,1],idelta+ideltashift)  # we additionally shift the off-source by ideltashift (we will try to search the moon in a wrong position)

        nonansqr0=np.sum(np.where(~np.isnan(acumulador[:,:,0]),1,0),axis=0)**0.5
        nonansqr1=np.sum(np.where(~np.isnan(acumulador[:,:,1]),1,0),axis=0)**0.5
        nonansqr2=np.sum(np.where(~np.isnan(acumulador[:,:,2]),1,0),axis=0)**0.5
        
        stack[itac,ifas]=np.nanmean(acumulador,0)[:,0]    # We average it, so that the continuum comes out as 1
        stackstd[itac,ifas]=(np.nanstd(acumulador,0)[:,0])/nonansqr0  # and we store the standard deviation for the later computation of chi2
        stackcorr[itac,ifas]=np.nanmean(acumulador,0)[:,1]    
        stackcorrstd[itac,ifas]=(np.nanstd(acumulador,0)[:,1])/nonansqr1  
        stackoff[itac,ifas]=np.nanmean(acumulador,0)[:,2] # Ditto for the off-source
        stackoffstd[itac,ifas]=(np.nanstd(acumulador,0)[:,2])/nonansqr2



# (8) ON THE STACKS ALREADY MADE, WE COMPUTE THE CHI2 OF THE BEST FIT FOR EACH PHASE, RADIUS AND PERIOD:

# Here we will store the goodness of fit; first batch, on the planet's position; second, on the contratest position
Inte=np.zeros([Tluint,Fasint])    # Optimal chi2 scale factor
chi2=np.zeros([Tluint,Fasint])    # Chi2 of the stack vs the theoretical moon curve
chi2_0=np.zeros([Tluint,Fasint])  # Chi2 of the stack vs the curve of value=1
rfit=np.zeros([Tluint,Fasint])    # Here we store the figure of merit for the TTV-uncorrected transits

Intecorr=np.zeros([Tluint,Fasint])    # Optimal chi2 scale factor
chi2corr=np.zeros([Tluint,Fasint])    # Chi2 of the stack vs the theoretical moon curve
chi2_0corr=np.zeros([Tluint,Fasint])  # Chi2 of the stack vs the curve of value=1
rfitcorr=np.zeros([Tluint,Fasint])    # Here we store the figure of merit for the TTV-corrected transits

# Ditto for the off-source:
Inteoff=np.zeros([Tluint,Fasint])    # Optimal chi2 scale factor
chi2off=np.zeros([Tluint,Fasint])    # Chi2 of the stack vs the theoretical moon curve
chi2_0off=np.zeros([Tluint,Fasint])  # Chi2 of the stack vs the curve of value=1
rfitoff=np.zeros([Tluint,Fasint])    # Here we store the figure of merit for the off-source control


# Given that, if there were a moon, the mean curve "curvaprom" computed earlier would also contain part of the
# lunar transit signal, by subtracting curvaprom from each individual transit we would have overcorrected the lunar transits
# That means the expected lunar transit signal is no longer --.__.-- but something like --'\/`--
# although its exact shape depends on the moon's orbital radius. In what follows we are going to generate those search templates
# (kernels), which we will store in the vector "aa" and use to look for the signal in the previous stacks.

lunateo=np.zeros([Tluint,ilen])  # Here we will store the average light curve expected for the moon at a given orbital period (it is a function of the radius)
aa=np.zeros(ilen)  # Here we will store the current kernel (basically, 1-lunateo) for the current period

curvalunada=np.zeros([100,ilen])  # two auxiliary arrays to carry out the computation (it is computed for the whole circle
curvalunasin=np.zeros([100,ilen]) # every 3.6º, i.e. 100 bins; that is more than enough to estimate the kernel)

for itac in np.arange(Tluint):  # Loop over the possible orbital periods of the possible exomoon
    
    Tlu=np.linspace(Tlumin,Tlumax,Tluint)[itac]
    Rlu=R_luna(Tlu,mass)   # We compute the corresponding radius according to Kepler's 3rd law
    
    # We generate the theoretical shape of the lunar transit "lunateo" from "curvaprom",
    # and with them the kernels "aa" 

    # We compute the theoretical shape expected for the lunar transit after subtracting the planetary one:    
    curvalunada=0.*curvalunada   # we reset the auxiliary matrices for this radius/period
    curvalunasin=0.*curvalunasin
    frac=0.01                # We initially assume a moon that is 1% the size of the planet (it is a scale factor that is readjusted later, it is not relevant)

    itic=-1
    for fase in np.linspace(0.,dospi,100,endpoint=False):    
        itic=itic+1
        Tdelta=Rlu*np.cos(fase)*periodo_dospi_R_orb
        idelta=round(Tdelta/tlapse)   # That corresponds to that many time bins

        # From the planet's mean curve we subtract a rescaled (frac) version shifted according to the phase, 
        # to simulate a moon
        curvalunada[itic]=curvaprom-(frac*(1.-np.roll(curvaprom,idelta)))  

    # curvapromnew is the average of the previous curves, and we subtract it from the curves of each phase,
    # to estimate what shape a "de-mooned" transit should have.     
    curvapromnew=np.mean(curvalunada,0)
    curvalunasin[:]=curvalunada[:]-(curvapromnew-1.)

    itic=-1
    for fase in np.linspace(0.,dospi,100,endpoint=False):   
        itic=itic+1
        Tdelta=Rlu*np.cos(fase)*periodo_dospi_R_orb
        idelta=round(Tdelta/tlapse)   # That corresponds to that many time bins

        lunateo[itac]=lunateo[itac]+np.roll(curvalunasin[itic],-idelta)  # The sign of idelta must be the opposite of that in the previous loop
    lunateo[itac]=lunateo[itac]/100.                                     # This is the theoretical shape of the lunar transit for that period/radius
    aa=(1-lunateo[itac])/np.sum((1-lunateo[itac])[(1-lunateo[itac])>=0])       # And this is the corresponding search kernel
    
    # Once we have obtained the search kernel, we are going to compare it with all the stacks obtained in the previous block
    # for the current period, and for the different phases, with the kernel. And with that we calculate the chi2.    
    # We also compare with a flat curve (no lunar transit).
    for ifas in np.arange(Fasint):

        Inte[itac,ifas]=abs(np.sum((1.-stack[itac,ifas])*aa)/np.sum(aa*aa))
        chi2[itac,ifas]=np.sum((((1.-stack[itac,ifas])-(Inte[itac,ifas]*aa))/stackstd[itac,ifas])**2)
        chi2_0[itac,ifas]=np.sum(((1.-stack[itac,ifas])/stackstd[itac,ifas])**2)

        Intecorr[itac,ifas]=abs(np.sum((1.-stackcorr[itac,ifas])*aa)/np.sum(aa*aa))
        chi2corr[itac,ifas]=np.sum((((1.-stackcorr[itac,ifas])-(Intecorr[itac,ifas]*aa))/stackcorrstd[itac,ifas])**2)
        chi2_0corr[itac,ifas]=np.sum(((1.-stackcorr[itac,ifas])/stackcorrstd[itac,ifas])**2)

        Inteoff[itac,ifas]=abs(np.sum((1.-stackoff[itac,ifas])*aa)/np.sum(aa*aa))
        chi2off[itac,ifas]=np.sum((((1.-stackoff[itac,ifas])-(Inteoff[itac,ifas]*aa))/stackoffstd[itac,ifas])**2)
        chi2_0off[itac,ifas]=np.sum(((1.-stackoff[itac,ifas])/stackoffstd[itac,ifas])**2)


# (9) WE KEEP THE BEST FIGURE OF MERIT:

    
# Converting the times to array indices
Tidx=np.linspace(Tlumin,Tlumax,Tluint)

chidif=chi2_0-chi2                   # FoM for TTV-uncorrected transits
chidifcorr=chi2_0corr-chi2corr       # FoM for TTV-corrected transits
chidifoff=chi2_0off-chi2off          # FoM for the off-source control

# we find the maxima along axis 1 (phases) i.e., for each possible period T_lu we keep the phase
# that gives the largest difference between both chi2
rfit=np.max(chidif,1)
rfitcorr=np.max(chidifcorr,1)
rfitoff=np.max(chidifoff,1)

# we identify which lunar period value corresponds to the maximum of chi2_0-chi2 (and compute its corresponding radius)
Tlu_max=float(Tidx[np.where(rfit==np.max(rfit))[0]])
Rlu_max=R_luna(Tlu_max,mass)

# we identify which lunar period value corresponds to the maximum of chi2_0-chi2 (and compute its corresponding radius)
Tlu_maxcorr=float(Tidx[np.where(rfitcorr==np.max(rfitcorr))[0]])
Rlu_maxcorr=R_luna(Tlu_maxcorr,mass)

# Ditto for the off-source
Tlu_maxoff=float(Tidx[np.where(rfitoff==np.max(rfitoff))[0]])
Rlu_maxoff=R_luna(Tlu_maxoff,mass)


# We print the results
fig = plt.figure(figsize=(20, 15), dpi=100)
gs = fig.add_gridspec(4, hspace=0)
axs = gs.subplots(sharex=True, sharey=True)
fig.suptitle('Merit figure vs period (d)')
plt.xlabel("Orbital period (d)")
plt.ylim(0,np.max(rfit))
plt.xlim(Tlumin,Tlumax)
axs[0].set_ylabel("On-source")
axs[0].plot(np.linspace(Tlumin,Tlumax,Tluint),rfit)
axs[1].set_ylabel("Lomb-Scargle periodogram")
axs[1].plot(1/frequency,power*np.max(rfit)/(8.*np.mean(power)))
axs[2].set_ylabel("On-source TTV-corrected")
axs[2].plot(np.linspace(Tlumin,Tlumax,Tluint),rfitcorr)
axs[3].set_ylabel("Off-source control")
axs[3].plot(np.linspace(Tlumin,Tlumax,Tluint),rfitoff)

# Hide x labels and tick labels for all but bottom plot.
for ax in axs:
    ax.label_outer()
plt.show()


stck=stack[np.where(chidif==np.max(rfit))][0]
stckcorr=stackcorr[np.where(chidifcorr==np.max(rfitcorr))][0]
stckoff=stackoff[np.where(chidifoff==np.max(rfitoff))][0]

# Plotting the best stacks in each case
plt.tittle("Uncorrected")
plt.scatter(xtimes,stck)
plt.show()
plt.tittle("TTV-corrected")
plt.scatter(xtimes,stckcorr)
plt.show()
plt.tittle("Off-source control")
plt.scatter(xtimes,stckcorr)
plt.show()

