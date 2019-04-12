import numpy as np, matplotlib.pyplot as plt
import astropy.cosmology as c
import scipy.stats as s
import sys

cosmo = c.Planck15

# dispersion measures of CHIME FRBs (total and Galactic / NE2001)
dms_tot_c = np.array([109.61,169.134,238.32,317.37,414.95,642.07,656.2,715.98,739.98,802.57,849.047,1006.84])
dms_gal_c = np.array([31.,47.,41.,95.,104.,21.,90.,71.,41.,83.,57.,28.])
nFRB = 12 # number of frbs

frb_dms = dms_tot_c-dms_gal_c-50.-DM_host # 50 is minimum in prochaska/zheng19
frb_dms[frb_dms<0.] = 0.

# search reach
# CHIME was 4600 beam-days * 0.256 deg^2 = 28262.4 deg^2 hr
total_time = 7.8e-5

def igm_dm(d,f_igm=0.84):

    zz = (np.arange(10000)+1.)*4./10000.
    diffs = np.abs(d-cosmo.comoving_distance(zz).value)
    z = zz[np.argmin(diffs)]
    
    myz = 10.**((np.arange(1000)+0.5)*(np.log10(z)+3.)/1000.-3.)
    mydz = 10.**((np.arange(1000)+1.)*(np.log10(z)+3.)/1000.-3.)-10.**((np.arange(1000)+0.)*(np.log10(z)+3.)/1000.-3.)
    Iz = np.sum(mydz*(1.+myz)**2./np.sqrt(cosmo.Om0*(1.+myz)**3.+cosmo.Ode0))
    
    return 935.*f_igm*(cosmo.h/0.7)**(-1.)*Iz

# returns probability of <d given DM
# d in Mpc
# assume 50 to 80 from halo: prochaska & zheng 19
def prob_d(dm,d,uncc_halo=30.,scat=10.):

    # fix unc_halo
    # 2.5 only really matters for nearest FRB, which is always within volume considered.
    if dm < uncc_halo + 2.5*scat:
        unc_halo = dm - 2.5*scat
    else:
        unc_halo = uncc_halo
    
    # get dm for d
    dm_pred = igm_dm(d)

    # get normalization factors
    pk_of_gaus = 1./np.sqrt(2.*np.pi*scat*scat)
    tot_box = unc_halo*pk_of_gaus
    gprob = 1./(1.+tot_box)
    tprob = tot_box/(1.+tot_box)
    
    # get prob

    # this means that the DM is larger than the DM(d) -> FRB is certainly outside d
    if dm-unc_halo>=dm_pred:
        ddm = (-dm_pred+dm-unc_halo)/scat
        prob = ((s.chisqprob(ddm**2.,1))/2.)*gprob
    
    # this means that the DM is smaller than the DM(d) -> FRB is within d
    if dm-unc_halo<dm_pred:
    
        if dm<dm_pred:
            ddm = (dm_pred-dm)/scat
            prob = 1.-gprob*(s.chisqprob(ddm**2.,1))/2.
    
        else:
            ddm = dm_pred-(dm-unc_halo)
            prob = (ddm/unc_halo)*tprob+0.5*gprob

    if prob<1e-50:
        prob=1e-50
    return prob


# returns 95% conf upper limit on distance 
def d_lt(dm,conf=0.95):

    d = np.linspace(10.,900.,200)
    p = np.zeros(200)

    fnd = 0
    d_use = 0.
    for i in range(200):
        p[i] = prob_d(dm,d[i])
        if p[i]>conf:
            if fnd==0:
                fnd=1
                d_use = d[i]

    return d_use


# probs of each DM being within d
def probs_DM(d):

    probs = np.zeros(nFRB)
    for i in np.arange(nFRB):
        probs[i] = prob_d(frb_dms[i],d)

    return probs

# to estimate correction in rate due to time dilation
def rate_corr(d):

    zz = (np.arange(10000)+1.)*4./10000.
    diffs = np.abs(d-cosmo.comoving_distance(zz).value)
    z = zz[np.argmin(diffs)]

    tot_v = cosmo.comoving_volume(z).value
    v_prev = 0.
    av = 0.
    for zi in np.linspace(0.000001,z,100):

        v1 = cosmo.comoving_volume(zi).value
        vol = v1-v_prev
        av += (vol/tot_v)*(1.+zi)
        v_prev = v1

    return av

        
    


# main. calculates 90% confidence lower limits on FRB rate in cases containing two and three events

dm_host = np.asarray([0.,10.,20.,23.,30.,35.,40.,50.])

for dm in dms:

    # in Mpc
    d2 = d_lt(frb_dms[1]-dm)
    d3 = d_lt(frb_dms[2]-dm)

    # prefactors from Gehrels86 Table 2
    r2 = 0.532/((4./3.)*np.pi*(d2/1000.)**3.)/total_time
    r3 = 1.102/((4./3.)*np.pi*(d3/1000.)**3.)/total_time

    print dm,r2,r3


