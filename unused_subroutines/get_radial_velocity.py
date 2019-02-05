'''
Created on 9 Nov. 2017

@author: Christoph Bergmann
'''


import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import scipy.optimize as op
import time
from helper_functions import gausslike_with_amp_and_offset_and_slope, central_parts_of_mask
from flat_fielding import deblaze_orders



# #speed of light in m/s
# c = 2.99792458e8
# #oversampling factor for logarithmic wavelength rebinning
# osf = 2
# 
# #read dispersion solution from file
# dispsol = np.load('/Users/christoph/OneDrive - UNSW/dispsol/mean_dispsol_by_orders_from_zemax.npy').item()
# 
# #read extracted spectrum from files (obviously this needs to be improved)
# xx = np.arange(4096)
# 
# #for simulated data only
# wl = get_simu_dispsol()
#     
# 
# #re-bin into logarithmic wavelength space for cross-correlation, b/c delta_log_wl = c * delta_v
# refdata = ascii.read('/Users/christoph/OneDrive - UNSW/rvtest/high_SNR_solar_template.txt', names=('pixnum','wl','flux','err'))
# # flux1 = data1['flux']
# # err1 = data1['err']
# #data2 = ascii.read('/Users/christoph/OneDrive - UNSW/rvtest/solar_red100ms.txt', names=('pixnum','wl','flux','err'))
# # flux2 = data2['flux']
# # err2 = data2['err']
# # obsdata = ascii.read('/Users/christoph/OneDrive - UNSW/rvtest/solar_0ms.txt', names=('pixnum','wl','flux','err'))
# # obsdata = ascii.read('/Users/christoph/OneDrive - UNSW/rvtest/solar_red100ms.txt', names=('pixnum','wl','flux','err'))
# obsdata = ascii.read('/Users/christoph/OneDrive - UNSW/rvtest/solar_red1000ms.txt', names=('pixnum','wl','flux','err'))
# # flux3 = data3['flux']
# # err3 = data3['err']
# flatdata = ascii.read('/Users/christoph/OneDrive - UNSW/rvtest/tramline_extracted_flat.txt', names=('pixnum','wl','flux','err'))
# # flatflux = flatdata['flux']
# # flaterr = flatdata['err']
# #wl1 = data1['wl']
# #wl2 = data2['wl']
# # wl = data1['wl']
# # pixnum = data1['pixnum']





# # f0 = template
# f0 = {}
# err0 = {}
# # f = observation
# f = {}
# err = {}
# # f_flat = master_white
# f_flat = {}
# err_flat = {}
# 
# ref_wl = {}
# obs_wl = {}
# flat_wl = {}
# for m in range(len(wl)):
#     ord = 'order_'+str(m+1).zfill(2)
#     f0[ord] = np.array(refdata['flux'][m*4096:(m+1)*4096])[::-1]              # need to turn around if using Zemax dispsol, otherwise np.interp fails
#     f[ord] = np.array(obsdata['flux'][m*4096:(m+1)*4096])[::-1]
#     f_flat[ord] = np.array(flatdata['flux'][m*4096:(m+1)*4096])[::-1]
#     err0[ord] = np.array(refdata['err'][m*4096:(m+1)*4096])[::-1]
#     err[ord] = np.array(obsdata['err'][m*4096:(m+1)*4096])[::-1]
#     err_flat[ord] = np.array(flatdata['err'][m*4096:(m+1)*4096])[::-1]
#     ref_wl[ord] = np.array(refdata['wl'][m*4096:(m+1)*4096])[::-1]            # units do not matter, as log(x) - log(y) = log(ax) - log(ay) = log(a) + log(x) - log(a) - log(y)
#     obs_wl[ord] = np.array(obsdata['wl'][m*4096:(m+1)*4096])[::-1]
#     flat_wl[ord] = np.array(flatdata['wl'][m*4096:(m+1)*4096])[::-1]
    
    

def get_rvs_from_xcorr(extracted_spectra, obsnames, mask, smoothed_flat, debug_level=0):
    """
    This is a wrapper for the actual RV routine "get_RV_from_xcorr", which is called for all observations within 'obsnames'.
    
    INPUT:
    'extracted_spectra'  : dictionary containing keys 'pix', 'wl', 'flux', and 'err' for every observation (each containing keys=orders), plus the template and the master white
    'obsnames'           : list containing the names of the observations
    'mask'               : dictionary of masks from 'find_stripes' (with the keys being the orders)
    'smoothed_flat'      : dictionary containing the smoothed master white for each order (with the keys being the orders)
    'debug_level'        : boolean - for debugging...
    
    OUTPUT:
    'rv'     : RVs (dictionary with 'obsnames' --> 'orders' as keys)
    'rverr'  : RV errors ((dictionary with 'obsnames' --> 'orders' as keys)
    """
    
    rv = {}
    rverr = {}
    
    f0 = extracted_spectra['template']['flux'].copy()
    wl0 = extracted_spectra['template']['wl'].copy()
    
    for obs in sorted(obsnames):
        #prepare arrays
        f = extracted_spectra[obs]['flux'].copy()
        err = extracted_spectra[obs]['err'].copy()
        wl = extracted_spectra[obs]['wl'].copy()
        
        #if using cross-correlation, we need to de-blaze the spectra first
        f_dblz, err_dblz = deblaze_orders(f, wl, smoothed_flat, mask, err=err)
        f0_dblz = deblaze_orders(f0, wl0, smoothed_flat, mask, err=None)
        
        #we also only want to use the central TRUE parts of the masks, ie want ONE consecutive stretch per order
        cenmask = central_parts_of_mask(mask)
        
        #call RV routine
        if debug_level >= 1:
            print('Calculating RV for observation: '+obs)
        rv[obs],rverr[obs] = get_RV_from_xcorr(f_dblz, err_dblz, wl, f0_dblz, wl0, mask=cenmask, filter_width=25, debug_level=0)

    return rv,rverr

    
    
    
    
def get_RV_from_xcorr(f, err, wl, f0, wl0, mask=None, smoothed_flat=None, osf=2, delta_log_wl=1e-6, relgrid=False,
                      filter_width=25, bad_threshold=0.05, simu=False, debug_level=0, timit=False):
    """
    This routine calculates the radial velocity of an observed spectrum relative to a template using cross-correlation. 
    Note that input spectra should be de-blazed already!!!
    If the mask from "find_stripes" has gaps, do the filtering for each segment independently. If no mask is provided, create a simple one on the fly.
    
    INPUT:
    'f'             : dictionary containing the observed flux (keys = orders)
    'err'           : dictionary containing the uncertainties in the observed flux (keys = orders)
    'wl'            : dictionary containing the wavelengths of the observed spectrum (keys = orders)
    'f0'            : dictionary containing the template (keys = orders) (errors assumed to be negligible)
    'wl0'           : dictionary containing the wavelengths of the template spectrum (keys = orders)
    'mask'          : mask-dictionary from "find_stripes" (keys = orders)
    'smoothed_flat' : if no mask is provided, need to provide the smoothed_flat, so that a mask can be created on the fly
    'osf'           : oversampling factor for the logarithmic wavelength rebinning (only used if 'relgrid' is TRUE)
    'delta_log_wl'  : stepsize of the log-wl grid (only used if 'relgrid' is FALSE)
    'relgrid'       : boolean - do you want to use an absolute stepsize of the log-wl grid, or relative using 'osf'?
    'filter_width'  : width of smoothing filter in pixels; needed b/c of edge effects of the smoothing; number of pixels to disregard should be >~ 2 * width of smoothing kernel  
    'bad_threshold' : if no mask is provided, create a mask that requires the flux in the extracted white to be larger than this fraction of the maximum flux in that order
    'simu'          : boolean - are you using ES simulated spectra? (only used if mask is not provided)
    'debug_level'   : boolean - for debugging...
    'timit'         : boolean - for timing the execution run time...
    
    OUTPUT:
    'rv'         : dictionary with the measured RVs for each order
    'rverr'      : dictionary with the uncertainties in the measured RVs for each order
    
    MODHIST:
    Dec 2017 - CMB create
    04/06/2018 - CMB fixed bug when turning around arrays (need to use new variable)
    28/06/2018 - CMB fixed bug with interpolation of log wls
    """
    
    if timit:
        start_time = time.time()
    
    #speed of light in m/s
    c = 2.99792458e8
    
    rv = {}
    rverr = {}
    
    #loop over orders
    for ord in sorted(f.iterkeys()):
        
        if debug_level >= 1:
            print(ord)
        
        #only use pixels that have enough signal
        if mask is None:
            normflat = smoothed_flat[ord]/np.max(smoothed_flat[ord])
            ordmask = np.ones(len(normflat), dtype = bool)
            if np.min(normflat) < bad_threshold:
                ordmask[normflat < bad_threshold] = False
                #once the blaze function falls below a certain value, exclude what's outside of that pixel column, even if it's above the threshold again, ie we want to only use a single consecutive region
                leftmask = ordmask[: len(ordmask)//2]
                leftkill_index = [i for i,x in enumerate(leftmask) if not x]
                try:
                    ordmask[: leftkill_index[0]] = False
                except:
                    pass
                rightmask = ordmask[len(ordmask)//2 :]
                rightkill_index = [i for i,x in enumerate(rightmask) if not x]
                if ord == 'order_01' and simu:
                    try:
                        #if not flipped then the line below must go in the leftkillindex thingy
                        #ordmask[: leftkill_index[-1] + 100] = False
                        ordmask[len(ordmask)//2 + rightkill_index[0] - 100 :] = False
                    except:
                        pass
                else:
                    try:
                        ordmask[len(mask)//2 + rightkill_index[-1] + 1 :] = False
                    except:
                        pass
        else:
            #ordmask  = mask[ord][::-1]
            ordmask  = mask[ord]
        
        #either way, disregard #(edge_cut) pixels at either end; this is slightly dodgy, but the gaussian filtering above introduces edge effects due to mode='reflect'
        ordmask[:2*int(filter_width)] = False
        ordmask[-2*int(filter_width):] = False
    
#         f0_unblazed = f0[ord] / np.max(f0[ord]) / normflat   #see COMMENT above
#         f_unblazed = f[ord] / np.max(f[ord]) / normflat
        
        #create logarithmic wavelength grid
        logwl = np.log(wl[ord])
        logwl0 = np.log(wl0[ord])
        if relgrid:
            logwlgrid = np.linspace(np.min(logwl[ordmask]), np.max(logwl[ordmask]), osf*np.sum(ordmask))
            delta_log_wl = logwlgrid[1] - logwlgrid[0]
            if debug_level >= 2:
                print(ord, ' :  delta_log_wl = ',delta_log_wl)
        else:
            logwlgrid = np.arange(np.min(logwl[ordmask]), np.max(logwl[ordmask]), delta_log_wl)
        
        #wavelength array must be increasing for "InterpolatedUnivariateSpline" to work --> turn arrays around if necessary!!!
        if (np.diff(logwl) < 0).any():
            logwl_sorted = logwl[::-1].copy()
            logwl0_sorted = logwl0[::-1].copy()
            ordmask_sorted = ordmask[::-1].copy()
            ord_f0_sorted = f0[ord][::-1].copy()
            ord_f_sorted = f[ord][::-1].copy()
        else:
            logwl_sorted = logwl.copy()
            logwl0_sorted = logwl0.copy()
            ordmask_sorted = ordmask.copy()
            ord_f0_sorted = f0[ord].copy()
            ord_f_sorted = f[ord].copy()
        
        #rebin spectra onto logarithmic wavelength grid
#         rebinned_f0 = np.interp(logwlgrid,logwl[mask],f0_unblazed[mask])
#         rebinned_f = np.interp(logwlgrid,logwl[mask],f_unblazed[mask])
        spl_ref_f0 = interp.InterpolatedUnivariateSpline(logwl0_sorted[ordmask_sorted], ord_f0_sorted[ordmask_sorted], k=3)    #slightly slower than linear, but best performance for cubic spline
        rebinned_f0 = spl_ref_f0(logwlgrid)
        spl_ref_f = interp.InterpolatedUnivariateSpline(logwl_sorted[ordmask_sorted], ord_f_sorted[ordmask_sorted], k=3)    #slightly slower than linear, but best performance for cubic spline
        rebinned_f = spl_ref_f(logwlgrid)
    
        # do we want to cross-correlate the entire order???
        #xc = np.correlate(rebinned_f0 - np.median(rebinned_f0), rebinned_f - np.median(rebinned_f), mode='same')
    #     #now this is slightly dodgy, but cutting off the edges works better because the division by the normflat introduces artefacts there
    #     if ord == 'order_01':
    #         xcorr_region = np.arange(2500,16000,1)
    #     else:
    #         xcorr_region = np.arange(2500,17500,1)
        
        xc = np.correlate(rebinned_f0 - np.median(rebinned_f0), rebinned_f - np.median(rebinned_f), mode='same')
        #now fit Gaussian to central section of CCF
        if relgrid:
            fitrangesize = osf*6    #this factor was simply eye-balled
        else:
            #fitrangesize = 30
            fitrangesize = int(np.round(0.0036 * len(xc) / 2. - 1,0))     #this factor was simply eye-balled
            
        xrange = np.arange(np.argmax(xc)-fitrangesize, np.argmax(xc)+fitrangesize+1, 1)
        #parameters: mu, sigma, amp, beta, offset, slope
        guess = np.array((np.argmax(xc), 0.0006 * len(xc), (xc[np.argmax(xc)]-xc[np.argmax(xc)-fitrangesize]), 2., xc[np.argmax(xc)-fitrangesize], 0.))
        #guess = np.array((np.argmax(xc), 5., (xc[np.argmax(xc)]-xc[np.argmax(xc)-fitrangesize]), 2., xc[np.argmax(xc)-fitrangesize], 0.))
        #guess = np.array((np.argmax(xc), 10., (xc[np.argmax(xc)]-xc[np.argmax(xc)-fitrangesize])/np.max(xc), 2., xc[np.argmax(xc)-fitrangesize]/np.max(xc), 0.))
        #popt, pcov = op.curve_fit(gaussian_with_offset_and_slope, xrange, xc[np.argmax(xc)-fitrangesize : np.argmax(xc)+fitrangesize+1]/np.max(xc[xrange]), p0=guess)
        #popt, pcov = op.curve_fit(gausslike_with_amp_and_offset_and_slope, xrange, xc[xrange]/np.max(xc[xrange]), p0=guess)
        popt, pcov = op.curve_fit(gausslike_with_amp_and_offset_and_slope, xrange, xc[xrange], p0=guess)
        mu = popt[0]
#         print(ord, f[ord][3000:3003])
#         print(ord, f[ord][::-1][3000:3003])
        mu_err = pcov[0,0]
        #convert to RV in m/s
        rv[ord] = c * (mu - (len(xc)//2)) * delta_log_wl
        rverr[ord] = c * mu_err * delta_log_wl
    
    if timit:
        delta_t = time.time() - start_time
        print('Time taken for calculating RV: '+str(np.round(delta_t,2))+' seconds')
    
    return rv,rverr





def get_RV_from_xcorr_2(f, wl, f0, wl0, mask=None, smoothed_flat=None, delta_log_wl=1e-6, relgrid=False, osf=5, addrange=40, 
                        fitrange=10, flipped=False, individual_fibres=True, individual_orders=True, debug_level=0, timit=False):
    """
    This routine calculates the radial velocity of an observed spectrum relative to a template using cross-correlation.
    Note that input spectra should be de-blazed already!!!
    If the mask from "find_stripes" has gaps, do the filtering for each segment independently. If no mask is provided, create a simple one on the fly.

    INPUT:
    'f'             : numpy array containing the observed flux (n_ord, n_fib, n_pix)
    'wl'            : numpy array containing the wavelengths of the observed spectrum (n_ord, n_fib, n_pix)
    'f0'            : numpy array containing the flux of the template spectrum (n_ord, n_fib, n_pix)
    'wl0'           : numpy array containing the wavelengths of the template spectrum (n_ord, n_fib, n_pix)
    'mask'          : mask-dictionary from "find_stripes" (keys = orders)
    'smoothed_flat' : if no mask is provided, need to provide the smoothed_flat, so that a mask can be created on the fly
    'delta_log_wl'  : stepsize of the log-wl grid (only used if 'relgrid' is FALSE)
    'relgrid'       : boolean - do you want to use an absolute stepsize of the log-wl grid (DEFAULT), or relative using 'osf'?
    'osf'           : oversampling factor for the logarithmic wavelength rebinning (only used if 'relgrid' is TRUE)
    'addrange'      : the central (2*addrange + 1) pixels of the CCFs will be added
    'fitrange'      : a Gauss-like function will be fitted to the central (2*fitrange + 1) pixels
    'flipped'       : boolean - reverse order of inputs to xcorr routine?
    'individual_fibres'  : boolean - do you want to return the RVs for individual fibres? (if FALSE, then the RV is calculated from the sum of the ind. fib. CCFs)
    'individual_orders'  : boolean - do you want to return the RVs for individual orders? (if FALSE, then the RV is calculated from the sum of the ind. ord. CCFs)
    'debug_level'   : boolean - for debugging...
    'timit'         : boolean - for timing the execution run time...

    OUTPUT:
    'rv'         : dictionary with the measured RVs for each order
    'rverr'      : dictionary with the uncertainties in the measured RVs for each order

    MODHIST:
    Dec 2017 - CMB create
    04/06/2018 - CMB fixed bug when turning around arrays (need to use new variable)
    28/06/2018 - CMB fixed bug with interpolation of log wls
    """

    if timit:
        start_time = time.time()

    # speed of light in m/s
    c = 2.99792458e8

    # the dummy wavelengths for orders 'order_01' and 'order_40' cannot be zero as we're taking a log!!!
    if wl.shape[0] == 40:
        wl[0, :, :] = 1.
        wl[-1, :, :] = 1.
        wl0[0, :, :] = 1.
        wl0[-1, :, :] = 1.
    elif wl.shape[0] == 39:
        wl[0, :, :] = 1.
        wl0[0, :, :] = 1.


    # make cross-correlation functions (list of length n_orders used)
    xcs = make_ccfs(f, wl, f0, wl0, mask=None, smoothed_flat=None, delta_log_wl=delta_log_wl, relgrid=False,
                    flipped=flipped, individual_fibres=individual_fibres, debug_level=debug_level, timit=timit)


    # now fit Gaussian to central section of CCF for that order
    if relgrid:
        fitrangesize = osf * 6  # this factor was simply eye-balled
    else:
        # fitrangesize = 30
        # fitrangesize = int(np.round(0.0036 * len(xc) / 2. - 1, 0))  # this factor was simply eye-balled
        fitrangesize = fitrange


    if individual_fibres:
        
        # make array only containing the central parts of the CCFs (which can have different total lengths) for fitting
        xcarr = np.zeros((len(xcs), len(xcs[0]), 2 * addrange + 1))
        for i in range(xcarr.shape[0]):
            for j in range(xcarr.shape[1]):
                dum = np.array(xcs[i][j])
                xcarr[i,j,:] = dum[len(dum) // 2 - addrange : len(dum) // 2 + addrange + 1]
        
        if individual_orders:
            if debug_level >= 1:
                print('Calculating independent RVs for each order and for each fibre...')
            xcsum = np.sum(np.sum(xcarr, axis=0), axis=0)
        else:
            if debug_level >= 1:
                print('Calculating one RV for each fibre (summing up CCFs over individual orders)...')
            # sum up the CCFs for all orders
            xcarr = np.sum(xcarr, axis=0)
            xcarr = xcarr[np.newaxis,:]   # need that extra dimension for the for-loop below
            xcsum = np.sum(xcarr, axis=0)
                
        # format is (n_ord, n_fib)
        rv = np.zeros((xcarr.shape[0], xcarr.shape[1]))
        rverr = np.zeros((xcarr.shape[0], xcarr.shape[1]))
        for o in range(xcarr.shape[0]):
            for f in range(xcarr.shape[1]):
                if debug_level >= 3:
                    print('order = ',o,' ; fibre = ',f)
                xc = xcarr[o,f,:]
                xrange = np.arange(np.argmax(xc) - fitrangesize, np.argmax(xc) + fitrangesize + 1, 1)
                # parameters: mu, sigma, amp, beta, offset, slope
                guess = np.array((np.argmax(xc), 15, np.max(xc) - np.min(xc), 2., np.min(xc), 0.))
                try:
                    popt, pcov = op.curve_fit(gausslike_with_amp_and_offset_and_slope, xrange, xc[xrange], p0=guess, maxfev=1000000)
                    mu = popt[0]
                    mu_err = pcov[0, 0]
                except:
                    popt, pcov = (np.nan, np.nan)
                    mu = np.nan
                    mu_err = np.nan
    
                # convert to RV in m/s
                rv[o,f] = c * (mu - (len(xc) // 2)) * delta_log_wl
                rverr[o,f] = c * mu_err * delta_log_wl
                
    else:
        
        # make array only containing the central parts of the CCFs (which can have different total lengths) for fitting
        xcarr = np.zeros((len(xcs), 2 * addrange + 1))
        for i in range(xcarr.shape[0]):
            dum = np.array(xcs[i])
            xcarr[i,:] = dum[len(dum) // 2 - addrange : len(dum) // 2 + addrange + 1]
        
        if individual_orders:
            if debug_level >= 1:
                print('Calculating one RV for each order (summing up CCFs over individual fibres)...')
        else:
            if debug_level >= 1:
                print('Calculating one RV (summing up CCFs over individual fibres and over individual orders)')
            # sum up the CCFs for all orders
            xcarr = np.sum(xcarr, axis=0)
            xcarr = xcarr[np.newaxis,:]   # need that extra dimension for the for-loop below
            
        xcsum = np.sum(xcarr, axis=0)
        rv = np.zeros(xcarr.shape[0])
        rverr = np.zeros(xcarr.shape[0])
        for o in range(xcarr.shape[0]):
            xc = xcarr[o,:]
            xrange = np.arange(np.argmax(xc) - fitrangesize, np.argmax(xc) + fitrangesize + 1, 1)
            # parameters: mu, sigma, amp, beta, offset, slope
            guess = np.array((np.argmax(xc), 10, (xc[np.argmax(xc)] - xc[np.argmax(xc) - fitrangesize]), 2.,
                              xc[np.argmax(xc) - fitrangesize], 0.))
            try:
                popt, pcov = op.curve_fit(gausslike_with_amp_and_offset_and_slope, xrange, xc[xrange], p0=guess, maxfev=1000000)
                mu = popt[0]
                mu_err = pcov[0, 0]
            except:
                popt, pcov = (np.nan, np.nan)
                mu = np.nan
                mu_err = np.nan
            

            # convert to RV in m/s
            rv[o] = c * (mu - (len(xc) // 2)) * delta_log_wl
            rverr[o] = c * mu_err * delta_log_wl
            # # plot a single fit for debugging
            # plot_osf = 10
            # plot_os_grid = np.linspace(xrange[0], xrange[-1], plot_osf * (len(xrange)-1) + 1)
            # plt.plot(c * (xrange - (len(xcsums) // 2)) * delta_log_wl, xcsum[xrange], 'b.', label='data')
            # plt.plot(c * (plot_os_grid - (len(xcsums) // 2)) * delta_log_wl, gausslike_with_amp_and_offset_and_slope(plot_os_grid, *guess),'r--', label='initial guess')
            # plt.plot(c * (plot_os_grid - (len(xcsums) // 2)) * delta_log_wl, gausslike_with_amp_and_offset_and_slope(plot_os_grid, *popt),'g-', label='best fit')
            # plt.axvline(c * (mu - (len(xcsum) // 2)) * delta_log_wl, color='g', linestyle=':')
            # plt.legend()
            # plt.xlabel('delta RV [m/s]')
            # plt.ylabel('power')
            # plt.title('CCF')

    if timit:
        delta_t = time.time() - start_time
        print('Time taken for calculating RV: ' + str(np.round(delta_t, 2)) + ' seconds')

    if individual_fibres:
        if individual_orders:
            return rv, rverr, np.array(xcsum)
        else:
            return rv, rverr, np.array(xcsum)
    else:
        if individual_orders:
            return rv, rverr, np.array(xcsum)
        else:
            return rv, rverr, np.array(xcsum)
    





def make_ccfs(f, wl, f0, wl0, mask=None, smoothed_flat=None, delta_log_wl=1e-6, relgrid=False, osf=5,
             filter_width=25, bad_threshold=0.05, flipped=False, individual_fibres=True, debug_level=0, timit=False):
    """
    This routine calculates the CCFs of an observed spectrum and a template spectrum for each order.
    Note that input spectra should be de-blazed already!!!
    If the mask from "find_stripes" has gaps, do the filtering for each segment independently. If no mask is provided, create a simple one on the fly.

    INPUT:
    'f'                  : numpy array containing the observed flux (n_ord, n_fib, n_pix)
    'wl'                 : numpy array containing the wavelengths of the observed spectrum (n_ord, n_fib, n_pix)
    'f0'                 : numpy array containing the flux of the template spectrum (n_ord, n_fib, n_pix)
    'wl0'                : numpy array containing the wavelengths of the template spectrum (n_ord, n_fib, n_pix)
    'mask'               : mask-dictionary from "find_stripes" (keys = orders)
    'smoothed_flat'      : if no mask is provided, need to provide the smoothed_flat, so that a mask can be created on the fly
    'delta_log_wl'       : stepsize of the log-wl grid (only used if 'relgrid' is FALSE)
    'relgrid'            : boolean - do you want to use an absolute stepsize of the log-wl grid (DEFAULT), or relative using 'osf'?
    'osf'                : oversampling factor for the logarithmic wavelength rebinning (only used if 'relgrid' is TRUE)
    'filter_width'       : width of smoothing filter in pixels; needed b/c of edge effects of the smoothing; number of pixels to disregard should be >~ 2 * width of smoothing kernel
    'bad_threshold'      : if no mask is provided, create a mask that requires the flux in the extracted white to be larger than this fraction of the maximum flux in that order
    'flipped'            : boolean - reverse order of inputs to xcorr routine?
    'individual_fibres'  : boolean - do you want to return the CCFs for individual fibres? (if FALSE, then the sum of the ind. fib. CCFs is returned)
    'debug_level'        : for debugging...
    'timit'              : boolean - do you want to measure execution run time?

    OUTPUT:
    'xcs'   : list containing the ind. fib. CCFs / sum of ind. fib. CCFs for each order (len = n_ord)

    MODHIST:
    Nov 2018 - CMB create
    """

    if timit:
        start_time = time.time()

    # the dummy wavelengths for orders 'order_01' and 'order_40' cannot be zero as we're taking a log!!!
    if wl.shape[0] == 40:
        wl[0, :, :] = 1.
        wl[-1, :, :] = 1.
        wl0[0, :, :] = 1.
        wl0[-1, :, :] = 1.
    if wl.shape[0] == 39:
        wl[0, :, :] = 1.
        wl0[0, :, :] = 1.

    xcs = []

    # loop over orders
    # for ord in sorted(f.iterkeys()):
    # for o in range(wl.shape[0]):
    # from Brendan's plots/table:
    # for o in [5, 6, 7, 17, 26, 27, 34, 35, 36, 37]:
    # Duncan's suggestion
    # for o in [4,5,6,25,26,33,34,35]:
    for o in [5, 6, 17, 25, 26, 27, 31, 34, 35, 36, 37]:
        
        if debug_level >= 2:
            print('Order ' + str(o+1).zfill(2))
            
        ord = 'order_' + str(o+1).zfill(2)    

        # # only use pixels that have enough signal
        # if mask is None:
        #     normflat = smoothed_flat[ord] / np.max(smoothed_flat[ord])
        #     ordmask = np.ones(len(normflat), dtype=bool)
        #     if np.min(normflat) < bad_threshold:
        #         ordmask[normflat < bad_threshold] = False
        #         # once the blaze function falls below a certain value, exclude what's outside of that pixel column, even if it's above the threshold again, ie we want to only use a single consecutive region
        #         leftmask = ordmask[: len(ordmask) // 2]
        #         leftkill_index = [i for i, x in enumerate(leftmask) if not x]
        #         try:
        #             ordmask[: leftkill_index[0]] = False
        #         except:
        #             pass
        #         rightmask = ordmask[len(ordmask) // 2:]
        #         rightkill_index = [i for i, x in enumerate(rightmask) if not x]
        #         if ord == 'order_01' and simu:
        #             try:
        #                 # if not flipped then the line below must go in the leftkillindex thingy
        #                 # ordmask[: leftkill_index[-1] + 100] = False
        #                 ordmask[len(ordmask) // 2 + rightkill_index[0] - 100:] = False
        #             except:
        #                 pass
        #         else:
        #             try:
        #                 ordmask[len(mask) // 2 + rightkill_index[-1] + 1:] = False
        #             except:
        #                 pass
        # else:
        #     # ordmask  = mask[ord][::-1]
        #     ordmask = mask[ord]

        ordmask = np.ones(4112, dtype=bool)
        ordmask[:200] = False
        ordmask[4000:] = False


        # either way, disregard #(edge_cut) pixels at either end; this is slightly dodgy, but the gaussian filtering above introduces edge effects due to mode='reflect'
        ordmask[:2 * int(filter_width)] = False
        ordmask[-2 * int(filter_width):] = False

        # f0_unblazed = f0[ord] / np.max(f0[ord]) / normflat   #see COMMENT above
        # f_unblazed = f[ord] / np.max(f[ord]) / normflat

        # create logarithmic wavelength grid
        logwl = np.log(wl[o,:,:])
        logwl0 = np.log(wl0[o,:,:])
        if relgrid:
            logwlgrid = np.linspace(np.min(logwl[ordmask]), np.max(logwl[ordmask]), osf * np.sum(ordmask))
            delta_log_wl = logwlgrid[1] - logwlgrid[0]
            if debug_level >= 2:
                print(ord, ' :  delta_log_wl = ', delta_log_wl)
        else:
            #logwlgrid = np.arange(np.min(logwl[-1,ordmask]), np.max(logwl[-1,ordmask]), delta_log_wl)
            # use range from the maximum of the indfib-minima to the minimum of the indfib-maxima (also across wl and wl0) - use list comprehensions...
            logwlgrid = np.arange(np.max([np.min(logwl[fib,ordmask]) for fib in range(logwl.shape[0])] + [np.min(logwl0[fib,ordmask]) for fib in range(logwl0.shape[0])]), 
                                  np.min([np.max(logwl[fib,ordmask]) for fib in range(logwl.shape[0])] + [np.max(logwl0[fib,ordmask]) for fib in range(logwl0.shape[0])]), 
                                  delta_log_wl)

        # wavelength array must be increasing for "InterpolatedUnivariateSpline" to work --> turn arrays around if necessary!!!
        if (np.diff(logwl) < 0).any():
            logwl_sorted = logwl[:, ::-1].copy()
            logwl0_sorted = logwl0[:,::-1].copy()
            ordmask_sorted = ordmask[::-1].copy()
            ord_f0_sorted = f0[o,:,::-1].copy()
            ord_f_sorted = f[o,:,::-1].copy()
        else:
            logwl_sorted = logwl.copy()
            logwl0_sorted = logwl0.copy()
            ordmask_sorted = ordmask.copy()
            ord_f0_sorted = f0[ord].copy()
            ord_f_sorted = f[ord].copy()

        # rebin spectra onto logarithmic wavelength grid
        # rebinned_f0 = np.interp(logwlgrid,logwl[mask],f0_unblazed[mask])
        # rebinned_f = np.interp(logwlgrid,logwl[mask],f_unblazed[mask])
        nfib = ord_f0_sorted.shape[0]
        rebinned_f0 = np.zeros((nfib, len(logwlgrid)))
        rebinned_f = np.zeros((nfib, len(logwlgrid)))
        for i in range(nfib):
            spl_ref_f0 = interp.InterpolatedUnivariateSpline(logwl0_sorted[i,ordmask_sorted], ord_f0_sorted[i,ordmask_sorted], k=3)  # slightly slower than linear, but best performance for cubic spline
            rebinned_f0[i,:] = spl_ref_f0(logwlgrid)
            spl_ref_f = interp.InterpolatedUnivariateSpline(logwl_sorted[i,ordmask_sorted], ord_f_sorted[i,ordmask_sorted], k=3)  # slightly slower than linear, but best performance for cubic spline
            rebinned_f[i,:] = spl_ref_f(logwlgrid)

        if individual_fibres:
            ord_xcs = []
            for fib in range(nfib):
                if not flipped:
                    xc = np.correlate(rebinned_f0[fib,:], rebinned_f[fib,:], mode='full')
                else:
                    xc = np.correlate(rebinned_f[fib,:], rebinned_f0[fib,:], mode='full')
                ord_xcs.append(xc)
            xcs.append(ord_xcs)
        else:
            rebinned_f = np.sum(rebinned_f, axis=0)
            rebinned_f0 = np.sum(rebinned_f0, axis=0)
            # xc = np.correlate(rebinned_f0 - np.median(rebinned_f0), rebinned_f - np.median(rebinned_f), mode='full')
            if not flipped:
                xc = np.correlate(rebinned_f0, rebinned_f, mode='full')
            else:
                xc = np.correlate(rebinned_f, rebinned_f0, mode='full')
            xcs.append(xc)

    if timit:
        delta_t = time.time() - start_time
        print('Time taken for creating CCFs: ' + str(np.round(delta_t, 2)) + ' seconds')

    return xcs



def make_self_indfib_ccfs(f, wl, relto=9, mask=None, smoothed_flat=None, delta_log_wl=1e-6, filter_width=25, bad_threshold=0.05, debug_level=0, timit=False):
    """
    This routine calculates the CCFs of all fibres with respect to one user-specified (default = central) fibre for a given observation.
    If the mask from "find_stripes" has gaps, do the filtering for each segment independently. If no mask is provided, create a simple one on the fly.

    INPUT:
    'f'                  : numpy array containing the observed flux (n_ord, n_fib, n_pix)
    'wl'                 : numpy array containing the wavelengths of the observed spectrum (n_ord, n_fib, n_pix)
    'relto'              : which fibre do you want to use as the reference fibre [0, 1, ... , 18]
    'mask'               : mask-dictionary from "find_stripes" (keys = orders)
    'smoothed_flat'      : if no mask is provided, need to provide the smoothed_flat, so that a mask can be created on the fly
    'delta_log_wl'       : stepsize of the log-wl grid (only used if 'relgrid' is FALSE)
    'filter_width'       : width of smoothing filter in pixels; needed b/c of edge effects of the smoothing; number of pixels to disregard should be >~ 2 * width of smoothing kernel
    'bad_threshold'      : if no mask is provided, create a mask that requires the flux in the extracted white to be larger than this fraction of the maximum flux in that order
    'debug_level'        : for debugging...
    'timit'              : boolean - do you want to measure execution run time?

    OUTPUT:
    'xcs'   : list containing the ind. fib. CCFs / sum of ind. fib. CCFs for each order (len = n_ord)

    MODHIST:
    Dec 2018 - CMB create (clone of "make_ccfs") 
    """

    assert relto in np.arange(19), 'reference fibre not recognized (must be in [0, 1, ... , 18])'

    if timit:
        start_time = time.time()

    # the dummy wavelengths for orders 'order_01' and 'order_40' cannot be zero as we're taking a log!!!
    if wl.shape[0] == 40:
        wl[0, :, :] = 1.
        wl[-1, :, :] = 1.
    if wl.shape[0] == 39:
        wl[0, :, :] = 1.

    xcs = []

    # loop over orders
    # for ord in sorted(f.iterkeys()):
    # for o in range(wl.shape[0]):
    # for o in [4,5,6,25,26,33,34,35]:
    for o in [5, 6, 17, 25, 26, 27, 31, 34, 35, 36, 37]:

        if debug_level >= 2:
            print('Order ' + str(o+1).zfill(2))

        # # only use pixels that have enough signal
        # if mask is None:
        #     normflat = smoothed_flat[ord] / np.max(smoothed_flat[ord])
        #     ordmask = np.ones(len(normflat), dtype=bool)
        #     if np.min(normflat) < bad_threshold:
        #         ordmask[normflat < bad_threshold] = False
        #         # once the blaze function falls below a certain value, exclude what's outside of that pixel column, even if it's above the threshold again, ie we want to only use a single consecutive region
        #         leftmask = ordmask[: len(ordmask) // 2]
        #         leftkill_index = [i for i, x in enumerate(leftmask) if not x]
        #         try:
        #             ordmask[: leftkill_index[0]] = False
        #         except:
        #             pass
        #         rightmask = ordmask[len(ordmask) // 2:]
        #         rightkill_index = [i for i, x in enumerate(rightmask) if not x]
        #         if ord == 'order_01' and simu:
        #             try:
        #                 # if not flipped then the line below must go in the leftkillindex thingy
        #                 # ordmask[: leftkill_index[-1] + 100] = False
        #                 ordmask[len(ordmask) // 2 + rightkill_index[0] - 100:] = False
        #             except:
        #                 pass
        #         else:
        #             try:
        #                 ordmask[len(mask) // 2 + rightkill_index[-1] + 1:] = False
        #             except:
        #                 pass
        # else:
        #     # ordmask  = mask[ord][::-1]
        #     ordmask = mask[ord]

        ordmask = np.ones(4112, dtype=bool)
        ordmask[:200] = False
        ordmask[4000:] = False


        # either way, disregard #(edge_cut) pixels at either end; this is slightly dodgy, but the gaussian filtering above introduces edge effects due to mode='reflect'
        ordmask[:2 * int(filter_width)] = False
        ordmask[-2 * int(filter_width):] = False

        # f0_unblazed = f0[ord] / np.max(f0[ord]) / normflat   #see COMMENT above
        # f_unblazed = f[ord] / np.max(f[ord]) / normflat

        # create logarithmic wavelength grid
        logwl = np.log(wl[o,:,:])
        # use range from the maximum of the indfib-minima to the minimum of the indfib-maxima - use list comprehensions...
        logwlgrid = np.arange(np.max([np.min(logwl[fib,ordmask]) for fib in range(logwl.shape[0])]), 
                              np.min([np.max(logwl[fib,ordmask]) for fib in range(logwl.shape[0])]), 
                              delta_log_wl)

        # wavelength array must be increasing for "InterpolatedUnivariateSpline" to work --> turn arrays around if necessary!!!
        if (np.diff(logwl) < 0).any():
            logwl_sorted = logwl[:, ::-1].copy()
            ordmask_sorted = ordmask[::-1].copy()
            ord_f_sorted = f[o,:,::-1].copy()
        else:
            logwl_sorted = logwl.copy()
            ordmask_sorted = ordmask.copy()
            ord_f_sorted = f[ord].copy()

        # rebin spectra onto logarithmic wavelength grid
        nfib = ord_f_sorted.shape[0]
        rebinned_f = np.zeros((nfib, len(logwlgrid)))
        for i in range(nfib):
            spl_ref_f = interp.InterpolatedUnivariateSpline(logwl_sorted[i,ordmask_sorted], ord_f_sorted[i,ordmask_sorted], k=3)  # slightly slower than linear, but best performance for cubic spline
            rebinned_f[i,:] = spl_ref_f(logwlgrid)

        
        ord_xcs = []
        for fib in range(nfib):
            xc = np.correlate(rebinned_f[fib,:], rebinned_f[relto,:], mode='full')
            ord_xcs.append(xc)
        xcs.append(ord_xcs)
        

    if timit:
        delta_t = time.time() - start_time
        print('Time taken for creating CCFs: ' + str(np.round(delta_t, 2)) + ' seconds')

    return xcs






##########################################################################################
### CMB - 15/11/2017                                                                   ###
### The following functions are taken from the RV parts from Mike Ireland's "pymfe"    ###
### I also made them stand-alone routines rather than part of an object-oriented class ###
##########################################################################################

def rv_shift_resid(parms, wave, spect, spect_sdev, spline_ref, return_spect=False):
    """Find the residuals to a fit of a (subsampled) reference spectrum to an 
    observed spectrum. 
    
    The function for parameters p[0] through p[3] is:
    
    .. math::
        y(x) = Ref[ wave(x) * (1 - p[0]/c) ] * exp(p[1] * x^2 + p[2] * x + p[3])
    
    Here "Ref" is a function f(wave)
    
    Parameters
    ----------        
    params: array-like
    wave: float array
        Wavelengths for the observed spectrum.        
    spect: float array
        The observed spectra     
    spect_sdev: float array
        standard deviation of the input spectra.        
    spline_ref: InterpolatedUnivariateSpline instance
        For interpolating the reference spectrum
    return_spect: boolean
        Whether to return the fitted spectrum or the residuals.
    wave_ref: float array
        The wavelengths of the reference spectrum
    ref: float array
        The reference spectrum
    
    Returns
    -------
    resid: float array
        The fit residuals
    """
    
    ny = len(spect)
    # CMB change: necessary to make xx go smoothly from -0.5 to 0.5, rather than a step function (step at ny//2) from -1.0 to 0.0      
    #xx = (np.arange(ny)-ny//2)/ny
    xx = (np.arange(ny)-ny//2)/float(ny)
    norm = np.exp(parms[1]*xx*xx + parms[2]*xx + parms[3])     #CMB change for speed (was *xx**2)
    # Lets get this sign correct. A redshift (positive velocity) means that
    # a given wavelength for the reference corresponds to a longer  
    # wavelength for the target, which in turn means that the target 
    # wavelength has to be interpolated onto shorter wavelengths for the 
    # reference.
    #fitted_spect = spline_ref(wave*(1.0 - parms[0]/const.c.si.value))*norm
    # CMB change: just declared c above
    fitted_spect = spline_ref(wave * (1.0 - parms[0]/c)) * norm
    
    if return_spect:
        return fitted_spect
    else:
        return (fitted_spect - spect)/spect_sdev



def rv_shift_chi2(parms, wave, spect, spect_sdev, spline_ref):
    """Find the chi-squared for an RV fit. Just a wrapper for rv_shift_resid,
    so the docstring is cut and paste!
    
    The function for parameters p[0] through p[3] is:
    
    .. math::
        y(x) = Ref[ wave(x) * (1 - p[0]/c) ] * exp(p[1] * x^2 + p[2] * x + p[3])
    
    Here "Ref" is a function f(wave)
     
    Parameters
    ----------
    
    params: 
        ...
    wave: float array
        Wavelengths for the observed spectrum.
    spect: float array
        The observed spectrum
    spect_sdev: 
        ...
    spline_ref: 
        ...
    return_spect: boolean
        Whether to return the fitted spectrum or the 
        
    wave_ref: float array
        The wavelengths of the reference spectrum
    ref: float array
        The reference spectrum
    
    Returns
    -------
    chi2:
        The fit chi-squared
    """
    return np.sum(rv_shift_resid(parms, wave, spect, spect_sdev, spline_ref)**2)



def rv_shift_jac(parms, wave, spect, spect_sdev, spline_ref):
    """Explicit Jacobian function for rv_shift_resid. 
    
    This is not a completely analytic solution, but without it there seems to be 
    numerical instability.
    
    The key equations are:
    
    .. math:: f(x) = R( \lambda(x)  (1 - p_0/c) ) \times \exp(p_1 x^2 + p_2 x + p_3)
    
       g(x) = (f(x) - d(x))/\sigma(x)
    
       \frac{dg}{dp_0}(x) \approx  [f(x + 1 m/s) -f(x) ]/\sigma(x)
    
       \frac{dg}{dp_1}(x) = x^2 f(x) / \sigma(x)
    
       \frac{dg}{dp_2}(x) = x f(x) / \sigma(x)
    
       \frac{dg}{dp_3}(x) = f(x) / \sigma(x)
    
    Parameters
    ----------
    
    params: float array
    wave: float array
        Wavelengths for the observed spectrum.
    spect: float array
        The observed spectrum
    spect_sdev: 
        ...
    spline_ref: 
        ...
        
    Returns
    -------
    jac: 
        The Jacobian.
    """
    
    ny = len(spect)
    # CMB change: necessary to make xx go smoothly from -0.5 to 0.5, rather than a step function (step at ny//2) from -1.0 to 0.0      
    #xx = (np.arange(ny)-ny//2)/ny
    xx = (np.arange(ny)-ny//2)/float(ny)
    norm = np.exp(parms[1]*xx*xx + parms[2]*xx + parms[3])     #CMB change for speed (was *xx**2)
    #fitted_spect = spline_ref(wave*(1.0 - parms[0]/const.c.si.value))*norm
    fitted_spect = spline_ref(wave * (1.0 - parms[0]/c)) * norm
    
    #The Jacobian is the derivative of fitted_spect/spect_sdev with respect to p[0] through p[3]
    jac = np.empty((ny,4))
    jac[:,3] = fitted_spect / spect_sdev
    jac[:,2] = fitted_spect*xx / spect_sdev
    jac[:,1] = fitted_spect*xx*xx / spect_sdev     #CMB change for speed (was *xx**2)
    #jac[:,0] = (spline_ref(wave*(1.0 - (parms[0] + 1.0)/const.c.si.value))*
    #            norm - fitted_spect)/spect_sdev
    jac[:,0] = ((spline_ref(wave * (1.0 - (parms[0] + 1.0)/c)) * norm) - fitted_spect) / spect_sdev
    
    return jac



def calculate_rv_shift(wave_ref, ref_spect, fluxes, vars, bcors, wave, return_fitted_spects=False, bad_threshold=10):
    """Calculates the Radial Velocity of each spectrum
    
    The radial velocity shift of the reference spectrum required
    to match the flux in each order in each input spectrum is calculated.
    
    The input fluxes to this method are flat-fielded data, which are then fitted with 
    a barycentrically corrected reference spectrum :math:`R(\lambda)`, according to 
    the following equation:

    .. math::
        f(x) = R( \lambda(x)  (1 - p_0/c) ) \\times \exp(p_1 x^2 + p_2 x + p_3)

    The first term in this equation is simply the velocity corrected spectrum, based on a 
    the arc-lamp derived reference wavelength scale :math:`\lambda(x)` for pixels coordinates x.
    The second term in the equation is a continuum normalisation - a shifted Gaussian was 
    chosen as a function that is non-zero everywhere. The scipy.optimize.leastsq function is used
    to find the best fitting set fof parameters :math:`p_0` through to :math`p_3`. 

    The reference spectrum function :math:`R(\lambda)` is created using a wavelength grid 
    which is over-sampled with respect to the data by a factor of 2. Individual fitted 
    wavelengths are then found by cubic spline interpolation on this :math:`R_j(\lambda_j)` 
    discrete grid.
    
    Parameters
    ----------
    wave_ref: 2D np.array(float)
        Wavelength coordinate map of form (Order, Wavelength/pixel*2+2), 
        where the wavelength scale has been interpolated.
    ref_spect: 2D np.array(float)
        Reference spectrum of form (Order, Flux/pixel*2+2), 
        where the flux scale has been interpolated.
    fluxes: 3D np.array(float)
        Fluxes of form (Observation, Order, Flux/pixel)
    vars: 3D np.array(float)
        Variance of form (Observation, Order, Variance/pixel)    
    bcors: 1D np.array(float)
        Barycentric correction for each observation.
    wave: 2D np.array(float)
        Wavelength coordinate map of form (Order, Wavelength/pixel)

    Returns
    -------
    rvs: 2D np.array(float)
        Radial velocities of format (Observation, Order)
    rv_sigs: 2D np.array(float)
        Radial velocity sigmas of format (Observation, Order)
    """
    nm = fluxes.shape[1]
    ny = fluxes.shape[2]
    nf = fluxes.shape[0]
    
    # initialise output arrays
    rvs = np.zeros( (nf,nm) )
    rv_sigs = np.zeros( (nf,nm) )
    initp = np.zeros(4)
    initp[3]=0.5
    initp[0]=0.0
    spect_sdev = np.sqrt(vars)
    fitted_spects = np.empty(fluxes.shape)
    
    #loop over all fibres(?)
    for i in range(nf):
        # Start with initial guess of no intrinsic RV for the target.
        initp[0] = -bcors[i] #!!! New Change 
        nbad=0
        #loop over all orders(?)
        for j in range(nm):
            # This is the *only* non-linear interpolation function that 
            # doesn't take forever
            spl_ref = interp.InterpolatedUnivariateSpline(wave_ref[j,::-1], ref_spect[j,::-1])
            args = (wave[j,:], fluxes[i,j,:], spect_sdev[i,j,:], spl_ref)
            
            # Remove edge effects in a slightly dodgy way. 
            # 20 pixels is about 30km/s. 
            args[2][:20] = np.inf
            args[2][-20:] = np.inf
            the_fit = op.leastsq(rv_shift_resid, initp, args=args, diag=[1e3,1,1,1],Dfun=rv_shift_jac, full_output=True)
            #the_fit = op.leastsq(self.rv_shift_resid, initp, args=args,diag=[1e3,1e-6,1e-3,1], full_output=True,epsfcn=1e-9)
            
            #The following line also doesn't work "out of the box".
            #the_fit = op.minimize(self.rv_shift_chi2,initp,args=args)
            #pdb.set_trace()
            #Remove bad points...
            resid = rv_shift_resid( the_fit[0], *args)
            wbad = np.where( np.abs(resid) > bad_threshold)[0]
            nbad += len(wbad)
            #15 bad pixels in a single order is *crazy*
            if len(wbad)>20:
                fitted_spect = rv_shift_resid(the_fit[0], *args, return_spect=True)
                plt.clf()
                plt.plot(args[0], args[1])
                plt.plot(args[0][wbad], args[1][wbad],'o')
                plt.plot(args[0], fitted_spect)
                plt.xlabel("Wavelength")
                plt.ylabel("Flux")
                #print("Lots of 'bad' pixels. Type c to continue if not a problem")
                #pdb.set_trace()

            args[2][wbad] = np.inf
            the_fit = op.leastsq(rv_shift_resid, initp, args=args, diag=[1e3,1,1,1], Dfun=rv_shift_jac, full_output=True)
            #the_fit = op.leastsq(self.rv_shift_resid, initp,args=args, diag=[1e3,1e-6,1e-3,1], full_output=True, epsfcn=1e-9)
            
            #Some outputs for testing
            fitted_spects[i,j] = rv_shift_resid(the_fit[0], *args, return_spect=True)
            #the_fit[0][0] is the RV shift
            if ( np.abs(the_fit[0][0] - bcors[i]) < 1e-4 ):
                #pdb.set_trace() #This shouldn't happen, and indicates a problem with the fit.
                pass
            #Save the fit and the uncertainty.
            rvs[i,j] = the_fit[0][0]
            try:
                rv_sigs[i,j] = np.sqrt(the_fit[1][0,0])
            except:
                rv_sigs[i,j] = np.NaN
        print("Done file {0:d}. Bad spectral pixels: {1:d}".format(i,nbad))
    if return_fitted_spects:
        return rvs, rv_sigs, fitted_spects
    else:
        return rvs, rv_sigs
 
#########################################################################################
#########################################################################################
#########################################################################################











