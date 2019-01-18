'''
Created on 29 Nov. 2017

@author: Christoph Bergmann
'''

import time
import numpy as np
from scipy import signal
from scipy import ndimage
import matplotlib.pyplot as plt
import scipy.optimize as op
import lmfit
from lmfit import Model
from lmfit.models import GaussianModel, LorentzianModel, VoigtModel, PseudoVoigtModel, MoffatModel, Pearson7Model, StudentsTModel, BreitWignerModel, LognormalModel
from lmfit.models import DampedOscillatorModel, DampedHarmonicOscillatorModel, ExponentialGaussianModel, SkewedGaussianModel, DonaichModel
from readcol import readcol 
import datetime
import astropy.io.fits as pyfits

from basic_reduction.helper_functions import fibmodel_with_amp, CMB_pure_gaussian, multi_fibmodel_with_amp, CMB_multi_gaussian, offset_pseudo_gausslike
from basic_reduction.helper_functions import fit_poly_surface_2D, single_sigma_clip, find_nearest, gaussian_with_offset_and_slope, fibmodel_with_amp_and_offset
# from veloce_reduction.utils.linelists import make_gaussmask_from_linelist
# from veloce_reduction.veloce_reduction.lfc_peaks import find_affine_transformation_matrix, divide_lfc_peaks_into_orders







def find_suitable_peaks(rawdata, thresh = 5000., bgthresh = 2000., maxthresh = None, gauss_filter_sigma=1., slope=1e-4,
                        clip_edges=True, remove_bg=False, return_masks=False, debug_level=0, timit=False):
    """
    This routine finds (rough, to within 1 pixel) peak locations in 1D data.
    Detection threshold, background threshold, maximum threshold, and smoothing kernel width can be provided as keyword parameters.

    INPUT:
    'rawdata'   : one-dimensional (raw) data array

    KEYWORD PARAMETERS:
    'thresh'              : minimum height of peak in the FILTERED data to be considered a good peak
    'bgthresh'            : minimum height of peak in the filtered data to be considered a real (but not good) peak
    'maxthresh'           : maximum height of peak in the filtered data to be included (ie you can exclude saturated peaks etc)
    'gauss_filter_sigma'  : width of the Gaussian smoothing kernel (set to 0 if you want no smoothing)
    'slope'               : value of the (tiny) slope that's added in order to break possible degeneracies between adjacent pixels
    'clip_edges'          : boolean - do you want to clip the first and last peak, just to be on the safe side and avoid edge effects?
    'remove_bg'           : boolean - do you want to run a crude background removal before identifying peaks?
    'return_masks'        : boolean - do you want to return the masks as well as the data arrays?
    'debug_level'         : for debugging...
    'timit'               : boolean - do you want to clock execution run-time?

    OUTPUT:
    'goodpeaks'   : 1D array containing the "good" peak locations (above thresh and below maxthresh)
    'mostpeaks'   : 1D array containing all "real" peak locations (above bg_thresh and below maxthresh)
    'allpeaks'    : 1D array containing ALL peak locations (ie all local maxima, except for edges)
    'first_mask'  : mask that masks out shallow noise peaks from allpeaks (ONLY IF 'return_masks' is set to TRUE)
    'second_mask' : mask that masks out saturated lines above maxthresh from mostpeaks (ONLY IF 'return_masks' is set to TRUE)
    'third_mask'  : mask that masks out lines below thresh from mostpeaks (ONLY IF 'return_masks' is set to TRUE)
                    ie: goodpeaks = allpeaks[first_mask][second_mask][third_mask]
    
    TODO:
    (1) make a relative threshold criterion
    """

    #this routine is extremely fast, no need to optimise for speed
    if timit:
        start_time = time.time()
    
    xx = np.arange(len(rawdata))
    
    #run a crude background removal first:
    if remove_bg:
        bgfit = ndimage.gaussian_filter(ndimage.minimum_filter(rawdata,size=50.), 10.)
    else:
        bgfit = np.zeros(len(rawdata))
    data = rawdata - bgfit
    
    #smooth data to make sure we are not finding noise peaks, and add tiny slope to make sure peaks are found even when pixel-values are like [...,3,6,18,41,41,21,11,4,...]
    filtered_data = ndimage.gaussian_filter(data.astype(np.float), gauss_filter_sigma) + xx*slope
    
    #find all local maxima in smoothed data (and exclude the leftmost and rightmost maxima to avoid nasty edge effects...)
    allpeaks = signal.argrelextrema(filtered_data, np.greater)[0]
    
    ### this alternative version of finding extrema is completely equivalent (except that the below version accepts first and last as extrema, which I do not want)
    #testix = np.r_[True, filtered_data[1:] > filtered_data[:-1]] & np.r_[filtered_data[:-1] > filtered_data[1:], True]
    #testpeaks = np.arange(len(xx))[testix]
    
    #exclude first and last peaks to avoid nasty edge effects
    if clip_edges:
        allpeaks = allpeaks[1:-1]
    
    #make first mask to determine which peaks from linelist to use in wavelength solution
    first_mask = np.ones(len(allpeaks), dtype='bool')
    
    #remove shallow noise peaks
    first_mask[filtered_data[allpeaks] < bgthresh] = False
    mostpeaks = allpeaks[first_mask]
    
    #make mask which we need later to determine which peaks from linelist to use in wavelength solution
    second_mask = np.ones(len(mostpeaks), dtype='bool')
    
    #remove saturated lines
    if maxthresh is not None:
        second_mask[filtered_data[mostpeaks] > maxthresh] = False
        mostpeaks = mostpeaks[second_mask]
     
    #make mask which we need later to determine which peaks from linelist to use in wavelength solution    
    third_mask = np.ones(len(mostpeaks), dtype='bool')
    
    #only select good peaks higher than a certain threshold
    third_mask[filtered_data[mostpeaks] < thresh] = False
    goodpeaks = mostpeaks[third_mask]           #ie goodpeaks = allpeaks[first_mask][second_mask][third_mask]
    
    #for testing and debugging...
    if debug_level >= 1:
        print('Total number of peaks found: '+str(len(allpeaks)))
        print('Number of peaks found that are higher than '+str(int(thresh))+' counts: '+str(len(goodpeaks)))
        plt.figure()
        plt.plot(data)
        plt.plot(filtered_data)
        plt.scatter(goodpeaks, data[goodpeaks], marker='x', color='r', s=40)
        plt.plot((0,len(data)),(bgthresh,bgthresh),'r--')
        plt.plot((0,len(data)),(thresh,thresh),'g--')
        plt.xlabel('pixel')
        plt.ylabel('counts')
        #plt.vlines(thar_pos_guess, 0, np.max(data))
        plt.show()
    
    if timit:
        delta_t = time.time() - start_time
        print('Time elapsed: '+str(round(delta_t,5))+' seconds')
    
    if return_masks:
        return goodpeaks, mostpeaks, allpeaks, first_mask, second_mask, third_mask
    else:
        return goodpeaks, mostpeaks, allpeaks





def fit_emission_lines(data, fitwidth=4, thresh = 5000., bgthresh = 2000., maxthresh = None, slope=1e-4, laser=False,
                       varbeta=True, offset=False, minsigma=0.2, maxsigma=np.inf, sigma_0=1., minamp=0., maxamp=np.inf,
                       minbeta=1., maxbeta=4., beta_0=2., return_all_pars=False, return_qualflag=False, verbose=False, timit=False):
    """
    This routine identifies and fits emission lines in a 1-dim spectrum (ie generally speaking it finds and fits peaks in a 1dim array), using "scipy.optimize.curve_fit".
    Detection threshold, background threshold, and maximum threshold can be provided as keyword parameters. Different models for the peak-like function can be selected.

    INPUT:
    'data'   : one-dimensional data array

    KEYWORD PARAMETERS:
    'fitwidth'         : range around the identified peaks to be used for the peak-fitting 
    'thresh'           : minimum height of peak in the FILTERED data to be considered a good peak
    'bgthresh'         : minimum height of peak in the FILTERED data to be considered a real (but not good) peak
    'maxthresh'        : maximum height of peak in the FILTERED data to be included (ie you can exclude saturated peaks etc)
    'slope'            : value of the (tiny) slope that's added in order to break possible degeneracies between adjacent pixels
    'laser'            : boolean - is this a LFC spectrum? (if set to true, there is no check for blended lines)
    'varbeta'          : boolean - if set to TRUE, use a Gauss-like function for fitting, if set to FALSE use a plain Gaussian
    'offset'           : boolean - do you want to fit an offset to each peak as well?
    'minsigma'         : lower threshold for allowed sigma values
    'maxsigma'         : upper threshold for allowed sigma values
    'sigma_0'          : initial guess for sigma
    'minamp'           : lower threshold for allowed amplitude values
    'maxamp'           : upper threshold for allowed amplitude values
    'minbeta'          : lower threshold for allowed beta values
    'maxbeta'          : upper threshold for allowed beta values
    'beta_0'           : initial guess for beta
    'return_all_pars'  : boolean - do you want to return all fit parameters?
    'return_qualflag'  : boolean - do you want to return a quality flag for each line fit?
    'verbose'          : boolean - for debugging...
    'timit'            : boolean - do you want to clock execution run-time?    
    
    OUTPUT:
    'line_pos_fitted'    : fitted line positions
    'line_sigma_fitted'  : fitted line sigmas
    'line_amp_fitted'    : fitted line amplitudes
    'line_beta_fitted'   : fitted line betas (exponent in Gauss-like functions, which is equal to 2 for pure Gaussians)
    'qualflag'           : quality flag (1=good, 0=bad) for each line (only if 'return_qualflag' is set to TRUE)
    
    TODO:
    (1) make a relative threshold criterion
    """
    
    print('ATTENTION: You should consider using the more up-to-date and more robust function "fit_emission_lines_lmfit" instead!')
#     choice = None
#     while choice is None:
#         choice = raw_input("Do you want to continue? [y/n]: ")
#         if choice.lower() not in ('y','n'):
#             print('Invalid input! Please try again...')
#             choice = None
#    
#     #stop here if you'd rather use "fit_emission_lines_lmfit"
#     if choice.lower() == 'n':
#         return
    
    #OK, continue, you have been warned...
    if timit:
        start_time = time.time()
    
    xx = np.arange(len(data))
    
    #find rough peak locations
    goodpeaks,mostpeaks,allpeaks = find_suitable_peaks(data, thresh=thresh, bgthresh=bgthresh, maxthresh=maxthresh, slope=slope)
    
    if verbose:
        print('Fitting '+str(len(goodpeaks))+' emission lines...')
    
    line_pos_fitted = []
    if return_all_pars:
        line_amp_fitted = []
        line_sigma_fitted = []
        if varbeta:
            line_beta_fitted = []
    if return_qualflag:
        qualflag = []
        
    for xguess in goodpeaks:
        if verbose:
            print('xguess = ',xguess)
        ################################################################################################################################################################################
        #METHOD 1 (using curve_fit; slightly faster than method 2, but IDK how to make sure the fit converged (as with .ier below))

        if not laser:
            #check if there are any other peaks in the vicinity of the peak in question (exclude the peak itself)
            checkrange = np.r_[xx[np.max([0,xguess - 2*fitwidth]) : xguess], xx[xguess+1 : np.min([xguess + 2*fitwidth+1, len(data)-1])]]
            peaks = np.r_[xguess]
            #while len((set(checkrange) & set(allpeaks))) > 0:    THE RESULTS ARE COMPARABLE, BUT USING MOSTPEAKS IS MUCH FASTER
            while len((set(checkrange) & set(mostpeaks))) > 0:
                #where are the other peaks?
                #other_peaks = np.intersect1d(checkrange, allpeaks)    THE RESULTS ARE COMPARABLE, BUT USING MOSTPEAKS IS MUCH FASTER
                other_peaks = np.intersect1d(checkrange, mostpeaks)
                peaks = np.sort(np.r_[peaks, other_peaks])
                #define new checkrange
                checkrange = xx[np.max([0,peaks[0] - 2*fitwidth]) : np.min([peaks[-1] + 2*fitwidth + 1,len(data)-1])]
                dum = np.in1d(checkrange, peaks)
                checkrange = checkrange[~dum]
        else:
            peaks = np.r_[xguess]

        npeaks = len(peaks)
        xrange = xx[np.max([0,peaks[0] - fitwidth]) : np.min([peaks[-1] + fitwidth + 1,len(data)-1])]      # this should satisfy: len(xrange) == len(checkrange) - 2*fitwidth + len(peaks)

        if npeaks == 1:
            if offset:
                if varbeta:
                    guess = np.array([xguess, sigma_0, data[xguess], beta_0, np.min(data[xrange])])
                    popt, pcov = op.curve_fit(fibmodel_with_amp_and_offset, xrange, data[xrange], p0=guess, bounds=([xguess-2,minsigma,minamp,minbeta,-np.inf],[xguess+2,maxsigma,maxamp,maxbeta,np.inf]))
                else:
                    guess = np.array([xguess, sigma_0, data[xguess], np.min(data[xrange])])
                    popt, pcov = op.curve_fit(gaussian_with_offset, xrange, data[xrange], p0=guess, bounds=([xguess-2,minsigma,minamp,-np.inf],[xguess+2,maxsigma,maxamp,np.inf]))
            else:
                if varbeta:
                    guess = np.array([xguess, sigma_0, data[xguess], beta_0])
                    popt, pcov = op.curve_fit(fibmodel_with_amp, xrange, data[xrange], p0=guess, bounds=([xguess-2,minsigma,minamp,minbeta],[xguess+2,maxsigma,maxamp,maxbeta]))
                else:
                    guess = np.array([xguess, sigma_0, data[xguess]])
                    popt, pcov = op.curve_fit(CMB_pure_gaussian, xrange, data[xrange], p0=guess, bounds=([xguess-2,minsigma,minamp],[xguess+2,maxsigma,maxamp]))
            fitted_pos = popt[0]
            if return_all_pars:
                fitted_sigma = popt[1]
                fitted_amp = popt[2]
                if varbeta:
                    fitted_beta = popt[3]
        else:
            guess = []
            lower_bounds = []
            upper_bounds = []
            for i in range(npeaks):
                if varbeta:
                    guess.append(np.array([peaks[i], sigma_0, data[peaks[i]], beta_0]))
                    lower_bounds.append([peaks[i]-2,minsigma,minamp,minbeta])
                    upper_bounds.append([peaks[i]+2,maxsigma,maxamp,maxbeta])
                else:
                    guess.append(np.array([peaks[i], sigma_0, data[peaks[i]]]))
                    lower_bounds.append([peaks[i]-2,minsigma,minamp])
                    upper_bounds.append([peaks[i]+2,maxsigma,maxamp])
            guess = np.array(guess).flatten()
            lower_bounds = np.array(lower_bounds).flatten()
            upper_bounds = np.array(upper_bounds).flatten()
            if varbeta:
                popt, pcov = op.curve_fit(multi_fibmodel_with_amp, xrange, data[xrange], p0=guess, bounds=(lower_bounds,upper_bounds))
            else:
                popt, pcov = op.curve_fit(CMB_multi_gaussian, xrange, data[xrange], p0=guess, bounds=(lower_bounds,upper_bounds))

            #now figure out which peak is the one we wanted originally
            q = np.argwhere(peaks==xguess)[0]
            if varbeta:
                fitted_pos = popt[q*4]
                if return_all_pars:
                    fitted_sigma = popt[q*4+1]
                    fitted_amp = popt[q*4+2]
                    fitted_beta = popt[q*4+3]
            else:
                fitted_pos = popt[q*3]
                if return_all_pars:
                    fitted_sigma = popt[q*3+1]
                    fitted_amp = popt[q*3+2]


        #make sure we actually found a good peak
        if abs(fitted_pos - xguess) >= 2.:
            line_pos_fitted.append(xguess)
            if return_qualflag:
                qualflag.append(0)
            if return_all_pars:
                line_sigma_fitted.append(fitted_sigma)
                line_amp_fitted.append(fitted_amp)
                if varbeta:
                    line_beta_fitted.append(fitted_beta)
        else:
            line_pos_fitted.append(fitted_pos)
            if return_qualflag:
                qualflag.append(1)
            if return_all_pars:
                line_sigma_fitted.append(fitted_sigma)
                line_amp_fitted.append(fitted_amp)
                if varbeta:
                    line_beta_fitted.append(fitted_beta)
        ################################################################################################################################################################################
        
#         ################################################################################################################################################################################
#         #METHOD 2 (using lmfit) (NOTE THAT THE TWO METHODS HAVE different amplitudes for the Gaussian b/c of different normalization, but we are only interested in the position)
#         #xguess = int(xguess)
#         gm = GaussianModel()
#         gm_pars = gm.guess(data[xguess - fitwidth:xguess + fitwidth], xx[xguess - fitwidth:xguess + fitwidth])
#         gm_fit_result = gm.fit(data[xguess - fitwidth:xguess + fitwidth], gm_pars, x=xx[xguess - fitwidth:xguess + fitwidth])
#         
#         #make sure we actually found the correct peak
#         if gm_fit_result.ier not in (1,2,3,4):     #if this is any other value it means the fit did not converge
#         #if gm_fit_result.ier > 4:   
#             # gm_fit_result.plot()
#             # plt.show()
#             thar_pos_fitted.append(xguess)
#         elif abs(gm_fit_result.best_values['center'] - xguess) > 2.:
#             thar_pos_fitted.append(xguess)
#         else:
#             thar_pos_fitted.append(gm_fit_result.best_values['center'])
#         ################################################################################################################################################################################
    
    if verbose:    
        plt.figure()
        plt.plot(xx,data)
        #plt.vlines(thar_pos_guess, 0, np.max(data))
        plt.vlines(line_pos_fitted, 0, np.max(data) * 1.2, color='g', linestyles='dotted')
        plt.show()
    
    if timit:
        print('Time taken for fitting emission lines: '+str(time.time() - start_time)+' seconds...')
    
    if return_all_pars:
        if varbeta:
            if return_qualflag:
                return np.array(line_pos_fitted), np.array(line_sigma_fitted), np.array(line_amp_fitted), np.array(line_beta_fitted), np.array(qualflag)
            else:
                return np.array(line_pos_fitted), np.array(line_sigma_fitted), np.array(line_amp_fitted), np.array(line_beta_fitted)
        else:
            if return_qualflag:
                return np.array(line_pos_fitted), np.array(line_sigma_fitted), np.array(line_amp_fitted), np.array(qualflag)
            else:
                return np.array(line_pos_fitted), np.array(line_sigma_fitted), np.array(line_amp_fitted)
    else:
        if return_qualflag:
            return np.array(line_pos_fitted), np.array(qualflag)
        else:
            return np.array(line_pos_fitted)





def fit_emission_lines_lmfit(data, fitwidth=None, thresh=5000., bgthresh=2000., maxthresh=None, laser=False, model='gauss', no_weights=False,
                             offset=False, return_all_pars=False, return_qualflag=False, return_stats=False, return_full=False, timit=False, debug_level=0):
    """
    This routine identifies and fits emission lines in a 1-dim spectrum (ie generally speaking it finds and fits peaks in a 1dim array), using the LMFIT package.
    Detection threshold, background threshold, and maximum threshold can be provided as keyword parameters. Different models for the peak-like function can be selected.

    INPUT:
    'data'   : one-dimensional data array

    KEYWORD PARAMETERS:
    'fitwidth'         : range around the identified peaks to be used for the peak-fitting 
                         (if None, it will automatically be determined by the minimum spacing between peaks - this option should only be used for LFC spectra, won't work properly for ThAr or ThXe)
    'thresh'           : minimum height of peak in the FILTERED data to be considered a good peak
    'bgthresh'         : minimum height of peak in the FILTERED data to be considered a real (but not good) peak
    'maxthresh'        : maximum height of peak in the FILTERED data to be included (ie you can exclude saturated peaks etc)
    'laser'            : boolean - is this a LFC spectrum? (if set to true, there is no check for blended lines)
    'model'            : peak-like function model to be used for the fitting; valid options are (check code below for exact names):
                         [gaussian, gauss-like, double gaussian, lorentzian, voigt, pseudo-voigt, offset pseudo-voigt, offset "pseudo-gausslike", 
                         moffat, pearson7, students, breit-wigner, lognormal, damped oscillator, damped harmonic oscillator, exp gauss, skew gauss, donaich] 
    'no_weights'       : boolean - set to TRUE if you DO NOT WANT to use weights (not recommended)                        
    'offset'           : boolean - do you want to estimate and subtract an offset before fitting?                        
    'return_all_pars'  : boolean - do you want to return all fit parameters?
    'return_qualflag'  : boolean - do you want to return a quality flag for each line fit?
    'return_stats'     : boolean - do you want to return goodness of fit statistics (ie AIC, BIC, CHISQ and REDCHISQ)?
    'return_full'      : boolean - do you want to return the full "fit_result" class for each line? (WARNING: this ONLY returns the line positions and the full 'fit_result' classes,
                         ie it overwrites the 'return_all_pars' / 'return_qualflag' / 'return_stats' keywords)
    'debug_level'      : boolean - for debugging...
    'timit'            : boolean - do you want to clock execution run-time?    
    
    OUTPUT:
    'full_results'     : the full results, ie the instance of the LMFIT 'ModelResult' class (if 'return_full' is set to TRUE)
    -------------------
    'line_pos_fitted'  : fitted line positions
    'allpars'          : (only if 'return_all_pars' is set to TRUE)
    'qualflag'         : quality flag (1=good, 0=bad) for each line (only if 'return_qualflag' is set to TRUE)
    'stats'            : (only if 'return_stats' is set to TRUE)
    
    TODO:
    (1) make a relative threshold criterion
    """

    if timit:
        start_time = time.time()

    xx = np.arange(len(data))

    if model.lower() == 'gausslike':
        mod = Model(fibmodel_with_amp)
    if model.lower() in ('double', 'dbl', 'dg', 'dbl_gauss', 'double_gaussian'):
        mod1 = GaussianModel(prefix='first_')
        mod2 = GaussianModel(prefix='second_')
        mod = mod1 + mod2
    if model.lower() in ('gauss', 'gaussian'):
        mod = GaussianModel()
    if model.lower() in ('lorentz', 'lorentzian'):
        mod = LorentzianModel()
    if model.lower() == 'voigt':
        mod = VoigtModel()
    if model.lower() in ('pseudo', 'pseudovoigt'):
        mod = PseudoVoigtModel()
    if model.lower() == 'offset_pseudo':
        gmod = GaussianModel(prefix='G_')
        lmod = LorentzianModel(prefix='L_')
        mod = gmod + lmod
    if model.lower() == 'offset_pseudo_gausslike':
        mod = Model(offset_pseudo_gausslike)
    if model.lower() == 'moffat':
        mod = MoffatModel()
    if model.lower() in ('pearson', 'pearson7'):
        mod = Pearson7Model(nan_policy='omit')
    if model.lower() in ('student', 'students', 'studentst'):
        mod = StudentsTModel()
    if model.lower() == 'breitwigner':
        mod = BreitWignerModel()
    if model.lower() == 'lognormal':
        mod = LognormalModel()
    if model.lower() == 'dampedosc':
        mod = DampedOscillatorModel()
    if model.lower() == 'dampedharmosc':
        mod = DampedHarmonicOscillatorModel()
    if model.lower() == 'expgauss':
        mod = ExponentialGaussianModel(nan_policy='omit')
    if model.lower() == 'skewgauss':
        mod = SkewedGaussianModel()
    if model.lower() == 'donaich':
        mod = DonaichModel()


    # find rough peak locations
    goodpeaks, mostpeaks, allpeaks = find_suitable_peaks(data, thresh=thresh, bgthresh=bgthresh, maxthresh=maxthresh, remove_bg=True)

    #determine fit-window if not explicitly given
    if fitwidth is None:
        fitwidth = np.min(np.diff(goodpeaks))//2
    
    print('Fitting ' + str(len(goodpeaks)) + ' emission lines...')

    #prepare output variables
    line_pos_fitted = []
    if return_full:
        full_results = []
    else:
        if return_qualflag:
            qualflag = []
        if return_all_pars:
            allpars = []
        if return_stats:
            stats = []
    
    
    #loop over all lines
    for xguess in goodpeaks:
        
        if debug_level >= 2:
            print('xguess = ',xguess)

        if not laser:
            # check if there are any other peaks in the vicinity of the peak in question (exclude the peak itself)
            checkrange = np.r_[xx[np.max([0, xguess - 2 * fitwidth]): xguess], xx[xguess + 1: np.min([xguess + 2 * fitwidth + 1, len(data) - 1])]]
            peaks = np.r_[xguess]
            # while len((set(checkrange) & set(allpeaks))) > 0:    THE RESULTS ARE COMPARABLE, BUT USING MOSTPEAKS IS MUCH FASTER
            while len((set(checkrange) & set(mostpeaks))) > 0:
                # where are the other peaks?
                # other_peaks = np.intersect1d(checkrange, allpeaks)    THE RESULTS ARE COMPARABLE, BUT USING MOSTPEAKS IS MUCH FASTER
                other_peaks = np.intersect1d(checkrange, mostpeaks)
                peaks = np.sort(np.r_[peaks, other_peaks])
                # define new checkrange
                checkrange = xx[np.max([0, peaks[0] - 2 * fitwidth]): np.min([peaks[-1] + 2 * fitwidth + 1, len(data) - 1])]
                dum = np.in1d(checkrange, peaks)
                checkrange = checkrange[~dum]
        else:
            peaks = np.r_[xguess]

        npeaks = len(peaks)
        xrange = xx[np.max([0, peaks[0] - fitwidth]): np.min([peaks[-1] + fitwidth + 1, len(data) - 1])]   # this should satisfy: len(xrange) == len(checkrange) - 2*fitwidth + len(peaks)


        ################################################################################################################################################################################
        #using LMFIT package:

        if npeaks == 1:

            #estimate and subtract an offset if desired
            if offset:
                off = np.mean([data[xrange][0],data[xrange][1],data[xrange][-1],data[xrange][-2]])
            else:
                off = 0.
            fitdata = data[xrange] - off

            # create instance of Parameters-class needed for fitting with LMFIT
            if model.lower() in ('gausslike', 'offset_pseudo', 'offset_pseudo_gausslike', 'double', 'dbl', 'dg', 'dbl_gauss', 'double_gaussian'):
                parms = lmfit.Parameters()
            else:
                parms = mod.guess(fitdata, x=xrange)

            # fill/tweak initial Parameters() instance
            if model.lower() in ('gauss', 'gaussian'):
                parms['amplitude'].set(min=0.)
                parms['sigma'].set(min=0.)
            if model.lower() in ('double', 'dbl', 'dg', 'dbl_gauss', 'double_gaussian'):
                parms = mod1.guess(fitdata, x=xrange)
                parms.update(mod2.guess(fitdata, x=xrange))
                parms['second_amplitude'].set(0., vary=True)
                parms['first_center'].set(min=xguess - 3, max=xguess + 3)
                parms['second_center'].set(min=xguess - 3, max=xguess + 3)
            if model.lower() == 'gausslike':
                gmod = GaussianModel()
                gparms = gmod.guess(fitdata,xrange)
                parms.add('mu', xguess, min=xguess - 3, max=xguess + 3)
                parms.add('sigma', gparms['sigma'].value, min=0.2)
                parms.add('amp', data[xguess], min=0.)
                parms.add('beta', 2., min=1., max=4.)
            if model.lower() == 'moffat':
                parms['amplitude'].set(min=0.)
                parms['sigma'].set(min=0.)
                parms['beta'].set(min=0.)
            if model.lower() in ('pseudo', 'pseudovoigt'):
                # parms['fraction'].set(0.5,min=0.,max=1.)
                parms['amplitude'].set(min=0.)
                parms['sigma'].set(min=0.)
            if model.lower() == 'lognormal':
                #parms['sigma'].set(value=1e-4, vary=True, expr='')
                parms['center'].set(value=np.log(xguess), vary=True, expr='', min=0., max=8.5)
                parms['amplitude'].set(data[xguess], vary=True, min=0., expr='')
            # if model.lower() == 'dampedosc':
            #     parms['sigma'].set(1e-4, vary=True, expr='')
            #     parms['amplitude'].set(1e-4, vary=True, min=0., expr='')
            if model.lower() == 'offset_pseudo':
                parms = gmod.guess(fitdata, x=xrange)
                parms.update(lmod.guess(fitdata, x=xrange))
                parms['G_amplitude'].set(parms['G_amplitude'] / 2., min=0., vary=True)
                parms['L_amplitude'].set(parms['L_amplitude'] / 2., min=0., vary=True)
                parms['G_center'].set(min=xguess - 3, max=xguess + 3)
                parms['L_center'].set(min=xguess - 3, max=xguess + 3)
            if model.lower() == 'skewgauss':
                parms['amplitude'].set(min=0.)
                parms['sigma'].set(min=0.)
            if model.lower() == 'offset_pseudo_gausslike':
                gmod = GaussianModel(prefix='G_')
                lmod = LorentzianModel(prefix='L_')
                parms = gmod.guess(fitdata, x=xrange)
                parms.update(lmod.guess(fitdata, x=xrange))
                parms['G_amplitude'].set(parms['G_amplitude'] / 2., min=0., vary=True)
                parms['L_amplitude'].set(parms['L_amplitude'] / 2., min=0., vary=True)
                parms['G_center'].set(min=xguess - 3, max=xguess + 3)
                parms['L_center'].set(min=xguess - 3, max=xguess + 3)
                parms.add('beta', 2., min=1., max=4.)

            
            #perform the actual fit (get weights from array before offset subtraction)
            if not no_weights:
                fit_result = mod.fit(fitdata, parms, x=xrange, weights=np.sqrt(data[xrange]))
            else:
                fit_result = mod.fit(fitdata, parms, x=xrange)

            #goodness of fit statistics
            fitstats = {}
            fitstats['aic'] = fit_result.aic
            fitstats['bic'] = fit_result.bic
            fitstats['chi2'] = fit_result.chisqr
            fitstats['chi2red'] = fit_result.redchi 

            #fill bestpos variable depending on model used
            if model.lower() not in ('offset_pseudo', 'offset_pseudo_gausslike', 'double', 'dbl', 'dg', 'dbl_gauss', 'double_gaussian'):
                try:
                    bestpos = fit_result.best_values['center']
                except:
                    bestpos = fit_result.best_values['mu']
            elif model.lower() in ('offset_pseudo', 'offset_pseudo_gausslike'):
                bestpos = np.array((fit_result.best_values['G_center'], fit_result.best_values['L_center']))
            elif model.lower() in ('double', 'dbl', 'dg', 'dbl_gauss', 'double_gaussian'):
                bestpos = np.array((fit_result.best_values['first_center'], fit_result.best_values['second_center']))
            else:
                print('ERROR!!! No valid model defined!!!')
                return

            #make sure we actually found a good / the correct peak
            if fit_result.ier not in (1,2,3,4):     #if this is any other value it means the fit did not converge
            #if fit_result.ier > 4:
                fit_result.plot()
                plt.show()
                print('WARNING: Bad fit encountered!')
                choice = raw_input('Do you wish to continue? [y/n]')
                #choice = 'y'
                if choice.lower() == 'y':
                    line_pos_fitted.append(xguess)
                    if return_full:
                        full_results.append(fit_result)
                    else:
                        if return_qualflag:
                            qualflag.append(0)
                        if return_all_pars:
                            allpars.append(fit_result.params)
                        if return_stats:
                            stats.append(fitstats)    
                else:
                    return
            elif np.all(abs(bestpos - xguess) >= 2.):
                fit_result.plot()
                plt.show()
                print('WARNING: Bad fit encountered!')
                choice = raw_input('Do you wish to continue? [y/n]')
                #choice = y
                if choice.lower() == 'y':
                    line_pos_fitted.append(xguess)
                    if return_full:
                        full_results.append(fit_result)
                    else:
                        if return_qualflag:
                            qualflag.append(0)
                        if return_all_pars:
                            allpars.append(fit_result.params)
                        if return_stats:
                            stats.append(fitstats)
                else:
                    return
            else:
                line_pos_fitted.append(bestpos)
                if return_full:
                    full_results.append(fit_result)
                else:
                    if return_qualflag:
                        qualflag.append(0)
                    if return_all_pars:
                        allpars.append(fit_result.params)
                    if return_stats:
                        stats.append(fitstats)

# #             # # #this way you can quickly plot the indidvidual fits for debugging
#             plot_osf = 10
#             plot_os_grid = np.linspace(xrange[0], xrange[-1], plot_osf * (len(xrange) - 1) + 1)
#             guessmodel = mod.eval(fit_result.init_params, x=plot_os_grid)
#             bestmodel = mod.eval(fit_result.params, x=plot_os_grid)
#             plt.figure()
#             plt.title('model = ' + model.title())
#             plt.xlabel('pixel number (dispersion direction)')
#             plt.plot(xrange, fitdata, 'bo')
#             plt.plot(plot_os_grid, guessmodel, 'k--', label='initial guess')
#             plt.plot(plot_os_grid, bestmodel, 'r-', label='best-fit model')
#             plt.legend()


        else:
            print('ERROR: multi-peak fitting with LMFIT has not been implemented yet')
            return
        ################################################################################################################################################################################

    if debug_level >= 1:
        plt.figure()
        plt.plot(xx, data)
        # plt.vlines(thar_pos_guess, 0, np.max(data))
        plt.vlines(line_pos_fitted, 0, np.max(data) * 1.2, color='g', linestyles='dotted')
        plt.show()

    if timit:
        print('Time taken for fitting emission lines: ' + str(time.time() - start_time) + ' seconds...')

    #return results
    if return_full:
        return full_results 
    else:
        if return_stats:
            if return_all_pars:
                if return_qualflag:
                    return np.array(line_pos_fitted), allpars, np.array(qualflag), stats
                else:
                    return np.array(line_pos_fitted), allpars, stats
            else:
                if return_qualflag:
                    return np.array(line_pos_fitted), np.array(qualflag), stats
                else:
                    return np.array(line_pos_fitted), stats
        else:
            if return_all_pars:
                if return_qualflag:
                    return np.array(line_pos_fitted), allpars, np.array(qualflag)
                else:
                    return np.array(line_pos_fitted), allpars
            else:
                if return_qualflag:
                    return np.array(line_pos_fitted), np.array(qualflag)
                else:
                    return np.array(line_pos_fitted)





def get_wavelength_solution_from_thorium(thflux, poly_deg=5, polytype='chebyshev', savetable=False, return_full=True, timit=False):
    """ 
    INPUT:
    'thflux'           : extracted 1-dim thorium / laser-only image (initial tests used this one : '/Users/christoph/OneDrive - UNSW/veloce_spectra/reduced/tests/...')
    'poly_deg'         : the order of the polynomials to use in the fit (for both dimensions)
    'polytype'         : either 'polynomial', 'legendre', or 'chebyshev' (default) are accepted 
    'return_full'      : boolean - if TRUE, then the wavelength solution for each pixel for each order is returned; otherwise just the set of coefficients that describe it
    'saveplots'        : boolean - do you want to create plots for each order? 
    'savetable'        : boolean - if TRUE, then an output file is created, containing a summary of all lines used in the fit, and details about the fit
    'timit'            : time it...
    
    OUTPUT:
    'p'      : functional form of the coefficients that describe the wavelength solution
    'p_wl'   : wavelength solution for each pixel for each order (n_ord x n_pix numpy-array) 
    
    TODO:
    include order 40 (m=65) as well (just 2 lines!?!?!?)
    figure out how to properly use weights here (ie what should we use as weights?)
    """
    
    if timit:
        start_time = time.time()
    
#     #read in pre-defined thresholds (needed for line identification)
#     thresholds = np.load('/Users/christoph/OneDrive - UNSW/linelists/AAT_folder/thresholds.npy').item()
    
    #prepare arrays for fitting
    x = []
    order = []
    m_order = []
    wl = []

    for ord in sorted(thflux.keys()):    #make sure there are enough lines in every order
    #for ord in sorted(thflux.keys())[:-1]:    #don't have enough emission lines in order 40
        ordnum = ord[-2:]
        #m = 105 - int(ordnum)   #original orientation of the spectra
        m = 64 + int(ordnum)    #'correct' orientation of the spectra
        print('OK, fitting '+ord+'   (m = '+str(m)+')')
        
        data = thflux[ord]        
        xx = np.arange(len(data))
        
        #adjust thresholds here
        fitted_line_pos = fit_emission_lines(data,return_all_pars=False,varbeta=False,timit=False,verbose=False,thresh=thresholds['thresh'][ord],bgthresh=thresholds['bgthresh'][ord],maxthresh=thresholds['maxthresh'][ord])
        #fitted_line_pos = fit_emission_lines_lmfit(data, fitwidth=4, model='gausslike', return_all_pars=False,thresh=thresholds['thresh'][ord],
        #                                          bgthresh=thresholds['bgthresh'][ord],maxthresh=thresholds['maxthresh'][ord]) 
        goodpeaks,mostpeaks,allpeaks = find_suitable_peaks(data,thresh=thresholds['thresh'][ord],bgthresh=thresholds['bgthresh'][ord],maxthresh=thresholds['maxthresh'][ord])    
        
        #########################################################################################################################################
        #THIS IS THE STEP THAT NEEDS TO BE AUTOMATED
        line_number, refwlord = readcol('/Users/christoph/OneDrive - UNSW/linelists/AAT_folder/ThAr_linelist_order_'+ordnum+'.dat',fsep=';',twod=False)        
        #this tells us which peaks we have known wavelengths for
        mask_order = np.load('/Users/christoph/OneDrive - UNSW/linelists/posmasks/mask_order'+ordnum+'.npy')
        xord = fitted_line_pos[mask_order]
        
#         #ie we want to do sth like this:
#         guess_wls = get_wavelengths_for_peaks(fitted_line_pos)   # ie get a rough guess for thew wavelengths of the lines we found
#         refwlord = identify_lines(guess_wls, linelist)              # ie find the corresponding "true" wavelengths
        #########################################################################################################################################
        
        #fill arrays for fitting
        x.append(xord)
        x = np.array(x)
        order.append(np.repeat(int(ordnum), len(xord)))
        order = np.array(order)
        #m_order.append(np.repeat(105 - int(ordnum), len(xord)))
        m_order.append(np.repeat(64 + int(ordnum), len(xord)))
        m_order = np.array(m_order)
        wl.append(refwlord)
        wl = np.array(wl)
    
    
    #now re-normalize arrays to [-1,+1]
    x_norm = (x / ((len(data)-1)/2.)) - 1.
    order_norm = ((order-1) / ((len(thflux)-1)/2.)) - 1.       
           
    #call the fitting routine
    p = fit_poly_surface_2D(x_norm, order_norm, wl, weights=None, polytype=polytype, poly_deg=poly_deg, debug_level=0)    
    
#     if return_all_pars:
#         return dispsol,fitshapes
#     else:
#         return dispsol

    if savetable:
        now = datetime.datetime.now()
        model_wl = p(x_norm, order_norm)
        resid = wl - model_wl
        outfn = '/Users/christoph/OneDrive - UNSW/linelists/AAT_folder/lines_used_in_fit_as_of_'+str(now)[:10]+'.dat'
        outfn = open(outfn, 'w')
        outfn.write('line number   order_number   physical_order_number     pixel      reference_wl[A]   model_wl[A]    residuals[A]\n')
        outfn.write('=====================================================================================================================\n')
        for i in range(len(x)):
                outfn.write("   %3d             %2d                 %3d           %11.6f     %11.6f     %11.6f     %9.6f\n" %(i+1, order[i], m_order[i], x[i], wl[i], model_wl[i], resid[i]))
        outfn.close()
              

    if return_full:
        #xx = np.arange(len(data))            # already done above
        xxn = (xx / ((len(xx)-1)/2.)) - 1.
        oo = np.arange(1,len(thflux))
        oon = ((oo-1) / ((len(thflux)-1)/2.)) - 1.   
        X,O = np.meshgrid(xxn,oon)
        p_wl = p(X,O)

    
    if timit:
        print('Time elapsed: ',time.time() - start_time,' seconds')
        

    if return_full:
        return p,p_wl
    else:
        return p





def get_simu_dispsol(fibre=None, path='/Users/christoph/OneDrive - UNSW/dispsol/', npix=4096):
    """
    Get wavelength solution from Veloce Zemax file for a given fibre (1...28)
    WARNING: If no fibre is given, a mean wavelength solution across all orders is calculated!!!
    """
    #read dispersion solution from file
    if fibre is None:
        dispsol = np.load(path + 'mean_dispsol_by_orders_from_zemax.npy').item()
        orders = dispsol.keys()
    else:
        dbf = np.load(path + 'dispsol_by_fibres_from_zemax.npy').item()
        orders = dbf['fiber_1'].keys()
     
    #read extracted spectrum from files (obviously this needs to be improved)
    xx = np.arange(npix)
     
    #this is so as to match the order number with the physical order number (66 <= m <= 108)
    # order01 corresponds to m=66
    # order43 corresponds to m=108
    wl = {}
    for o in orders:
        m = o[5:]
        ordnum = str(int(m)-65).zfill(2)
        if fibre is None:
            wl['order_'+ordnum] = dispsol['order'+m]['model'](xx)
        else:
            wl['order_'+ordnum] = dbf['fiber_'+str(fibre)][o]['fitparms'](xx)
 
    return wl





def quick_bg_fix(raw_data, npix=4112):
    left_xx = np.arange(npix/2)
    right_xx = np.arange(npix/2, npix)
    left_bg = ndimage.minimum_filter(ndimage.gaussian_filter(raw_data[:npix/2],3), size=100)
    right_bg = ndimage.minimum_filter(ndimage.gaussian_filter(raw_data[npix/2:],3), size=100)
    data = raw_data.copy()
    data[left_xx] = raw_data[left_xx] - left_bg
    data[right_xx] = raw_data[right_xx] - right_bg
    return data





def get_arc_dispsol(thflux, satmask=None, polytype='chebyshev', lamptype=None, deg_spectral=5, deg_spatial=3,
                    return_full=True, savetables=True, debug_level=0, timit=False):

    if lamptype.lower() not in ['thar', 'thxe']:
        lamptype = raw_input('Please enter valid lamp type ["thar" / "thxe"]: ')

    # read the sacred Michael Murphy ThAr line list 
    wn, wlair, relint, species1, species2, reference = readcol('/Users/christoph/OneDrive - UNSW/linelists/thar_mm_new.txt', twod=False, skipline=1)
    wlvac = 1.e8 / wn
    
    if debug_level >= 1:
        # mm_grid, mm_spec_air = make_gaussmask_from_linelist(wlair, relint=10.**relint)
        mm_grid, mm_spec_unit_air = make_gaussmask_from_linelist(wlair, relint=None)
        # mm_grid, mm_spec_vac = make_gaussmask_from_linelist(wlvac, relint=10.**relint)
        # mm_grid, mm_spec_unit_vac = make_gaussmask_from_linelist(wlvac, relint=None)
    
    # read best existing wavelength solution
    # p = np.load('/Users/christoph/OneDrive - UNSW/dispsol/veloce_thar_dispsol_coeffs_20180921.npy').item()
    # thar_dispsol = pyfits.getdata('/Users/christoph/OneDrive - UNSW/dispsol/veloce_thar_dispsol_20180916.fits')
    p = np.load('/Users/christoph/OneDrive - UNSW/dispsol/veloce_thar_dispsol_coeffs_air_as_of_20180923.npy').item()
    thar_dispsol = pyfits.getdata('/Users/christoph/OneDrive - UNSW/dispsol/veloce_thar_dispsol_17sep30080_7x7_air.fits')
    ref_linenum, ref_ordnum, ref_m, ref_pix, ref_wl, _, _, _, _, _ = readcol('/Users/christoph/OneDrive - UNSW/linelists/thar_lines_used_in_7x7_fit_as_of_2018-10-19.dat', twod=False, skipline=2)
    
    # prepare output file
    if savetables:
        now = datetime.datetime.now()
        #model_wl = p(x_norm, order_norm)
        #resid = wl - model_wl
        outfn = '/Users/christoph/OneDrive - UNSW/linelists/'+lamptype+'_lines_found_as_of_'+str(now)[:10]+'.dat'
        outfile = open(outfn, 'w')
        outfile.write('line number   order_number   physical_order_number     pixel       air_ref_wl[A]   vac_ref_wl[A] \n')
        outfile.write('=================================================================================================\n')
        outfile.close()
    
    # prepare global output arrays
    x = []
    nlines = []
    order = []
    ref_wl_air = []
    ref_wl_vac = []
    
    # if not passed into the routine, define a mask that masks out saturated arc lines and some other crap
    if satmask is None:
        n_ord = thflux.shape[0]
        satmask = define_arc_mask(lamptype, n_ord)
    
    # search region size in Angstroms
    # search_region_size = 0.7
    search_region_size = 0.5
        
    #######################################################################################################################################
    #loop over orders
    for ord in range(thflux.shape[0]):
    
        # order numbers for output file
        ordnum = str(ord+1).zfill(2)
        mord = (ord+1) + 64
    
        # prepare output arrays for this order
        ord_x = []
        ord_ref_wl_air = []
        ord_ref_wl_vac = []
        
        # normalized co-ordinates
        ord_norm = (ord / ((40-1)/2.)) - 1.
        
        #define raw 1-dim data
        raw_data = thflux[ord,:]
        
        # clean up data
        data = quick_bg_fix(raw_data)
        
        # mask out saturated lines
        ordmask = satmask[ord,:]
        data[ordmask] = 0.
        
        # find peaks
        ####### TODO: automate this!!!
        thresh = 1500
        bgthresh = 1000
        fitted_line_pos = fit_emission_lines(data, return_all_pars=False, varbeta=False, timit=False, verbose=False, thresh=thresh, bgthresh=bgthresh)
        
        if debug_level >= 1:
            print(str(len(fitted_line_pos)) + ' lines found in order ' + ordnum)
            
        # select appropriate region in master line list
        testrange = np.logical_and(mm_grid > np.min(thar_dispsol[ord,:]) - 20, mm_grid < np.max(thar_dispsol[ord,:]) + 20)
        
        # PLOT
        if debug_level >= 1:
            xx = np.arange(len(data))
            plt.plot(mm_grid[testrange], mm_spec_unit_air[testrange])
            plt.plot(thar_dispsol[ord,:],data/np.max(data),'.-')
            plt.title('Order '+str(ord+1).zfill(2)) 
        
        # loop over all lines found - search for possible matches in the ThAr master line list
        for i,checkval in enumerate(fitted_line_pos):
            
            # normalized co-ordinates
            x_norm = (checkval / ((len(data)-1)/2.)) - 1.
            
            if debug_level >= 1:
                print('Trying to identify line in order ' + ordnum + ', at pixel ' + str(checkval) + '  -  rough wl estimate: ' + str(p(x_norm, ord_norm)) + ' A')
            
            if debug_level >= 3:
                # print(i,checkval)
                # print(i, checkval, thar_dispsol[order, np.round(checkval,0).astype(int)])
                # print(i, checkval, thar_dispsol[order, np.round(checkval,0).astype(int)], p(x_norm, ord_norm))
                print(i, checkval, p(x_norm, ord_norm))
            
            # select match(es)
            # q = np.logical_and(ord_pix > checkval + wshift - search_region_size, ord_pix < checkval + wshift + search_region_size)
            # q = np.logical_and(wlair > thar_dispsol[order, np.round(checkval,0).astype(int)] - search_region_size, wlair < thar_dispsol[order, np.round(checkval,0).astype(int)] + search_region_size)
            q = np.logical_and(wlair > p(x_norm, ord_norm) - search_region_size, wlair < p(x_norm, ord_norm) + search_region_size)
            
            found = np.sum(q)
            if found > 0:
                if debug_level >= 1:
                    print(str(found) + ' possible match(es) found!')
                if found == 1:
                    ord_x.append(checkval)
                    ord_ref_wl_air.append(wlair[q])
                    ord_ref_wl_vac.append(wlvac[q])
                else:
                    if debug_level >= 1:
                        print('Selecting closest match...')
                    qq = find_nearest(wlair, p(x_norm, ord_norm), return_index=True)
                    ord_x.append(checkval)
                    ord_ref_wl_air.append(wlair[qq])
                    ord_ref_wl_vac.append(wlvac[qq])
            else:
                if debug_level >= 1:
                    print('NO LINE FOUND in MASTERTABLE!!!')
        
        # convert to numpy arrays
        ord_x = np.array(ord_x)
        ord_ref_wl_air = np.array(ord_ref_wl_air)
        ord_ref_wl_vac = np.array(ord_ref_wl_vac)
        
        if debug_level >= 1:
            print(str(len(ord_x)) + ' lines could initially be matched to known lines from MM line list')
        
        
        # perform sanity-check 1D fit
        order_fit = np.poly1d(np.polyfit(ord_x, ord_ref_wl_air, 3))
        if debug_level >= 2:
            plt.figure()
            plt.title('Order '+str(ord+1).zfill(2))
            plt.scatter(ord_x, ord_ref_wl_air, color='r')
            plt.plot(xx, order_fit(xx))
            plt.xlim(0,4111)
        # remove obvious outliers
        fitres = ord_ref_wl_air - order_fit(ord_x)
        # do sigma clipping
        goodres,goodix,badix = single_sigma_clip(fitres, 2, return_indices=True)
        if debug_level >= 2:
            plt.figure()
            plt.title('Order '+str(ord+1).zfill(2)+'  -  residuals')
            plt.plot(ord_x, fitres,'x-')
            plt.scatter(ord_x[badix], fitres[badix], marker='o', color='red')
            plt.xlim(0,4111)
        # this does not work properly yet, do it manually for now
#         while len(badix) > 0:
#             ord_x = ord_x[goodix]
#             ord_ref_wl_air = ord_ref_wl_air[goodix]
#             ord_ref_wl_vac = ord_ref_wl_vac[goodix]
#             order_fit = np.poly1d(np.polyfit(ord_x, ord_ref_wl_air, 3))
#             fitres = ord_ref_wl_air - order_fit(ord_x)
#             goodres,goodix,badix = single_sigma_clip(fitres, 2, return_indices=True)
#             plt.clf()
#             plt.plot(ord_x, fitres,'x')
#             plt.scatter(ord_x[badix], fitres[badix], marker='o', color='red')
        
        ######################################################################
        allix = np.arange(len(ord_x))
        #badix = [2,4,8,9]
        goodix = [ix for ix in allix if ix not in badix]
        
        ord_x = ord_x[goodix]
        ord_ref_wl_air = ord_ref_wl_air[goodix]
        ord_ref_wl_vac = ord_ref_wl_vac[goodix]
        order_fit = np.poly1d(np.polyfit(ord_x, ord_ref_wl_air, 3))
        fitres = ord_ref_wl_air - order_fit(ord_x)
        plt.plot(ord_x, fitres,'x-')
        ######################################################################
        
        # now we can fill the global arrays and save info to table 
        if debug_level >= 1:
            print(str(len(ord_x)) + ' good lines found in order ' + ordnum)
        nlines.append(len(ord_x))
        order.append(np.repeat(ord+1, len(ord_x)))
        x.append(ord_x)
        ref_wl_air.append(ord_ref_wl_air)
        ref_wl_vac.append(ord_ref_wl_vac)
        if savetables:
            outfile = open(outfn, 'a+')
            for i in range(len(ord_x)):
                outfile.write("   %3d             %2d                 %3d           %11.6f     %11.6f     %11.6f\n" %(i+1, int(ordnum), mord, ord_x[i], ord_ref_wl_air[i], ord_ref_wl_vac[i]))
            outfile.close()                                                                                                                                                                        
    #######################################################################################################################################
    # end of loop over all orders
    
    # convert to numpy arrays
    x = np.array(x)
    nlines = np.array(nlines)
    order = np.array(order)
    ref_wl_air = np.array(ref_wl_air)
    ref_wl_vac = np.array(ref_wl_vac)
        
    # go to normalized co-ordinates
    x_norm = (x / ((len(data)-1)/2.)) - 1.
    order_norm = ((order-1) / ((40-1)/2.)) - 1.

    # call the 2D fitting routine
    p_air = fit_poly_surface_2D(x_norm, order_norm, ref_wl_air, weights=None, polytype=polytype, poly_deg_x=deg_spectral, poly_deg_y=deg_spatial, debug_level=0)    
    p_vac = fit_poly_surface_2D(x_norm, order_norm, ref_wl_vac, weights=None, polytype=polytype, poly_deg_x=deg_spectral, poly_deg_y=deg_spatial, debug_level=0)    
    
    if savetables:
        # get residuals from fit
        model_wl_air = p_air(x_norm, order_norm)
        model_wl_vac = p_vac(x_norm, order_norm)
        resid_air = ref_wl_air - model_wl_air
        resid_vac = ref_wl_vac - model_wl_vac
        now = datetime.datetime.now()
        outfn2 = '/Users/christoph/OneDrive - UNSW/linelists/'+lamptype+'_lines_used_in_' + str(deg_spectral) + 'x' + str(deg_spatial) + '_fit_as_of_'+str(now)[:10]+'.dat'
        outfile2 = open(outfn2, 'w')
        outfile2.write('line number   order_number   physical_order_number     pixel       air_ref_wl[A]   vac_ref_wl[A]   air_model_wl[A]   vac_model_wl[A]   air_resid[A]   vac_resid[A] \n')
        outfile2.write('===================================================================================================================================================================\n')
        for i in range(len(x)):
            outfile2.write("   %3d             %2d                 %3d           %11.6f     %11.6f     %11.6f     %11.6f       %11.6f    %11.6f    %11.6f\n" 
                         %(i+1, order[i], order[i]+64, x[i], ref_wl_air[i], ref_wl_vac[i], model_wl_air[i], model_wl_vac[i], resid_air[i], resid_vac[i]))
        outfile2.close()                                                                                                                                                                     

    # evaluate for every pixel along each order
    if return_full:
        xxn = (xx / ((len(xx)-1)/2.)) - 1.
        oo = np.arange(1,len(thflux)+1)
        oon = ((oo-1) / ((len(thflux)-1)/2.)) - 1.   
        X,O = np.meshgrid(xxn,oon)
        p_air_wl = p_air(X,O)
        p_vac_wl = p_vac(X,O)
        return p_air_wl, p_vac_wl
    else:
        return p_air, p_vac
    




def define_arc_mask(lamptype, n_ord):

    mask = np.zeros((n_ord,4112), dtype='bool')

    if lamptype.lower() == 'thar':
        mask[1, 3060:3140] = True
        mask[2, 2340:2590] = True
        mask[7, 1650:1870] = True
        mask[7, 3550:3670] = True
        mask[8, 1180:1450] = True
        mask[8, 1560:1830] = True
        mask[9, 2260:2420] = True
        mask[10, 2920:3500] = True
        mask[11, 500:1150] = True
        mask[11, 2930:3260] = True
        mask[12, 475:850] = True
        mask[12, 2120:2320] = True
        mask[14, 2640:2820] = True
        mask[15, 2380:2685] = True
        mask[16, 0:50] = True
        mask[16, 3060:3160] = True
        mask[16, 3265:3435] = True
        mask[17, 785:960] = True
        mask[17, 1100:1275] = True
        mask[18, 1850:2050] = True
        mask[22, 1326:1420] = True
        mask[23, 1850:2050] = True
        mask[24, 1390:1765] = True
    elif lamptype.lower() == 'thxe':
        print('WARNING: masks for ThXe lamp are not defined yet!!!!!')
    else:
        print('ERROR: lamp type must either be "ThAr" or "ThXe"!!!')
        return -1
    
    return mask





def get_dispsol_from_known_lines(thflux, fibre=None, fitwidth=4, satmask=None, lamptype='thar', minsigma=0.4, maxsigma=2.,
                                 sigma_0=0.85, minamp=0., maxamp=np.inf, return_all_pars=False, deg_spectral=7, deg_spatial=7,
                                 polytype='chebyshev', return_full=True, savetable=True, outpath=None, debug_level=0, timit=False):

    #read master table
    linenum, order, m, pix, wlref, vac_wlref, _, _, _, _ = readcol('/Users/christoph/OneDrive - UNSW/linelists/thar_lines_used_in_7x7_fit_as_of_2018-10-19.dat', twod=False, skipline=2)
    del _
   
    xx = np.arange(thflux.shape[1])
   
    all_pix = []
    all_wl = []
    all_vac_wl = []
    all_order = []
    if return_all_pars:
        allsigma = []
        allamp = []
        alloffset = []
        allslope = []
   
    if satmask is None:
        n_ord = thflux.shape[0]
        satmask = define_arc_mask(lamptype, n_ord)
   
   
    # loop over all orders
    for ord in range(thflux.shape[0]):
    
        # order numbers for output file
        ordnum = str(ord+1).zfill(2)
        mord = (ord+1) + 64
    
        if debug_level >= 1:
            print('Order ' + ordnum)
        
        #define raw 1-dim data
        raw_data = thflux[ord,:]
        
        # clean up data
        data = quick_bg_fix(raw_data)
        
        # mask out saturated lines
        ordmask = satmask[ord,:]
        data[ordmask] = 0.
        
        # select individual orders from master table
        ix = np.argwhere(m == mord).flatten()
        ord_pix = pix[ix]
        ord_wl = wlref[ix]
        ord_vac_wl = vac_wlref[ix]
        
        # some housekeeping...
        new_ord_pix = []
        new_ord_wl = []
        new_ord_vac_wl = []
        new_ord_order = []
        if return_all_pars:
            new_ord_sigma = []
            new_ord_amp = []
            new_ord_offset = []
            new_ord_slope = []
            
        # loop over all good lines in that order
        for xguess, wl, vac_wl in zip(ord_pix, ord_wl, ord_vac_wl):
            if debug_level >= 2:
                print(xguess)
            peaks = np.r_[int(np.round(xguess, 0))] 
            xrange = xx[np.max([0,peaks[0] - fitwidth]) : np.min([peaks[-1] + fitwidth + 1,len(data)-1])] 
#             guess = np.array([xguess, sigma_0, data[int(np.round(xguess, 0))], np.min(data[xrange]), 0.])
#             popt, pcov = op.curve_fit(gaussian_with_offset_and_slope, xrange, data[xrange], p0=guess, bounds=([xguess-2,minsigma,minamp,-np.inf,-np.inf],[xguess+2,maxsigma,maxamp,np.inf,np.inf]))
            # shift the peak to roughly zero, so that offset has sensible values 
            guess = np.array([xguess - peaks, sigma_0, np.max(data[xrange]), np.min(data[xrange]), 0.])
            try:
                popt, pcov = op.curve_fit(gaussian_with_offset_and_slope, xrange - peaks, data[xrange], p0=guess, 
                                          bounds=([xguess-peaks-2,minsigma,minamp,-np.inf,-np.inf], [xguess-peaks+2,maxsigma,maxamp,np.inf,np.inf]))
                
                # remove rubbish fits (very low signal)
                if popt[2] > 50 and np.max(data[xrange]) > 30:
                    new_ord_pix.append(popt[0] + peaks)
                    new_ord_wl.append(wl)
                    new_ord_vac_wl.append(vac_wl)
                    new_ord_order.append(ord)
                    if return_all_pars:
                        new_ord_sigma.append(popt[1])
                        new_ord_amp.append(popt[2])
                        new_ord_offset.append(popt[3])
                        new_ord_slope.append(popt[4])
                else:
                    print('WARNING: not enough signal to fit a good peak at location', xguess)
                    
            except RuntimeError:
                print('WARNING: failed to fit a peak at location', xguess)
                
#             # plot a single fit for debugging
#             plot_osf = 10
#             plot_os_grid = np.linspace((xrange-peaks)[0], (xrange-peaks)[-1], plot_osf * (len(xrange)-1) + 1)
#             plt.plot(xx-peaks, data,'b.-')
#             plt.plot(xrange-peaks, data[xrange],'y.-')
#             plt.xlim(-10,10)
#             plt.ylim(np.min(data[xrange]) - 0.25 * (np.max(data[xrange] - np.min(data[xrange]))), np.max(data[xrange]) + 0.25 * (np.max(data[xrange] - np.min(data[xrange])))) 
#             plt.plot(plot_os_grid, gaussian_with_offset_and_slope(plot_os_grid, *guess),'r--', label='initial guess')
#             plt.plot(plot_os_grid, gaussian_with_offset_and_slope(plot_os_grid, *popt),'g-', label='best fit')
#             plt.axvline(popt[0], color='g', linestyle=':')
#             plt.legend()
#             plt.xlabel('pixel')
#             plt.ylabel('counts')
#             plt.text(5, np.max(data[xrange]) - 0.1 * (np.max(data[xrange] - np.min(data[xrange]))), 'mu = '+str(np.round(popt[0],4)))
    
        all_pix.append(new_ord_pix)
        all_wl.append(new_ord_wl)
        all_vac_wl.append(new_ord_vac_wl)
        all_order.append(new_ord_order)
        if return_all_pars:
            allsigma.append(new_ord_sigma)
            allamp.append(new_ord_amp)
            alloffset.append(new_ord_offset)
            allslope.append(new_ord_slope)
            
#     if return_all_pars:
#         return all_pix, all_wl, all_vac_wl, allsigma, allamp, alloffset, allslope
#     else:
#         return all_pix, all_wl, all_vac_wl
    
    
    # now make everything a simple 1-dim array
    x = np.array([item for sublist in all_pix for item in sublist])
    lam = np.array([item for sublist in all_wl for item in sublist])
    vac_lam = np.array([item for sublist in all_vac_wl for item in sublist])
    all_order_correct_format = np.array([item for sublist in all_order for item in sublist])
    
    # go to normalized co-ordinates
    x_norm = np.squeeze( (x / ((len(data)-1)/2.)) - 1. )
    order_norm = np.squeeze( ((all_order_correct_format) / ((40-1)/2.)) - 1. )
    
    # call the 2D fitting routine
    p_air = fit_poly_surface_2D(x_norm, order_norm, lam, weights=None, polytype=polytype, poly_deg_x=deg_spectral, poly_deg_y=deg_spatial, debug_level=0)    
    p_vac = fit_poly_surface_2D(x_norm, order_norm, vac_lam, weights=None, polytype=polytype, poly_deg_x=deg_spectral, poly_deg_y=deg_spatial, debug_level=0)  
    
    if savetable:
        # get residuals from fit
#         model_wl_air = p_air(x_norm, order_norm)
#         model_wl_vac = p_vac(x_norm, order_norm)
#         resid_air = lam - model_wl_air
#         resid_vac = vac_lam - model_wl_vac
        now = datetime.datetime.now()
        if outpath is None:
            outpath = '/Users/christoph/OneDrive - UNSW/linelists/'
        if fibre is None:
            outfn = outpath + lamptype + '_lines_as_of_'+str(now)[:10]+'.dat'
        else:
            outfn = outpath + lamptype + '_lines_fibre_' + fibre + '_as_of_'+str(now)[:10]+'.dat'
        outfile = open(outfn, 'w')
        outfile.write('line number   order_number   physical_order_number     pixel       air_ref_wl[A]   vac_ref_wl[A] \n') #  air_model_wl[A]   vac_model_wl[A]   air_resid[A]   vac_resid[A] \n')
        outfile.write('=================================================================================================\n')
        for i in range(len(x)):
            outfile.write("   %3d             %2d                 %3d           %11.6f     %11.6f     %11.6f \n" #     %11.6f       %11.6f    %11.6f    %11.6f\n" 
                         %(i+1, all_order_correct_format[i]+1, all_order_correct_format[i]+65, x[i], lam[i], vac_lam[i])) #, model_wl_air[i], model_wl_vac[i], resid_air[i], resid_vac[i]))
        outfile.close()                                                                                                                                                                     

    # evaluate for every pixel along each order
    if return_full:
        xxn = (xx / ((len(xx)-1)/2.)) - 1.
        oo = np.arange(1,len(thflux)+1+1)     # note the double-adding of 1 is intentional in order to make a wl-solution for 40 orders!!!
        oon = ((oo-1) / ((40-1)/2.)) - 1.   
        X,O = np.meshgrid(xxn,oon)
        p_air_wl = p_air(X,O)
        p_vac_wl = p_vac(X,O)
        return p_air_wl, p_vac_wl
    else:
        return p_air, p_vac
    
    



def define_pixel_offsets_between_fibres(relto='S1', savedict=False, saveplots=False, extra_lfc_offset=True, return_all=False, debug_level=0):
    # difference in pixel space (ie peak locations)    
    path = '/Users/christoph/OneDrive - UNSW/dispsol_tests/20180917/'
    xx = np.arange(4112)
    fibslot = [0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25]
    fibname = ['S5', 'S2', '07', '18', '17', '06', '16', '15', '05', '14', '13', '01', '12', '11', '04', '10', '09', '03', '08', '19', '02', 'S4', 'S3', 'S1']
    
    pixfit_coeffs = {}
    zeroth_coeffs = np.zeros((39,24))
    first_coeffs = np.zeros((39,24))
    
    for o in range(39):
        
        ord = 'order_'+str(o+1).zfill(2)
        
        ref_filename = path + 'thar_lines_fibre_' + relto + '_as_of_2018-11-09.dat'
        linenum, order, m, pix, wlref, vac_wlref = readcol(ref_filename, twod=False, skipline=2)
        # linenum, order, m, pix, wlref, vac_wlref, _, _, _, _ = readcol('/Users/christoph/OneDrive - UNSW/linelists/thar_lines_used_in_7x7_fit_as_of_2018-10-19.dat', twod=False, skipline=2)
        ix = np.argwhere(order == o+1).flatten()
        ref_x = pix[ix]
        ref_wlref = wlref[ix]
    #     order_fit = np.poly1d(np.polyfit(ref_x, ref_wlref, 5))
    #     ref_fit = order_fit(xx)
        
        pixfit_coeffs[ord] = {}
        ord_zeroth_coeffs = []
        ord_first_coeffs = []
        
        for fib in fibname:
            filename = path + 'thar_lines_fibre_' + fib + '_as_of_2018-11-09.dat'
            linenum, order, m, pix, wlref, vac_wlref = readcol(filename, twod=False, skipline=2)
            ix = np.argwhere(order == o+1).flatten()
            x = pix[ix]
            lam = wlref[ix]
            matched_ref_x = ref_x[np.in1d(ref_wlref, lam)]
            matched_x = x[np.in1d(lam, ref_wlref)]
            delta_x = matched_x - matched_ref_x
        #     plt.plot(matched_ref_x, x - matched_ref_x, 'x-', label='fibre '+fib)
            # perform sanity-check 1D fit (remove only very obvious outliers, as there is quite a bit of scatter)
            test_fit = np.poly1d(np.polyfit(matched_ref_x, delta_x, 1))
            if debug_level >= 2:
                plt.figure()
                plt.plot(matched_ref_x, delta_x, 'x-')
                plt.title('fibre '+fib)
                plt.plot(xx, test_fit(xx), 'r--')
                plt.xlim(0, 4111)
            # remove obvious outliers
            fitres = delta_x - test_fit(matched_ref_x)
            # do single sigma clipping
            goodres,goodix,badix = single_sigma_clip(fitres, 2, return_indices=True)
            #fit again
            pix_fit = np.poly1d(np.polyfit(matched_ref_x[goodix], delta_x[goodix], 1))
            if debug_level >= 2:
                plt.scatter(matched_ref_x[badix], delta_x[badix], color='r', marker='o')
                plt.plot(xx, pix_fit(xx), 'g-')
            
            pixfit_coeffs[ord]['fibre_'+fib] = pix_fit
            ord_zeroth_coeffs.append(pix_fit[0])
            ord_first_coeffs.append(pix_fit[1])
    #         #now fit new dispsol to the model-shifted lines
    #         order_fit = np.poly1d(np.polyfit(ref_x + pix_fit(ref_x), ref_wlref, 5))
    #         plt.plot(xx, order_fit(xx) - ref_fit, label='fibre '+fib)
        
        # now fill array of fit coefficients
        zeroth_coeffs[o,:] = ord_zeroth_coeffs
        first_coeffs[o,:] = ord_first_coeffs     
        
        # now estimate the extra offset between fibre 'S1' and the LFC fibre by extrapolating to the next fibre slot (ie slot 26)
        if extra_lfc_offset:
            ex_fit_0 = np.poly1d(np.polyfit(fibslot, zeroth_coeffs[o,:], 2))
            ex_fit_1 = np.poly1d(np.polyfit(fibslot, first_coeffs[o,:], 2))
            zeroth_coeffs[o,:] -= ex_fit_0(26)
            first_coeffs[o,:] -= ex_fit_1(26)
            for fib in pixfit_coeffs[ord].keys():
                pixfit_coeffs[ord][fib][0] -= ex_fit_0(26)
                pixfit_coeffs[ord][fib][1] -= ex_fit_1(26)
                
        # now save the dictionary containing the fit coefficients that describe shift and slope in pixel space to file
        if savedict:
            now = datetime.datetime.now()
            if extra_lfc_offset:
                np.save(path + 'pixfit_coeffs_relto_LFC_as_of_' + str(now)[:10] + '.npy', pixfit_coeffs)
            else:
                np.save(path + 'pixfit_coeffs_relto_' + relto + '_as_of_' + str(now)[:10] + '.npy', pixfit_coeffs)
        
        # now save a plot of both 0th and 1st order terms
        if saveplots:
            
            ordstring = 'Order_'+str(o+1).zfill(2)    
        
        #     plt.figure()
            plt.plot(fibslot, zeroth_coeffs, 'bo-')
            plt.title('0th order coefficients  -  ' + ordstring)
            plt.xlim(-2,27)
            plt.xlabel('fibre')
            plt.ylabel('offset [pix]')
            plt.axvline(2, color='gray', linestyle=':')
            plt.axvline(22, color='gray', linestyle=':')    
            plt.savefig(path + '0th_order_coeffs_pixel_shits_' + ordstring + '.eps')
            plt.clf()
            
        #     plt.figure()
            plt.plot(fibslot, first_coeffs, 'bo')
            plt.title('1st order coefficients  -  ' + ordstring)
            plt.xlim(-2,27)
            plt.xlabel('fibre')
            plt.ylabel('slope')
            plt.axvline(2, color='gray', linestyle=':')
            plt.axvline(22, color='gray', linestyle=':')    
            slope_fit = np.poly1d(np.polyfit(fibslot, first_coeffs, 2))
            plot_x = np.arange(-1,26,0.1)
            plt.plot(plot_x, slope_fit(plot_x), 'r')
            plt.savefig(path + '1st_order_coeffs_pixel_shits_' + ordstring + '.eps')
            plt.clf()
    
    if return_all:
        return pixfit_coeffs, zeroth_coeffs, first_coeffs
    else:
        return pixfit_coeffs





def get_dispsol_for_all_fibs(obsname, relto='LFC', twod=False, degpol=7, deg_spectral=7, deg_spatial=7, fibs='stellar', nx=4112,
                             polytype='chebyshev', refit=False, debug_level=0, timit=False, fudge=1., signflip_shift=True, signflip_slope=True):

    '''using Duncan's xcorrs to measure LFC shifts'''

    if timit:
        start_time = time.time()

    # laod default coefficients
    pixfit_coeffs = np.load('/Users/christoph/OneDrive - UNSW/dispsol_tests/20180917/pixfit_coeffs_relto_' + relto + '_as_of_2018-11-14.npy').item()
    
    # fibslot = [0,1,   3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,   23,24,25]
    fibname = ['S5', 'S2', '07', '18', '17', '06', '16', '15', '05', '14', '13', '01', '12', '11', '04', '10', '09', '03', '08', '19', '02', 'S4', 'S3', 'S1']
    
    nfib = len(fibname)
    
    # read master LFC linelist
    lfc_ord, lfc_pix, lfc_wl = readcol('/Users/christoph/OneDrive - UNSW/dispsol/laser_dispsol_20181015/PeakPos.txt', twod=False)
    lfc_pix -= 1.   # b/c DW is using Matlab, which starts indexing at 1 not at 0!!!!!!!

    # read file containing slope and offset as measured from LFC peak positions
    lfc_slope, lfc_shift = readcol('/Users/christoph/OneDrive - UNSW/dispsol/laser_offsets/relto_21sep30019/' + obsname + '_LFC_Slope_and_Offset.txt', twod=False)

    if signflip_slope:
        lfc_slope *= -1.
    if signflip_shift:
        lfc_shift *= -1.

    # some housekeeping...
    n_ord = len(np.unique(lfc_ord))
    xx = np.arange(nx)
    
    # prepare outputs
    wldict = {}
    # wl = np.zeros((n_ord, nfib, nx))
    wl = np.zeros((40, nfib, nx))
    
    if not twod:
        if debug_level >= 1:
            print('OK, getting a 1-D wavelength solution...')
        # loop over all orders
        for o in np.unique(lfc_ord)[:-1]:
            ord = 'order_'+str(o+1).zfill(2)     # DW's order numbers are offset by one to CB's
            wldict[ord] = {}
            ix = np.argwhere(lfc_ord == o).flatten()
            x0 = lfc_pix[ix]
            lam = lfc_wl[ix]
            master_lfc_fit = np.poly1d(np.polyfit(x0, lam, degpol))
            
            # either perform a new fit using the new x-values
            if refit:
                # FFFFUUUUU, broke my brain turning that around, but don't need to, b/c DW's LFC spectra are flipped (left-right) wrt to CBs
                # # now apply the shift to the pixel positions
                # diff = - lfc_shift[o-1] + lfc_slope[o-1] * x0     # signs are confusing here, but comparison to DW's results suggests this is correct
                # x = x0 + diff - np.max(diff) + np.min(diff)       # signs are confusing here, but comparison to DW's results suggests this is correctly
                diff = lfc_shift[o-1] + lfc_slope[o-1] * x0         # need the o-1 here because Duncan's indices start with 1
                x = x0 - fudge * diff
                lfc_fit = np.poly1d(np.polyfit(x, lam, degpol))
                newly_fit_dispsol = lfc_fit(xx)
                wldict[ord]['laser'] = newly_fit_dispsol[::-1]     # need to turn around, b/c DW's LFC spectra are flipped (left-right) wrt to CBs
                # need to flip x now, because the pixfit_coeffs are using CMB layout
                x_flipped = (nx - 1) - x
                # loop over all fibres
                for i,fib in enumerate(fibname):
                    xfib = x_flipped + pixfit_coeffs[ord]['fibre_'+fib](x_flipped)
                    fib_fit = np.poly1d(np.polyfit(xfib, lam, degpol))
                    wldict[ord][fib] = fib_fit(xx)
                    wl[o,i,:] = fib_fit(xx)
            # or re-evaluate the master fit at new x-values (better b/c less margin for fitting variations and hence more constancy)
            else:
                xx_prime = xx + fudge * (lfc_shift[o-1] + lfc_slope[o-1] * xx)
                eval_xprime_dispsol = master_lfc_fit(xx_prime)
                wldict[ord]['laser'] = eval_xprime_dispsol[::-1]   # need to turn around, b/c DW's LFC spectra are flipped (left-right) wrt to CBs
                # need to flip x now, because the pixfit_coeffs are using CMB layout
                x0_flipped = (nx - 1) - x0
                # loop over all fibres
                for i,fib in enumerate(fibname):
                    xfib_flipped = x0_flipped + pixfit_coeffs[ord]['fibre_'+fib](x0_flipped)
                    xfib = (nx - 1) - xfib_flipped   # and flip it back --> can't think right now, can I combine those two flips???
                    master_lfc_fit_fib = np.poly1d(np.polyfit(xfib, lam, degpol))
                    xx_prime_fib = xx + fudge * (lfc_shift[o-1] + lfc_slope[o-1] * xx)
                    eval_xprime_dispsol_fib = master_lfc_fit_fib(xx_prime_fib)
                    wldict[ord][fib] = eval_xprime_dispsol_fib[::-1]
                    wl[o,i,:] = eval_xprime_dispsol_fib[::-1]
       
    else:

        print('THIS IS STILL WRONG (needs to be flipped) - USE THE 1-DIM VERSION FOR NOW!!!')

        if debug_level >= 1:
            print('OK, getting a 2-D wavelength solution...')
            
        order = lfc_ord.copy() + 1     # DW's order numbers are offset by one to CB's
        pixel = np.tile(lfc_pix.copy(), (nfib,1))
        lam = lfc_wl.copy()
        
        # temporarily remove the blue-most order until I have incorporated that into the rest of the reduction
        dumix = np.argwhere(lfc_ord == 39).flatten()
        order = np.delete(order, dumix)
        pixel = np.delete(pixel,dumix,1)
        lam = np.delete(lam,dumix)
        
        # fit the pure LFC dispsol
        x_norm = np.squeeze( (pixel[0,:] / ((nx-1)/2.)) - 1. )
        order_norm = np.squeeze( ((order-1) / ((40-1)/2.)) - 1. )
        p_wl_lfc = fit_poly_surface_2D(x_norm, order_norm, lam, weights=None, polytype=polytype, poly_deg_x=deg_spectral, poly_deg_y=deg_spatial, debug_level=0)    
        # calculate wavelengths from model for full (n_ord x n_pix)-array
        xxn = (xx / ((len(xx)-1)/2.)) - 1.
        oo = np.arange(1,41)
        oon = ((oo-1) / ((40-1)/2.)) - 1.   
        X,O = np.meshgrid(xxn,oon)
        wl_lfc = p_wl_lfc(X,O)
            
        # loop over all orders
        for o in np.unique(lfc_ord)[:-1]:    
            ord = 'order_'+str(o+1).zfill(2)     # DW's order numbers are offset by one to CB's
            wldict[ord] = {}
            ix = np.argwhere(lfc_ord == o).flatten()
            x0 = lfc_pix[ix]
            # now apply the shift to the pixel positions
            diff = - lfc_shift[o-1] + lfc_slope[o-1] * x0   # signs are confusing here, but comparison to DW's results suggests this is correct
            x = x0 + diff - np.max(diff) + np.min(diff)     # signs are confusing here, but comparison to DW's results suggests this is correct
            # loop over all fibres
            for i,fib in enumerate(fibname):
                pixel[i,ix] = x + pixfit_coeffs[ord]['fibre_'+fib](x)
        
        # now fit a 2-D wavelength solution for each fibre individually
        # loop over all fibres
        for i,fib in enumerate(fibname):
            if debug_level >= 1:
                print(fib)
            # print(fib)
            # go to normalized co-ordinates
            x_norm = np.squeeze( (pixel[i,:] / ((nx-1)/2.)) - 1. )
            order_norm = np.squeeze( ((order-1) / ((40-1)/2.)) - 1. )
        
            # call the 2D fitting routine
            p_wl = fit_poly_surface_2D(x_norm, order_norm, lam, weights=None, polytype=polytype, poly_deg_x=deg_spectral, poly_deg_y=deg_spatial, debug_level=0)    
            
            # calculate wavelengths from model for full (n_ord x n_pix)-array
            xxn = (xx / ((len(xx)-1)/2.)) - 1.
            oo = np.arange(1,41)
            oon = ((oo-1) / ((40-1)/2.)) - 1.   
            X,O = np.meshgrid(xxn,oon)
            wl[:,i,:] = p_wl(X,O)
            
        for i,o in enumerate(np.unique(lfc_ord)[:-1]):    
            ord = 'order_'+str(o+1).zfill(2)
            for j,fib in enumerate(fibname):
                wldict[ord]['fibre_'+fib] = wl[o,j,:].copy()
            wldict[ord]['laser'] = wl_lfc[o,:].copy()

    if timit:
        print('Time elapsed: '+str(np.round(time.time() - start_time,1))+' seconds')

    if fibs not in ['all', 'stellar']:
        print('ERROR: "fibs" not valid!!!')
        return
    elif fibs == 'stellar':
        wl = wl[:,2:21,:]

    return wldict,wl





def get_dispsol_for_all_fibs_2(obsname, relto='LFC', degpol=7, fibs='stellar', nx=4112, eps=0.5, fudge=1., refit=False, debug_level=0, timit=False):
    '''using CGT's DAOPHOT results to measure LFC shifts'''

    lfc_path = '/Users/christoph/OneDrive - UNSW/lfc_peaks/'

    if timit:
        start_time = time.time()

    # load default coefficients
    pixfit_coeffs = np.load('/Users/christoph/OneDrive - UNSW/dispsol_tests/20180917/pixfit_coeffs_relto_' + relto + '_as_of_2018-11-14.npy').item()

    # fibslot = [0,1,   3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,   23,24,25]
    fibname = ['S5', 'S2', '07', '18', '17', '06', '16', '15', '05', '14', '13', '01', '12', '11', '04', '10', '09',
               '03', '08', '19', '02', 'S4', 'S3', 'S1']
    offsets = 1.97 * np.array(range(3) + range(4,23) + range(24,26))[::-1]   # 1.97 pix is the median fib-to-fib separation

    nfib = len(fibname)

    # read master LFC linelist
    lfc_ord, lfc_pix, lfc_wl = readcol('/Users/christoph/OneDrive - UNSW/dispsol/laser_dispsol_20181015/PeakPos.txt',
                                       twod=False)
    lfc_pix -= 1.  # b/c DW is using Matlab, which starts indexing at 1 not at 0!!!!!!!

    # read file containing LFC peak positions of observation
    _, yref, xref, _, _, _, _, _, _, _, _ = readcol(lfc_path + '21sep30019olc.nst', twod=False, skipline=2)
    try:
        _, y, x, _, _, _, _, _, _, _, _ = readcol(lfc_path + 'tauceti/' + obsname + 'olc.nst', twod=False, skipline=2)
    except:
        _, y, x, _, _, _, _, _, _ = readcol(lfc_path + 'tauceti/' + obsname + 'olc.nst', twod=False, skipline=2)
    del _
    xref = nx - xref
    x = nx - x
    yref = yref - 54.   # or 53??? but does not matter for getting the transformation matrix
    y = y - 54.         # or 53??? but does not matter for getting the transformation matrix
    # ref_peaks = divide_lfc_peaks_into_orders(xref, yref)
    # peaks = divide_lfc_peaks_into_orders(x, y)

    # some housekeeping...
    xx = np.arange(nx)

    if refit:
        # divide LFC peak positions into orders
        peaks = divide_lfc_peaks_into_orders(x, y)
    else:
        # get affine transformation matrix
        #     if   x' = x * M
        # # then   x  = x' * M_inv
        Minv = find_affine_transformation_matrix(xref, yref, x, y, timit=True, eps=2.)
        # also get the rough tracve of the LFC fibre (neede below)
        pid = np.load(lfc_path + 'lfc_P_id.npy').item()

    # prepare outputs
    wldict = {}
    # wl = np.zeros((n_ord, nfib, nx))
    wl = np.zeros((40, nfib, nx))

    # loop over all orders
    for o in np.unique(lfc_ord)[:-1]:
        ord = 'order_' + str(o + 1).zfill(2)  # DW's order numbers are offset by one to CB's
        wldict[ord] = {}
        ix = np.argwhere(lfc_ord == o).flatten()
        x0 = lfc_pix[ix]
        lam = lfc_wl[ix]
        master_lfc_fit = np.poly1d(np.polyfit(x0, lam, degpol))

        if refit:
            x = np.array(peaks[ord])[:,0]
            # check which peaks match peaks from the master solution above
            # also need to turn around x0 here
            x0 = (nx - 1) - x0
            good_x = []
            good_wls = []
            for xpos in x:
                distance = np.abs(x0 - xpos)
                if np.sum(distance < eps) > 0:
                    if np.sum(distance < eps) > 1:
                        print('FUGANDA: @', xpos)
                    # get indices of those peaks and hence their wavelengths
                    good_x.append(xpos)
                    good_wls.append(lam[distance < eps])

            # re-fit
            lfc_fit = np.poly1d(np.polyfit(good_x, np.squeeze(good_wls), degpol))
            newly_fit_dispsol = lfc_fit(xx)
            # no need to turn around anything anymore here
            wldict[ord]['laser'] = newly_fit_dispsol
            # loop over all fibres
            for i, fib in enumerate(fibname):
                xfib = good_x + pixfit_coeffs[ord]['fibre_' + fib](good_x)
                fib_fit = np.poly1d(np.polyfit(xfib, np.squeeze(good_wls), degpol))
                wldict[ord][fib] = fib_fit(xx)
                wl[o, i, :] = fib_fit(xx)

        else:
            # now get shifted coordinates
            coords = np.array([xx, pid[ord](xx), np.ones(nx)]).T
            shifted_coords = np.dot(coords, Minv)
            xx_prime = shifted_coords[:,0]

            # now re-evaluate the master fit at new x-values
            eval_xprime_dispsol = master_lfc_fit(xx_prime)

            wldict[ord]['laser'] = eval_xprime_dispsol[::-1]  # need to turn around, b/c DW's LFC spectra are flipped (left-right) wrt to CBs
            # need to flip x now, because the pixfit_coeffs are using CMB layout
            x0_flipped = (nx - 1) - x0

            # loop over all fibres
            for i, fib in enumerate(fibname):
                xfib_flipped = x0_flipped + pixfit_coeffs[ord]['fibre_' + fib](x0_flipped)
                xfib = (nx - 1) - xfib_flipped  # and flip it back --> can't think right now, can I combine those two flips???
                master_lfc_fit_fib = np.poly1d(np.polyfit(xfib, lam, degpol))

                # TODO :
                # coords_fib = np.array([xx, fibre_traces[fib](xx), np.ones(nx)]).T
                coords_fib = np.array([xx, pid[ord](xx) - offsets[i], np.ones(nx)]).T
                shifted_coords_fib = np.dot(coords_fib, Minv)
                xx_prime_fib = shifted_coords_fib[:,0]

                eval_xprime_dispsol_fib = master_lfc_fit_fib(xx_prime_fib)
                wldict[ord][fib] = eval_xprime_dispsol_fib[::-1]
                wl[o, i, :] = eval_xprime_dispsol_fib[::-1]

    if timit:
        print('Time elapsed: ' + str(np.round(time.time() - start_time, 1)) + ' seconds')

    if fibs not in ['all', 'stellar']:
        print('ERROR: "fibs" not valid!!!')
        return
    elif fibs == 'stellar':
        wl = wl[:, 2:21, :]

    return wldict, wl


