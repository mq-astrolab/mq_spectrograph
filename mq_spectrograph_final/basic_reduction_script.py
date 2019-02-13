'''
Created on 16 Jan 2019

@author: Christoph Bergmann
@editor: Jacob Parnell
'''

## importing statements - do not alter
import glob
import astropy.io.fits as pyfits
import numpy as np
import datetime
import copy
import matplotlib.pyplot as plt
from IPython.display import Image
import os
import time

from helper_functions import short_filenames
from calibration import get_bias_and_readnoise_from_bias_frames, make_offmask_and_ronmask, make_master_bias_from_coeffs, make_master_dark, correct_orientation, crop_overscan_region
from order_tracing import find_stripes, make_P_id, make_mask_dict, extract_stripes # find_tramlines
from spatial_profiles import fit_profiles, fit_profiles_from_indices
from extraction import *
from process_scripts import process_whites, process_science_images
from helper_functions import orientation_test_unautomated

## path to raw fits files
## for our Echelle++ test - mq simulations
path = '/Users/Jacob/Desktop/data_for_mq/' # mac example
#path = 'C:\\Users\\87463547\\Documents\\Physics_data\\' # windows example

## check to see if these directories exist, if not, create them - will be used to save generated plots to file
## this is for mac
if os.path.isdir(path + '/plots') == False:
    os.mkdir(path + 'plots')
save_plots = path + 'plots/' # added so that we can save the plots to file
if os.path.isdir(save_plots + 'order_tracing_plots/') == False:
    os.mkdir(save_plots + 'order_tracing_plots/')
save_plots = save_plots + 'order_tracing_plots/'

## this is for windows
#if os.path.isdir(path + '\\plots') == False:
#    os.mkdir(path + 'plots')
#save_plots = path + 'plots\\' # added so that we can save the plots to file
#if os.path.isdir(save_plots + '\\order_tracing_plots') == False:
#    os.mkdir(save_plots + 'order_tracing_plots\\')
#save_plots = save_plots + 'order_tracing_plots\\'

#######################################################################################################################

## (0) GET FILE INFO - calls the path and loads the fits files

stellar_list = glob.glob(path + 'mq_blackbody*.fits') #mq simulations
laser_list = glob.glob(path + 'mq_etalon*.fits') #mq simulations
white_list = glob.glob(path + 'mq_flat*.fits') #mq simulations

## (i) IMAGE ORIENTATION/CORRECTION
## here we test our input data to see if they are oriented correctly - may not be used when spec is built
## currently unautomated, only works for our inputs which are rotated 90 CW to our desired position
## only for mac as it imports it strangely
stellar_list,white_list,laser_list = orientation_test_unautomated(stellar_list, white_list, laser_list,path=path)

## generates a dummy image of the first .fit file from the stellar_list and returns the shape of that image
dumimg = pyfits.getdata(stellar_list[0])
ny,nx = dumimg.shape

#######################################################################################################################

## (1) BAD PIXEL MASK
## removed for our 'simpler code' but can be adapted from veloce code in other files

#######################################################################################################################

## (2) CALIBRATION - generate the bias, dark, and flat frames to calibrate the images to the right sizes

# gain = 1.
gain = [1., 1., 1., 1.]

## (i) BIAS - get offsets and read-out noise
## either from the bias frames or from the overscan regions. (units: [offsets] = ADUs; [RON] = e-)

# medbias,coeffs,offsets,rons = get_bias_and_readnoise_from_bias_frames(sorted(bias_list), degpol=5, clip=5., gain=gain,
#                                                                       save_medimg=True, debug_level=1, timit=True)

## create a MASTER BIAS frame and read-out noise mask (units = ADUs)
# offmask,ronmask = make_offmask_and_ronmask(offsets, rons, nx, ny, gain=gain, savefiles=True, path=path, timit=True)
# MB = make_master_bias_from_coeffs(coeffs, nx, ny, savefile=True, path=path, timit=True)

## OR
# MB = offmask.copy()

## OR
# MB = medbias.copy()

## OR
## for our test:
## setting the master bias to an array of all zeros with the same dimensions as the dummy image created earlier
## similar process for the readout noise mask, but with an array of ones (here set to 3s)
MB = np.zeros(dumimg.shape)
ronmask = np.ones(dumimg.shape) * 3.

## (ii) DARKS
## create (bias-subtracted) MASTER DARK frame (units = electrons)
# MD = make_master_dark(dark_list, MB=MB, gain=gain, scalable=False, savefile=True, path=path, timit=True)
# MDS = make_master_dark(dark_list, MB=medbias, gain=gain, scalable=True, savefile=True, path=path, debug_level=1,
#                        timit=True)

## for our test:
## setting the master dark to an array of all zeros with the same dimensions as the dummy image created earlier
## deletes the value assigned to the dummy image variable
MDS = np.zeros(dumimg.shape)
del dumimg

## (iii) WHITES
## create (bias- & dark-subtracted) MASTER WHITE frame and corresponding error array (units = electrons)
## need to check if the whites are in the correct orientation - and if not, then flip them
MW,err_MW = process_whites(white_list, MB=MB, ronmask=ronmask, MD=MDS, gain=gain, scalable=True, fancy=False,
                           clip=5., savefile=True, saveall=False, diffimg=False, path=path, debug_level=1, timit=False)

#######################################################################################################################

## (3) ORDER TRACING - find and trace the orders/stripes in a flat field Echelle spectrum
## find orders roughly
P,tempmask = find_stripes(MW, deg_polynomial=2, min_peak=0.20, gauss_filter_sigma=3., simu=True, debug_level=2, save_plots=save_plots, maskthresh=30)

## assign physical diffraction order numbers to order-fit polynomials and bad-region masks
## (this is only a dummy function for now)
P_id = make_P_id(P)
mask = make_mask_dict(tempmask)
np.save(path + 'P_id.npy', P_id)

## extract stripes of user-defined width from the science image, centred on the polynomial fits defined in step (1)
MW_stripes,MW_indices = extract_stripes(MW, P_id, return_indices=True, slit_height=7)

## quick method of extraction
pix,flux,err = extract_spectrum_from_indices(MW, err_MW, MW_indices, method='quick', slit_height=7, RON=ronmask,
                                             savefile=True, filetype='fits', obsname='master_white', path=path,
                                             timit=True)

np.save(path + 'flux.npy', flux) # added so the flux can be used in plot_script.py

## optimal method of extraction
## commented out because we are moving forward using the quick method
# pix,flux,err = extract_spectrum_from_indices(MW, err_MW, MW_indices, method='optimal', slope=False, offset=False,
#                                             fibs='single', fibpos='05', slit_height=7, RON=ronmask, simu=True,
#                                             savefile=True, filetype='fits', obsname='master_white', path=path,
#                                             timit=True)

#np.save(path + 'flux.npy', flux) # added so the flux can be used in plot_script.py

#######################################################################################################################

## leave this out until you have figured out wavelength calibration
## is not really needed until you start to use real data, rather than simulated data

## if we want to determine spatial profiles, then we should remove cosmics and background from MW like so:

# cosmic_cleaned_MW = remove_cosmics(MW, ronmask, obsname, path, Flim=3.0, siglim=5.0, maxiter=1, savemask=False,
#                                   savefile=False, save_err=False, verbose=True, timit=True)
# bg_corrected_MW = remove_background(cosmic_cleaned_MW, P_id, obsname, path, degpol=5, slit_height=5, save_bg=False,
#                                    savefile=False, save_err=False, exclude_top_and_bottom=True, verbose=True,
#                                    timit=True)

## before doing the following:
# MW_stripes,MW_stripe_indices = extract_stripes(MW, P_id, return_indices=True, slit_height=30)
# err_MW_stripes = extract_stripes(err_MW, P_id, return_indices=False, slit_height=30)
# pix_MW_q,flux_MW_q,err_MW_q = extract_spectrum_from_indices(MW, err_MW, MW_stripe_indices, method='quick',
#                                                            slit_height=30, RON=ronmask, savefile=True, filetype='fits',
#                                                            obsname='master_white', path=path, timit=True)
# pix_MW,flux_MW,err_MW = extract_spectrum_from_indices(MW, err_MW, MW_stripe_indices, method='optimal',
#                                                      individual_fibres=True, slit_height=30, RON=ronmask,
#                                                      savefile=True, filetype='fits', obsname='master_white',
#                                                      path=path, timit=True)

# fp = fit_profiles(P_id, MW_stripes, err_MW_stripes, mask=mask, stacking=True, slit_height=5,
#                  model='gausslike', return_stats=True, timit=True)

## OR
# fp2 = fit_profiles_from_indices(P_id, MW, err_MW, MW_stripe_indices, mask=mask, stacking=True, slit_height=5,
#                                model='gausslike', return_stats=True, timit=True)

#######################################################################################################################

## (4) PROCESS SCIENCE IMAGES
## quick extraction method
dum = process_science_images(stellar_list, P_id, mask=mask, sampling_size=25, slit_height=7, gain=gain, MB=MB,
                             ronmask=ronmask, MD=MDS, scalable=True, saveall=False, path=path, ext_method='quick',
                             offset='True', slope='True', fibs='single', from_indices=True, timit=True,
                             fibparms=fibparms, simu=simu)

## optimal extraction method - DOESN'T WORK AS YET
# dum = process_science_images(stellar_list, P_id, mask=mask, sampling_size=25, slit_height=7, gain=gain, MB=MB,
#                             ronmask=ronmask, MD=MDS, scalable=True, saveall=False, path=path, ext_method='optimal',
#                             offset='True', slope='True', fibs='single', from_indices=True, timit=True)

#######################################################################################################################

# (5) calculate barycentric correction and append to FITS header

#######################################################################################################################

# (6) calculate RV

#######################################################################################################################