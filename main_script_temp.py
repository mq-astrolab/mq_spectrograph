'''
Created on 5 Oct. 2018

@author: christoph
'''


'''
Created on 25 Jul. 2018

@author: christoph
'''

import glob
import astropy.io.fits as pyfits
import numpy as np
import datetime
import copy

from veloce_reduction.veloce_reduction.get_info_from_headers import get_obstype_lists_temp
from veloce_reduction.veloce_reduction.helper_functions import short_filenames
from veloce_reduction.veloce_reduction.calibration import get_bias_and_readnoise_from_bias_frames, make_offmask_and_ronmask, make_master_bias_from_coeffs, make_master_dark, correct_orientation, crop_overscan_region
from veloce_reduction.veloce_reduction.order_tracing import find_stripes, make_P_id, make_mask_dict, extract_stripes #, find_tramlines
from veloce_reduction.veloce_reduction.spatial_profiles import fit_profiles, fit_profiles_from_indices
from veloce_reduction.veloce_reduction.extraction import *
from veloce_reduction.process_scripts import process_whites, process_science_images



# desktop:
path = '/Volumes/BERGRAID/data/veloce/raw_goodonly/20180918/'
# laptop:
path = '/Users/christoph/data/raw_goodonly/20180919/'





# (0) INFO FROM FITS HEADERS ########################################################################################################################
#####TEMP#####
# bias_list,dark_list,white_list,thar_list,thxe_list,laser_list,stellar_list = identify_obstypes(path)
# bias_list = glob.glob(path + 'Bias*.fits')
# dark_list = glob.glob(path + 'Dark*.fits')
# white_list = glob.glob(path + 'Light*.fits')
# stellar_list = glob.glob(path + 'Light*.fits')
###END TEMP###
acq_list, bias_list, dark_list, flat_list, skyflat_list, domeflat_list, arc_list, thxe_list, laser_list, laser_and_thxe_list, stellar_list, unknown_list = get_obstype_lists_temp(path)
assert len(unknown_list) == 0, "WARNING: unknown files encountered!!!"
obsnames = short_filenames(stellar_list)
dumimg = crop_overscan_region(correct_orientation(pyfits.getdata(stellar_list[0])))
ny,nx = dumimg.shape
del dumimg
#####################################################################################################################################################



# # (1) BAD PIXEL MASK ################################################################################################################################
# bpm_list = glob.glob(path + '*bad_pixel_mask*')
# #read most recent bad pixel mask
# bpm_dates = [x[-12:-4] for x in bpm_list]
# most_recent_datestring = sorted(bpm_dates)[-1]
# bad_pixel_mask = np.load(path + 'bad_pixel_mask_' + most_recent_datestring + '.npy')
# 
# # update the pixel mask
# #blablabla
# 
# #save updated bad pixel mask
# now = datetime.datetime.now()
# dumstring = str(now)[:10].split('-')
# datestring = ''.join(dumstring)
# np.save(path+'bad_pixel_mask_'+datestring+'.npy', bad_pixel_mask)
# #####################################################################################################################################################



# (2) CALIBRATION ###################################################################################################################################
# gain = [0.88, 0.93, 0.99, 0.93]   # from "VELOCE_DETECTOR_REPORT_V1.PDF"
# gain = [1., 1., 1., 1.]
gain = [1., 1.095, 1.125, 1.]   # eye-balled from extracted flat fields
# (i) BIAS 
# get offsets and read-out noise
#either from bias frames (units: [offsets] = ADUs; [RON] = e-)
medbias,coeffs,offsets,rons = get_bias_and_readnoise_from_bias_frames(sorted(bias_list)[0:9], degpol=5, clip=5., gain=gain, save_medimg=True, debug_level=1, timit=True)
#or from the overscan regions

# create MASTER BIAS frame and read-out noise mask (units = ADUs)
offmask,ronmask = make_offmask_and_ronmask(offsets, rons, nx, ny, gain=gain, savefiles=True, path=path, timit=True)
MB = make_master_bias_from_coeffs(coeffs, nx, ny, savefile=True, path=path, timit=True)
# or
# MB = offmask.copy()
# #or
# MB = medbias.copy()
#XXXalso save read-noise and offsets for all headers to write later!?!?!?


# (ii) DARKS
# create (bias-subtracted) MASTER DARK frame (units = electrons)
# MD = make_master_dark(dark_list, MB=MB, gain=gain, scalable=False, savefile=True, path=path, timit=True)
MDS = make_master_dark(dark_list, MB=medbias, gain=gain, scalable=True, savefile=True, path=path, debug_level=1, timit=True)
# MDS = np.zeros(MB.shape)

# (iii) WHITES 
#create (bias- & dark-subtracted) MASTER WHITE frame and corresponding error array (units = electrons)
MW,err_MW = process_whites(flat_list, MB=medbias, ronmask=ronmask, MD=MDS, gain=gain, scalable=True, fancy=False,
                           clip=5., savefile=True, saveall=False, diffimg=False, path=path, debug_level=1, timit=False)
#####################################################################################################################################################



# (3) ORDER TRACING #################################################################################################################################
# find orders roughly
#P,tempmask = find_stripes(MW, deg_polynomial=2, min_peak=0.05, gauss_filter_sigma=3., simu=False)
P,tempmask = find_stripes(MW, deg_polynomial=2, min_peak=0.05, gauss_filter_sigma=10., simu=False, maskthresh = 400)
# assign physical diffraction order numbers (this is only a dummy function for now) to order-fit polynomials and bad-region masks
P_id_dum = make_P_id(P)
mask = make_mask_dict(tempmask)
P_id = copy.deepcopy(P_id_dum)
for o in P_id.keys():
    P_id[o][0] -= 2.
np.save(path + 'P_id.npy', P_id)

# extract stripes of user-defined width from the science image, centred on the polynomial fits defined in step (1)
MW_stripes,MW_indices = extract_stripes(MW, P_id, return_indices=True, slit_height=30)
pix,flux,err = extract_spectrum_from_indices(MW, err_MW, MW_indices, method='quick', slit_height=30, RON=ronmask,
                                             savefile=True, filetype='fits', obsname='master_white', path=path, timit=True)
pix,flux,err = extract_spectrum_from_indices(MW, err_MW, MW_indices, method='optimal', slope=True, offset=True, fibs='all', slit_height=30,
                                             RON=ronmask, savefile=True, filetype='fits', obsname='master_white', path=path, timit=True)
#####################################################################################################################################################


###
#if we want to determine spatial profiles, then we should remove cosmics and background from MW like so:

# cosmic_cleaned_MW = remove_cosmics(MW, ronmask, obsname, path, Flim=3.0, siglim=5.0, maxiter=1, savemask=False, savefile=False, save_err=False, verbose=True, timit=True)
# bg_corrected_MW = remove_background(cosmic_cleaned_MW, P_id, obsname, path, degpol=5, slit_height=5, save_bg=False, savefile=False, save_err=False, exclude_top_and_bottom=True, verbose=True, timit=True)
#before doing the following:
# MW_stripes,MW_stripe_indices = extract_stripes(MW, P_id, return_indices=True, slit_height=30)
# err_MW_stripes = extract_stripes(err_MW, P_id, return_indices=False, slit_height=30)
# pix_MW_q,flux_MW_q,err_MW_q = extract_spectrum_from_indices(MW, err_MW, MW_stripe_indices, method='quick', slit_height=30, RON=ronmask, savefile=True,
#                                                          filetype='fits', obsname='master_white', path=path, timit=True)
# pix_MW,flux_MW,err_MW = extract_spectrum_from_indices(MW, err_MW, MW_stripe_indices, method='optimal', individual_fibres=True, slit_height=30, RON=ronmask, savefile=True,
#                                                          filetype='fits', obsname='master_white', path=path, timit=True)

# fp = fit_profiles(P_id, MW_stripes, err_MW_stripes, mask=mask, stacking=True, slit_height=5, model='gausslike', return_stats=True, timit=True)
# #OR
# fp2 = fit_profiles_from_indices(P_id, MW, err_MW, MW_stripe_indices, mask=mask, stacking=True, slit_height=5, model='gausslike', return_stats=True, timit=True)
# ###



# (4) PROCESS SCIENCE IMAGES
dum = process_science_images(arc_list, P_id, mask=mask, sampling_size=25, slit_height=30, gain=gain, MB=medbias, ronmask=ronmask, MD=MDS, scalable=True,
                             saveall=False, path=path, ext_method='optimal', offset='True', slope='True', fibs='all', from_indices=True, timit=True)
dum = process_science_images(stellar_list, P_id, mask=mask, sampling_size=25, slit_height=24, gain=gain, MB=medbias, ronmask=ronmask, MD=MDS, scalable=True, 
                             saveall=False, path=path, ext_method='optimal', offset='True', slope='True', fibs='stellar', from_indices=True, timit=True)



# (5) calculate barycentric correction and append to FITS header



# (6) calculate RV













