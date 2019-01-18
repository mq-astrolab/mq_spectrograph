# -*- coding: utf-8 -*-
import glob
import astropy.io.fits as pyfits
import numpy as np
import datetime

from veloce_reduction.helper_functions import short_filenames
from veloce_reduction.calibration import get_bias_and_readnoise_from_bias_frames, make_offmask_and_ronmask, make_master_bias_from_coeffs, make_master_dark, correct_orientation, crop_overscan_region
from veloce_reduction.order_tracing import find_stripes, make_P_id, make_mask_dict, extract_stripes #, find_tramlines
from veloce_reduction.process_scripts import process_whites, process_science_images
from veloce_reduction.spatial_profiles import fit_profiles, fit_profiles_from_indices 

import time
import os
import barycorrpy

from veloce_reduction.helper_functions import binary_indices
from veloce_reduction.calibration import correct_for_bias_and_dark_from_filename
from veloce_reduction.cosmic_ray_removal import remove_cosmics
from veloce_reduction.background import remove_background
from veloce_reduction.order_tracing import extract_stripes
from veloce_reduction.extraction import extract_spectrum, extract_spectrum_from_indices
from veloce_reduction.relative_intensities import get_relints, get_relints_from_indices, append_relints_to_FITS
from veloce_reduction.get_info_from_headers import get_obs_coords_from_header

from veloce_reduction.subtract_overscan import read_and_overscan_correct

import warnings
warnings.filterwarnings('ignore')


bias_list = ["/Users/Brendan/Dropbox/Brendan/Veloce/Data/veloce/180815/bias1.fits", "/Users/Brendan/Dropbox/Brendan/Veloce/Data/veloce/180815/bias2.fits", "/Users/Brendan/Dropbox/Brendan/Veloce/Data/veloce/180815/bias5.fits"]

s_list = glob.glob("/Users/Brendan/Dropbox/Brendan/Veloce/Data/veloce/180815/ccd_3/15aug*")
c_list = glob.glob("/Users/Brendan/Dropbox/Brendan/Veloce/Data/veloce/180815/ccd_3/*overscan_corrected*")
stellar_list = list(set(s_list)-set(c_list))

white_list = ['/Users/Brendan/Dropbox/Brendan/Veloce/Data/veloce/180813/flat_0.5.fits', '/Users/Brendan/Dropbox/Brendan/Veloce/Data/veloce/180813/flat_4.fits', '/Users/Brendan/Dropbox/Brendan/Veloce/Data/veloce/180813/flat_5.fits', '/Users/Brendan/Dropbox/Brendan/Veloce/Data/veloce/180813/flat_6.fits', '/Users/Brendan/Dropbox/Brendan/Veloce/Data/veloce/180813/flat.fits', '/Users/Brendan/Dropbox/Brendan/Veloce/Data/veloce/180813/flat_noap.fits', '/Users/Brendan/Dropbox/Brendan/Veloce/Data/veloce/180813/flat_noap_2.fits', '/Users/Brendan/Dropbox/Brendan/Veloce/Data/veloce/180813/flat_noap_3.fits']

path = '/Users/Brendan/Dropbox/Brendan/Veloce/Data/output3'

saveall = '/Users/Brendan/Dropbox/Brendan/Veloce/Data'

corrected_bias_list = []
for i in bias_list:
    corrected_bias_list.append(read_and_overscan_correct(i))

corrected_white_list = []
for i in white_list:
    corrected_white_list.append(read_and_overscan_correct(i))

corrected_stellar_list = []
for i in stellar_list:
    corrected_stellar_list.append(read_and_overscan_correct(i))

print(corrected_stellar_list)

#print("pyfits.getdata(bias_list[0].shape", pyfits.getdata(bias_list[0]).shape)
#print("pyfits.getdata(corrected_bias_list[0].shape", pyfits.getdata(corrected_bias_list[0]).shape)

gain = [0.88, 0.93, 0.99, 0.93] #(depending on the camera setup)
nx=4112
ny=4096
medbias,coeffs,offsets,rons = get_bias_and_readnoise_from_bias_frames(corrected_bias_list, degpol=5, clip=5., gain=gain, debug_level=0, timit=True)
# create MASTER BIAS frame and read-out noise mask (units = electrons)
offmask,ronmask = make_offmask_and_ronmask(offsets, rons, nx, ny, gain=gain, savefiles=True, path=path, timit=True)
MB = make_master_bias_from_coeffs(coeffs, nx, ny, savefile=True, path=path, timit=True)
#we did not have darks, so I did this
MD = np.zeros(MB.shape)

#create (bias- & dark-subtracted) MASTER WHITE frame and corresponding error array (units = ADUs)
MW,err_MW = process_whites(white_list, corrected_white_list, MB=MB, ronmask=ronmask, MD=MD, gain=gain, scalable=False, fancy=False, clip=5., savefile=True, saveall=True, diffimg=False, path=path, timit=False)
# find orders
P,tempmask = find_stripes(MW, deg_polynomial=2, min_peak=0.05, gauss_filter_sigma=3., simu=False)
# assign physical diffraction order numbers (this is only a dummy function for now) to order-fit polynomials and bad-region masks
P_id = make_P_id(P)
mask = make_mask_dict(tempmask)
 
#loop over all files you want to extract a 1-dim spectrum for
for filename in corrected_stellar_list:
    #do some housekeeping with filenames
    dum = filename.split('/')
    dum2 = dum[-1].split('.')
    obsname = dum2[0]
 
    # (1) call routine that does all the bias and dark correction stuff and proper error treatment
    img = correct_for_bias_and_dark_from_filename(filename, MB, MD, gain=gain, scalable=False, savefile=True, path=path, timit=True)   #[e-]
    err_img = np.sqrt(np.clip(img,0,None) + ronmask*ronmask)   # [e-]
 
    # (5) extract stripes
    stripes,stripe_indices = extract_stripes(img, P_id, return_indices=True, slit_height=25, savefiles=True, obsname=obsname, path=path, timit=True)
    pix,flux,err = extract_spectrum_from_indices(img, err_img, stripe_indices, method="quick", slit_height=25, RON=ronmask, savefile=True, 
                                                 filetype='fits', obsname=obsname, path=path, timit=True)