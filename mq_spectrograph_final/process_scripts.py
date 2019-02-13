'''
Created on 25 Jul. 2018

@author: Christoph Bergmann
'''

import astropy.io.fits as pyfits
import numpy as np
import time
import scipy
from scipy import ndimage
# import os

from helper_functions import binary_indices
from calibration import correct_for_bias_and_dark_from_filename
# from basic_reduction.cosmic_ray_removal import remove_cosmics
# from basic_reduction.background import remove_background
from order_tracing import extract_stripes
from extraction import extract_spectrum, extract_spectrum_from_indices
# from basic_reduction.relative_intensities import get_relints, get_relints_from_indices, append_relints_to_FITS
# from basic_reduction.get_info_from_headers import get_obs_coords_from_header




def process_whites(white_list, MB=None, ronmask=None, MD=None, gain=None, scalable=False, fancy=False, clip=5., savefile=True, saveall=False, diffimg=False, path=None, debug_level=0, timit=False):
    """
    This routine processes all whites from a given list of file. It corrects the orientation of the image and crops the overscan regions,
    and subtracts both the MASTER BIAS frame [in ADU], and the MASTER DARK frame [in e-] from every image before combining them to create a MASTER WHITE frame.
    It currently does NOT do cosmic-ray removal or background subtraction!!!
    NOTE: the input image has units of ADU, but the output image has units of electrons!!!
    
    INPUT:
    'white_list'  : list of filenames of raw white images (incl. directories)
    'MB'          : the master bias frame [ADU]
    'ronmask'     : the read-noise mask (or frame) [e-]
    'MD'          : the master dark frame [e-]
    'gain'        : the gains for each quadrant [e-/ADU]
    'scalable'    : boolean - do you want to normalize the dark current to an exposure time of 1s? (ie do you want to make it "scalable"?)
    'fancy'       : boolean - do you want to use the 'fancy' method for creating the master white frame? (otherwise a simple median image will be used)
    'clip'        : number of 'expected-noise sigmas' a pixel has to deviate from the median pixel value across all images to be considered an outlier when using the 'fancy' method
    'savefile'    : boolean - do you want to save the master white frame as a FITS file?
    'saveall'     : boolean - do you want to save all individual bias- & dark-corrected images as well?
    'diffimg'     : boolean - do you want to save the difference image (ie containing the outliers)? only used if 'fancy' is set to TRUE
    'path'        : path to the output file directory (only needed if savefile is set to TRUE)
    'debug_level' : for debugging...
    'timit'       : boolean - do you want to measure execution run time?
    
    OUTPUT:
    'master'      : the master white image [e-] (also has been brought to 'correct' orientation and overscan regions cropped) 
    'err_master'  : the corresponding uncertainty array [e-]    
    """
    
    if timit:
        start_time = time.time()

    if debug_level >= 1:
        print('Creating master white frame from '+str(len(white_list))+' fibre flats...')

    #if INPUT arrays are not given, read them from default files
    if path is None:
        print('WARNING: output file directory not provided!!!')
        print('Using same directory as input file...')
        dum = white_list[0].split('/')
        path = white_list[0][0:-len(dum[-1])]
    if MB is None:
        #no need to fix orientation, this is already a processed file [ADU]
        MB = pyfits.getdata(path+'master_bias.fits')
    if ronmask is None:
        #no need to fix orientation, this is already a processed file [e-]
        ronmask = pyfits.getdata(path+'read_noise_mask.fits')
    if MD is None:
        if scalable:
            #no need to fix orientation, this is already a processed file [e-]
            MD = pyfits.getdata(path+'master_dark_scalable.fits', 0)
#             err_MD = pyfits.getdata(path+'master_dark_scalable.fits', 1)
        else:
            #no need to fix orientation, this is already a processed file [e-]
            MD = pyfits.getdata(path+'master_dark_t'+str(int(np.round(texp,0)))+'.fits', 0)
#             err_MD = pyfits.getdata(path+'master_dark_t'+str(int(np.round(texp,0)))+'.fits', 1)


    #prepare arrays
    allimg = []
    allerr = []

    #loop over all files in "white_list"; correct for bias and darks on the fly
    for n,fn in enumerate(sorted(white_list)):
        if debug_level >=1:
            print('Now processing file: '+str(fn))

        # if the darks have a different exposure time than the whites, then we need to re-scale the master dark
        try:
            texp = pyfits.getval(white_list[0], 'TOTALEXP')
        except:
            texp = 1.

        #call routine that does all the bias and dark correction stuff and converts from ADU to e-
        # if scalable:
        #     img = correct_for_bias_and_dark_from_filename(fn, MB, MD*texp, gain=gain, scalable=scalable, savefile=saveall,
        #                                                   path=path, timit=timit)     #these are now bias- & dark-corrected images; units are e-
        # else:
        #     img = correct_for_bias_and_dark_from_filename(fn, MB, MD, gain=gain, scalable=scalable, savefile=saveall,
        #                                                   path=path, timit=timit)  # these are now bias- & dark-corrected images; units are e-
        img = pyfits.getdata(fn)

        if debug_level >=2:
            print('min(img) = '+str(np.min(img)))
        allimg.append(img)
#         allerr.append(err)
#         allerr.append( np.sqrt(img + ronmask*ronmask) )   # [e-]
        #dumb fix for negative pixel values that can occur, if we haven't masked out bad pixels yet
        allerr.append( np.sqrt(np.abs(img) + ronmask*ronmask) )   # [e-]


    #########################################################################
    ### now we do essentially what "CREATE_MASTER_IMG" does for whites... ###
    #########################################################################
    #add individual-image errors in quadrature (need it either way, not only for fancy method)
    err_summed = np.sqrt(np.sum((np.array(allerr)**2),axis=0))
    #get median image
    medimg = np.median(np.array(allimg), axis=0)

    if fancy:
        #need to create a co-added frame if we want to do outlier rejection the fancy way
        summed = np.sum((np.array(allimg)),axis=0)
        if diffimg:
            diff = np.zeros(summed.shape)

        master_outie_mask = np.zeros(summed.shape, dtype='int')

        #make sure we do not have any negative pixels for the sqrt
        medimgpos = medimg.copy()
        medimgpos[medimgpos < 0] = 0.
        med_sig_arr = np.sqrt(medimgpos + ronmask*ronmask)       #expected STDEV for the median image (from LB Eq 2.1); still in ADUs
        for n,img in enumerate(allimg):
            #outie_mask = np.abs(img - medimg) > clip*med_sig_arr
            outie_mask = (img - medimg) > clip*med_sig_arr      #do we only want HIGH outliers, ie cosmics?
            #save info about which image contributes the outlier pixel using unique binary numbers technique
            master_outie_mask += (outie_mask * 2**n).astype(int)
        #see which image(s) produced the outlier(s) and replace outies by mean of pixel value from remaining images
        n_outie = np.sum(master_outie_mask > 0)
        print('Correcting '+str(n_outie)+' outliers...')
        #loop over all outliers
        for i,j in zip(np.nonzero(master_outie_mask)[0],np.nonzero(master_outie_mask)[1]):
            #access binary numbers and retrieve component(s)
            outnum = binary_indices(master_outie_mask[i,j])   #these are the indices (within allimg) of the images that contain outliers
            dumix = np.arange(len(white_list))
            #remove the images containing the outliers in order to compute mean from the remaining images
            useix = np.delete(dumix,outnum)
            if diffimg:
                diff[i,j] = summed[i,j] - ( len(outnum) * np.mean( np.array([allimg[q][i,j] for q in useix]) ) + np.sum( np.array([allimg[q][i,j] for q in useix]) ) )
            #now replace value in master image by the sum of all pixel values in the unaffected pixels
            #plus the number of affected images times the mean of the pixel values in the unaffected images
            summed[i,j] = len(outnum) * np.mean( np.array([allimg[q][i,j] for q in useix]) ) + np.sum( np.array([allimg[q][i,j] for q in useix]) )
        #once we have finished correcting the outliers, we want to "normalize" (ie divide by number of frames) the master image and the corresponding error array
        master = summed / len(white_list)
        err_master = err_summed / len(white_list)
    else:
        #ie not fancy, just take the median image to remove outliers
        medimg = np.median(np.array(allimg), axis=0)
        #now set master image equal to median image
        master = medimg.copy()
        #estimate of the corresponding error array (estimate only!!!)
        err_master = err_summed / len(white_list)


    #now save master white to file
    if savefile:
        outfn = path+'master_white.fits'
        pyfits.writeto(outfn, master, clobber=True)
        pyfits.setval(outfn, 'HISTORY', value='   MASTER WHITE frame - created '+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+' (GMT)')
        # pyfits.setval(outfn, 'EXPTIME', value=texp, comment='exposure time [s]')
        pyfits.setval(outfn, 'UNITS', value='ELECTRONS')
        if fancy:
            pyfits.setval(outfn, 'METHOD', value='fancy', comment='method to create master white and remove outliers')
        else:
            pyfits.setval(outfn, 'METHOD', value='median', comment='method to create master white and remove outliers')
        h = pyfits.getheader(outfn)
        h_err = h.copy()
        h_err['HISTORY'] = 'estimated uncertainty in MASTER WHITE frame - created '+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+' (GMT)'
        pyfits.append(outfn, err_master, h_err, clobber=True)

    #also save the difference image if desired
    if diffimg:
        hdiff = h.copy()
        hdiff['HISTORY'] = '   MASTER WHITE DIFFERENCE IMAGE - created '+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+' (GMT)'
        pyfits.writeto(path+'master_white_diffimg.fits', diff, hdiff, clobber=True)

    if timit:
        print('Total time elapsed: '+str(np.round(time.time() - start_time,1))+' seconds')

    return master, err_master





def process_science_images(imglist, P_id, mask=None, sampling_size=25, slit_height=25, gain=[1.,1.,1.,1.], MB=None, ronmask=None, MD=None, scalable=False, saveall=False, path=None, ext_method='optimal', 
                           from_indices=True, slope=True, offset=True, fibs='all', timit=False, fibparms=None, simu=False):
    """
    Process all science images. This includes:
    
    (1) bias and dark subtraction
    (2) cosmic ray removal 
    (3) background extraction and estimation
    (4) flat-fielding (ie removal of pixel-to-pixel sensitivity variations)
    =============================
    (5) extraction of stripes
    (6) extraction of 1-dim spectra
    (7) get relative intensities of different fibres
    (8) wavelength solution
    (9) barycentric correction
    """
    
    if timit:
        start_time = time.time()
    
    #####################################
    ### (1) bias and dark subtraction ###
    #####################################
    
    #if INPUT arrays are not given, read them from default files
    if path is None:
        print('WARNING: output file directory not provided!!!')
        print('Using same directory as input file...')
        dum = imglist[0].split('/')
        path = imglist[0][0:-len(dum[-1])]
    if MB is None:
        #no need to fix orientation, this is already a processed file [ADU]
        MB = pyfits.getdata(path+'master_bias.fits')
    if ronmask is None:
        #no need to fix orientation, this is already a processed file [e-]
        ronmask = pyfits.getdata(path+'read_noise_mask.fits')
    if MD is None:
        if scalable:
            #no need to fix orientation, this is already a processed file [e-]
            MD = pyfits.getdata(path+'master_dark_scalable.fits', 0)
#             err_MD = pyfits.getdata(path+'master_dark_scalable.fits', 1)
        else:
            #no need to fix orientation, this is already a processed file [e-]
            MD = pyfits.getdata(path+'master_dark_t'+str(int(np.round(texp,0)))+'.fits', 0)
#             err_MD = pyfits.getdata(path+'master_dark_t'+str(int(np.round(texp,0)))+'.fits', 1)
    
    if not from_indices:
        ron_stripes = extract_stripes(ronmask, P_id, return_indices=False, slit_height=slit_height, savefiles=False, timit=True)
    
    for i,filename in enumerate(sorted(imglist)):

        print('Extracting stellar spectrum '+str(i+1)+'/'+str(len(imglist)))

        #do some housekeeping with filenames
        dum = filename.split('/')
        dum2 = dum[-1].split('.')
        obsname = dum2[0]
              
        # (1) call routine that does all the bias and dark correction stuff and proper error treatment
        img = correct_for_bias_and_dark_from_filename(filename, MB, MD, gain=gain, scalable=scalable, savefile=saveall, path=path, timit=True, simu=simu)
        #err = np.sqrt(img + ronmask*ronmask)   # [e-]
        #TEMPFIX:
        err_img = np.sqrt(np.clip(img,0,None) + ronmask*ronmask)   # [e-]
        
        # (2) remove cosmic rays (ERRORS REMAIN UNCHANGED)
        # cosmic_cleaned_img = remove_cosmics(img, ronmask, obsname, path, Flim=3.0, siglim=5.0, maxiter=1, savemask=True, savefile=True, save_err=False, verbose=True, timit=True)   # [e-]
        #adjust errors?
        
        # (3) fit and remove background (ERRORS REMAIN UNCHANGED)
        # bg_corrected_img = remove_background(cosmic_cleaned_img, P_id, obsname, path, degpol=5, slit_height=slit_height, save_bg=True, savefile=True, save_err=False,
        #                                      exclude_top_and_bottom=True, verbose=True, timit=True)   # [e-]
        # bg_corrected_img = remove_background(img, P_id, obsname, path, degpol=5, slit_height=slit_height, save_bg=False, savefile=True, save_err=False,
        #                                      exclude_top_and_bottom=True, verbose=True, timit=True)   # [e-]
        #adjust errors?

        # (4) remove pixel-to-pixel sensitivity variations (2-dim)
        #XXXXXXXXXXXXXXXXXXXXXXXXXXX
        #TEMPFIX
        # final_img = bg_corrected_img.copy()   # [e-]
        final_img = img.copy()   # [e-]
        #adjust errors?

        # (5) extract stripes
        stripes,stripe_indices = extract_stripes(final_img, P_id, return_indices=True, slit_height=slit_height, savefiles=saveall, obsname=obsname, path=path, timit=True)
        if not from_indices:
            err_stripes = extract_stripes(err_img, P_id, return_indices=False, slit_height=slit_height, savefiles=saveall, obsname=obsname+'_err', path=path, timit=True)

        # (6) perform extraction of 1-dim spectrum
        if from_indices:
            pix,flux,err = extract_spectrum_from_indices(final_img, err_img, stripe_indices, method='quick', slit_height=slit_height, RON=ronmask, savefile=True,
                                                         filetype='fits', obsname=obsname, path=path, timit=True, fibparms=fibparms)
            pix,flux,err = extract_spectrum_from_indices(final_img, err_img, stripe_indices, method=ext_method, slope=slope, offset=offset, fibs=fibs, slit_height=slit_height,
                                                         RON=ronmask, savefile=True, filetype='fits', obsname=obsname, path=path, timit=True, simu=True, fibpos='05', fibparms=fibparms) # added simu=True and fibpos='05'
        else:
            pix2,flux2,err2 = extract_spectrum(stripes, err_stripes=err_stripes, ron_stripes=ron_stripes, method=ext_method, slope=slope, offset=offset, fibs=fibs, 
                                               slit_height=slit_height, RON=ronmask, savefile=False, filetype='fits', obsname=obsname, path=path, timit=True, fibparms=fibparms)
    

#         # (8) get wavelength solution
#         #XXXXX


        # # (9) get barycentric correction
        # bc = get_barycentric_correction(filename)
        # outfn = path + obsname + '_extracted.fits'
        # pyfits.setval(filename, 'BARYCORR', value=bc, comment='barycentric velocity correction [m/s]')

#         lat, long, alt = get_obs_coords_from_header(fn)    # not really necessary, obsname='AAO' does the trick (agree to within ~0.01 cm/s!!!
#         utmjd = pyfits.getval(fn, 'UTMJD') + 2.4e6
#         ra = pyfits.getval(fn, 'MEANRA')
#         dec = pyfits.getval(fn, 'MEANDEC')
#         # HMMM...using hip_id=xxx and actual coordinates from header makes a huge difference (~11m/s for the tau Ceti example I tried)!!!
#         bc1 = barycorrpy.get_BC_vel(JDUTC=utmjd, hip_id=8102, obsname='AAO', ephemeris='de430')
#         bc2 = barycorrpy.get_BC_vel(JDUTC=utmjd, ra=ra, dec=dec, obsname='AAO', ephemeris='de430')
#
#         #now append relints, wl-solution, and barycorr to extracted FITS file header
#         outfn = path + obsname + '_extracted.fits'
#         if os.path.isfile(outfn):
#             #relative fibre intensities
#             dum = append_relints_to_FITS(relints, outfn, nfib=19)
#             #wavelength solution
#             #pyfits.setval(fn, 'RELINT' + str(i + 1).zfill(2), value=relints[i], comment='fibre #' + str(fibnums[i]) + ' - ' + fibinfo[i] + ' fibre')
#             #barycentric correction
#             pyfits.setval(outfn, 'BARYCORR', value=np.array(bc[0])[0], comment='barycentric correction [m/s]')


    if timit:
        print('Total time elapsed: '+str(np.round(time.time() - start_time,1))+' seconds')
    
    return






























