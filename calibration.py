"""
Created on 13 Apr. 2018

@author: Christoph Bergmann
"""

import astropy.io.fits as pyfits
import numpy as np
from itertools import combinations
import time
import matplotlib.pyplot as plt

from helper_functions import correct_orientation, sigma_clip, polyfit2d, polyval2d





def make_median_image(imglist, MB=None, scalable=False, raw=False):
    """
    Make a median image from a given list of images.

    INPUT:
    'imglist'  : list of files (incl. directories)
    'MB'       : master bias frame - if provided, it will be subtracted from every image before median image is computed
    'scalable' : boolean - do you want to scale this to an exposure time of 1s (AFTER the bias is subtracted!!!!!)
    'raw'      : boolean - set to TRUE if you want to retain the original size and orientation;
                 otherwise the image will be brought to the 'correct' orientation and the overscan regions will be cropped

    OUTPUT:
    'medimg'   : median image
    """

    # from veloce_reduction.calibration import crop_overscan_region

    # prepare array
    allimg = []

    # loop over all files in "dark_list"
    for file in imglist:
        
        # read in dark image
        img = pyfits.getdata(file)
        if not raw:
            # bring to "correct" orientation
            img = correct_orientation(img)
            # remove the overscan region
            img = crop_overscan_region(img)
        if MB is not None:
            # subtract master bias (if provided)
            img = img - MB
        if scalable:
            texp = pyfits.getval(file, 'TOTALEXP')
            img /= texp

        # add image to list
        allimg.append(img)

    # get median image
    medimg = np.median(np.array(allimg), axis=0)

    return medimg





def make_quadrant_masks(nx, ny):
    # define four quadrants via masks
    q1 = np.zeros((ny, nx), dtype='bool')
    q1[:int(ny / 2), :int(nx / 2)] = True
    q2 = np.zeros((ny, nx), dtype='bool')
    q2[:int(ny / 2), int(nx / 2):] = True
    q3 = np.zeros((ny, nx), dtype='bool')
    q3[int(ny / 2):, int(nx / 2):] = True
    q4 = np.zeros((ny, nx), dtype='bool')
    q4[int(ny / 2):, :int(nx / 2)] = True

    return q1, q2, q3, q4





def crop_overscan_region(img):
    """
    As of July 2018, Veloce uses an e2v CCD231-84-1-E74 4kx4k chip.
    Image dimensions are 4096 x 4112 pixels, but the recorded images size including the overscan region is 4202 x 4112 pixels.
    We therefore have an overscan region of size 53 x 4112 at either end. 
    
    raw_img = pyfits.getdata(filename)     -->    raw_img.shape = (4112, 4202)
    img = correct_orientation(raw_img)     -->        img.shape = (4202, 4112)
    """

    #correct orientation if needed
    if img.shape == (4112, 4202):
        img = correct_orientation(img)

    if img.shape != (4202, 4112):
        print('ERROR: wrong image size encountered!!!')
        return

    #crop overscan region
    good_img = img[53:4149,:]

    return good_img





def extract_overscan_region(img):
    """
    As of July 2018, Veloce uses an e2v CCD231-84-1-E74 4kx4k chip.
    Image dimensions are 4096 x 4112 pixels, but the recorded images size including the overscan region is 4202 x 4112 pixels.
    We therefore have an overscan region of size 53 x 4112 at either end. 
    
    raw_img = pyfits.getdata(filename)     -->    raw_img.shape = (4112, 4202)
    img = correct_orientation(raw_img)     -->        img.shape = (4202, 4112)
    """

    #correct orientation if needed
    if img.shape == (4112, 4202):
        img = correct_orientation(img)

    if img.shape != (4202, 4112):
        print('ERROR: wrong image size encountered!!!')
        return

    ny,nx = img.shape

    #define overscan regions
    os1 = img[:53,:nx//2]
    os2 = img[:53,nx//2:]
    os3 = img[ny-53:,nx//2:]
    os4 = img[ny-53:,:nx//2]   
    
    return os1,os2,os3,os4





def get_bias_and_readnoise_from_bias_frames(bias_list, degpol=5, clip=5, gain=None, save_medimg=True, debug_level=0, timit=False):
    """
    Calculate the median bias frame, the offsets in the four different quadrants (assuming bias frames are flat within a quadrant),
    and the read-out noise per quadrant (ie the STDEV of the signal, but from difference images).
    
    INPUT:
    'bias_list'    : list of raw bias image files (incl. directories)
    'degpol'       : order of the polynomial (in each direction) to be used in the 2-dim polynomial surface fits to each quadrant's median bais frame
    'clip'         : number of 'sigmas' used to identify outliers when 'cleaning' each quadrant's median bais frame before the surface fitting
    'gain'         : array of gains for each quadrant (in units of e-/ADU)
    'save_medimg'  : boolean - do you want to save the median image to a FITS file?
    'debug_level'  : for debugging...
    'timit'        : boolean - do you want to measure execution run time?
    
    OUTPUT:
    'medimg'   : the median bias frame [ADU]
    'coeffs'   : the coefficients that describe the 2-dim polynomial surface fit to the 4 quadrants
    'offsets'  : the 4 constant offsets per quadrant (assuming bias frames are flat within a quadrant) [ADU]
    'rons'     : read-out noise for the 4 quadrants [e-]
    """
    
    if timit:
        start_time = time.time()

    print('Determining offset levels and read-out noise properties from bias frames for 4 quadrants...')

    img = pyfits.getdata(bias_list[0])

    #do some formatting things for real observations
    #bring to "correct" orientation
    img = correct_orientation(img)
    #remove the overscan region, which looks crap for actual bias images
    img = crop_overscan_region(img)


    ny,nx = img.shape

    #define four quadrants via masks
    q1,q2,q3,q4 = make_quadrant_masks(nx,ny)

    #co-add all bias frames
    #MB = create_master_img(bias_list, imgtype='bias', with_errors=False, savefiles=False, remove_outliers=True)

    #prepare arrays
    #means_q1 = []
    medians_q1 = []
    sigs_q1 = []
    #means_q2 = []
    medians_q2 = []
    sigs_q2 = []
    #means_q3 = []
    medians_q3 = []
    sigs_q3 = []
    #means_q4 = []
    medians_q4 = []
    sigs_q4 = []
    allimg = []

    if debug_level >= 1:
        print('Determining bias levels and read-out noise from '+str(len(bias_list))+' bias frames...')

    #first get mean / median for all bias images (per quadrant)
    for name in bias_list:
        
        if debug_level >= 1:
            print('OK, reading ',name)
        
        img = pyfits.getdata(name)
        
        #bring to "correct" orientation
        img = correct_orientation(img)
        #remove the overscan region, which looks crap for actual bias images
        img = crop_overscan_region(img)
        
        #means_q1.append(np.nanmean(img[q1]))
        medians_q1.append(np.nanmedian(img[q1]))
        #means_q2.append(np.nanmean(img[q2]))
        medians_q2.append(np.nanmedian(img[q2]))
        #means_q3.append(np.nanmean(img[q3]))
        medians_q3.append(np.nanmedian(img[q3]))
        #means_q4.append(np.nanmean(img[q4]))
        medians_q4.append(np.nanmedian(img[q4]))
        allimg.append(img)

    # now get sigma of RON for ALL DIFFERENT COMBINATIONS of length 2 of the images in 'bias_list'
    # by using the difference images we are less susceptible to funny pixels (hot, warm, cosmics, etc.)
    list_of_combinations = list(combinations(bias_list, 2))
    for (name1,name2) in list_of_combinations:
        
        # read in observations and bring to right format
        img1 = pyfits.getdata(name1)
        img2 = pyfits.getdata(name2)

        img1 = correct_orientation(img1)
        img1 = crop_overscan_region(img1)
        img2 = correct_orientation(img2)
        img2 = crop_overscan_region(img2)

        #take difference and do sigma-clipping
        diff = img1.astype(long) - img2.astype(long)
        sigs_q1.append(np.nanstd(sigma_clip(diff[q1], 5))/np.sqrt(2))
        sigs_q2.append(np.nanstd(sigma_clip(diff[q2], 5))/np.sqrt(2))
        sigs_q3.append(np.nanstd(sigma_clip(diff[q3], 5))/np.sqrt(2))
        sigs_q4.append(np.nanstd(sigma_clip(diff[q4], 5))/np.sqrt(2))

    #now average over all images
    #allmeans = np.array([np.median(medians_q1), np.median(medians_q2), np.median(medians_q3), np.median(medians_q4)])
    offsets = np.array([np.median(medians_q1), np.median(medians_q2), np.median(medians_q3), np.median(medians_q4)])
    rons = np.array([np.median(sigs_q1), np.median(sigs_q2), np.median(sigs_q3), np.median(sigs_q4)])
    
    #get median image as well
    medimg = np.median(np.array(allimg), axis=0)
    
    ##### now fit a 2D polynomial surface to the median bias image (for each quadrant separately)
        
    #now, because all quadrants are the same size, they have the same "normalized coordinates", so only have to do that once
    xq1 = np.arange(0,(nx/2))
    yq1 = np.arange(0,(ny/2))
    XX_q1,YY_q1 = np.meshgrid(xq1,yq1)
    x_norm = (XX_q1.flatten() / ((len(xq1)-1)/2.)) - 1.
    y_norm = (YY_q1.flatten() / ((len(yq1)-1)/2.)) - 1.
    
    #Quadrant 1
    medimg_q1 = medimg[:(ny/2), :(nx/2)]
    #clean this, otherwise the surface fit will be rubbish
    medimg_q1[np.abs(medimg_q1 - np.median(medians_q1)) > clip * np.median(sigs_q1)] = np.median(medians_q1)
    coeffs_q1 = polyfit2d(x_norm, y_norm, medimg_q1.flatten(), order=degpol)
    
    #Quadrant 2
    medimg_q2 = medimg[:(ny/2), (nx/2):]
    #clean this, otherwise the surface fit will be rubbish
    medimg_q2[np.abs(medimg_q2 - np.median(medians_q2)) > clip * np.median(sigs_q2)] = np.median(medians_q2)
#     xq2 = np.arange((nx/2),nx)
#     yq2 = np.arange(0,(ny/2))
#     XX_q2,YY_q2 = np.meshgrid(xq2,yq2)
#     xq2_norm = (XX_q2.flatten() / ((len(xq2)-1)/2.)) - 3.   #not quite right
#     yq2_norm = (YY_q2.flatten() / ((len(yq2)-1)/2.)) - 1.
    coeffs_q2 = polyfit2d(x_norm, y_norm, medimg_q2.flatten(), order=degpol)
    
    #Quadrant 3
    medimg_q3 = medimg[(ny/2):, (nx/2):]
    #clean this, otherwise the surface fit will be rubbish
    medimg_q3[np.abs(medimg_q3 - np.median(medians_q3)) > clip * np.median(sigs_q3)] = np.median(medians_q3)
    coeffs_q3 = polyfit2d(x_norm, y_norm, medimg_q3.flatten(), order=degpol)
    
    #Quadrant 4
    medimg_q4 = medimg[(ny/2):, :(nx/2)]
    #clean this, otherwise the surface fit will be rubbish
    medimg_q4[np.abs(medimg_q4 - np.median(medians_q4)) > clip * np.median(sigs_q4)] = np.median(medians_q4)
    coeffs_q4 = polyfit2d(x_norm, y_norm, medimg_q4.flatten(), order=degpol)
    
    #return all coefficients as 4-element array
    coeffs = np.array([coeffs_q1, coeffs_q2, coeffs_q3, coeffs_q4])            

    #convert read-out noise (but NOT offsets!!!) to units of electrons rather than ADUs by multiplying with the gain (which has units of e-/ADU)
    if gain is None:
        print('ERROR: gain(s) not set!!!')
        return
    else:
        rons = rons * gain

    if debug_level >= 1:
        #plot stuff
        print('plot the distributions for the four quadrants maybe!?!?!?')

    print('Done!!!')
    if timit:
        print('Time elapsed: '+str(np.round(time.time() - start_time,1))+' seconds')

    if save_medimg:
        #save median bias image
        dum = bias_list[0].split('/')
        path = bias_list[0][0:-len(dum[-1])]
        #write median bias image to file
        pyfits.writeto(path+'median_bias.fits', medimg, clobber=True)
        pyfits.setval(path+'median_bias.fits', 'UNITS', value='ADU')
        pyfits.setval(path+'median_bias.fits', 'HISTORY', value='   master BIAS frame - created '+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+' (GMT)')

    return medimg,coeffs,offsets,rons





def make_offmask_and_ronmask(offsets, rons, nx, ny, gain=None, savefiles=False, path=None, timit=False):
    """
    This routine creates a 1-level or 4-level master bias frame of size (ny,nx)
    
    INPUT:
    'offsets'    : offset/bias levels as measured by "get_offset_and_readnoise_from_bias_frames" (either 1-element or 4-element) [ADU]
    'rons'       : read-out noise level(s) from "get_offset_and_readnoise_from_bias_frames" [e-]
    'nx'         : image dimensions
    'ny'         : image dimensions
    'gain'       : array of gains for each quadrant (in units of e-/ADU) (only needed for writing the header really...)
    'savefiles'  : boolean - do you want to save the master bias to a fits file?
    'path'       : path to the output file directory (only needed if savefile is set to TRUE)
    'timit'      : boolean - do you want to measure execution run time?
    
    OUTPUT:
    'offmask'  : master bias image (or offset-image really...) [ADU]
    'ronmask'  : read-out noise mask (or RON-image really...) [e-]
    """

    if timit:
        start_time = time.time()

    nq = len(offsets)

    if nq == 1:
        offmask = np.ones((ny,nx)) * offsets
        ronmask = np.ones((ny,nx)) * rons
    elif nq == 4:
        offmask = np.ones((ny,nx))
        ronmask = np.ones((ny,nx))
        q1,q2,q3,q4 = make_quadrant_masks(nx,ny)
        for q,offset in zip([q1,q2,q3,q4],offsets):
            offmask[q] = offmask[q] * offset
        for q,RON in zip([q1,q2,q3,q4],rons):
            ronmask[q] = ronmask[q] * RON 
    else:
        print('ERROR: "offsets" must either be a scalar (for single-port readout) or a 4-element array/list (for four-port readout)!')
        return
            

    if savefiles:
        #check if gain is set
        if gain is None:
            print('ERROR: gain(s) not set!!!')
            return
                
        if path is None:
            print('ERROR: output file directory not provided!!!')
            return
        else:
            #write offmask to file
            pyfits.writeto(path+'offmask.fits', offmask, clobber=True)
            pyfits.setval(path+'offmask.fits', 'UNITS', value='ADU')
            pyfits.setval(path+'offmask.fits', 'HISTORY', value='   offset mask - created '+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+' (GMT)')
            if nq == 1:
                pyfits.setval(path+'offmask.fits', 'OFFSET', value=offsets, comment='in ADU')
                pyfits.setval(path+'offmask.fits', 'RNOISE', value=rons, comment='in ELECTRONS')
            elif nq == 4:
                pyfits.setval(path+'offmask.fits', 'OFFSET_1', value=offsets[0], comment='in ADU')
                pyfits.setval(path+'offmask.fits', 'OFFSET_2', value=offsets[1], comment='in ADU')
                pyfits.setval(path+'offmask.fits', 'OFFSET_3', value=offsets[2], comment='in ADU')
                pyfits.setval(path+'offmask.fits', 'OFFSET_4', value=offsets[3], comment='in ADU')
                pyfits.setval(path+'offmask.fits', 'RNOISE_1', value=rons[0], comment='in ELECTRONS')
                pyfits.setval(path+'offmask.fits', 'RNOISE_2', value=rons[1], comment='in ELECTRONS')
                pyfits.setval(path+'offmask.fits', 'RNOISE_3', value=rons[2], comment='in ELECTRONS')
                pyfits.setval(path+'offmask.fits', 'RNOISE_4', value=rons[3], comment='in ELECTRONS')

            #now make read-out noise mask
            pyfits.writeto(path+'read_noise_mask.fits', ronmask, clobber=True)
            pyfits.setval(path+'read_noise_mask.fits', 'UNITS', value='ELECTRONS')
            pyfits.setval(path+'read_noise_mask.fits', 'HISTORY', value='   read-noise frame - created '+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+' (GMT)')
            if nq == 1:
                pyfits.setval(path+'read_noise_mask.fits', 'OFFSET', value=offsets, comment='in ADU')
                pyfits.setval(path+'read_noise_mask.fits', 'GAIN', value=gain, comment='in e-/ADU')
                pyfits.setval(path+'read_noise_mask.fits', 'RNOISE', value=rons, comment='in ELECTRONS')
            elif nq == 4:
                pyfits.setval(path+'read_noise_mask.fits', 'OFFSET_1', value=offsets[0], comment='in ADU')
                pyfits.setval(path+'read_noise_mask.fits', 'OFFSET_2', value=offsets[1], comment='in ADU')
                pyfits.setval(path+'read_noise_mask.fits', 'OFFSET_3', value=offsets[2], comment='in ADU')
                pyfits.setval(path+'read_noise_mask.fits', 'OFFSET_4', value=offsets[3], comment='in ADU')
                pyfits.setval(path+'read_noise_mask.fits', 'RNOISE_1', value=rons[0], comment='in ELECTRONS')
                pyfits.setval(path+'read_noise_mask.fits', 'RNOISE_2', value=rons[1], comment='in ELECTRONS')
                pyfits.setval(path+'read_noise_mask.fits', 'RNOISE_3', value=rons[2], comment='in ELECTRONS')
                pyfits.setval(path+'read_noise_mask.fits', 'RNOISE_4', value=rons[3], comment='in ELECTRONS')
                pyfits.setval(path+'read_noise_mask.fits', 'GAIN_1', value=gain[0], comment='in e-/ADU')
                pyfits.setval(path+'read_noise_mask.fits', 'GAIN_2', value=gain[1], comment='in e-/ADU')
                pyfits.setval(path+'read_noise_mask.fits', 'GAIN_3', value=gain[2], comment='in e-/ADU')
                pyfits.setval(path+'read_noise_mask.fits', 'GAIN_4', value=gain[3], comment='in e-/ADU')

    if timit:
        print('Time elapsed: '+str(np.round(time.time() - start_time,1))+' seconds')

    return offmask,ronmask





def make_master_bias_from_coeffs(coeffs, nx, ny, savefile=False, path=None, timit=False):
    """
    Construct the master bais frame from the coefficients for the 2-dim polynomial surface fits to the 4 quadrants of the median bias frame.
    
    INPUT:
    'coeffs'    : coefficients for the 2-dim polynomial surface fits to the 4 quadrants of the median bias frame from "get_bias_and_readnoise_from_bias_frames"
    'nx'        : x-dimension of the full image (dispersion direction)
    'ny'        : y-dimension of the full image (cross-dispersion direction)
    'savefile'  : boolean - do you want to save the master bias frame to a fits file?
    'path'      : path to the output file directory (only needed if savefile is set to TRUE)
    'timit'     : boolean - do you want to measure execution run time?
    
    OUTPUT:
    'master_bias'  : the master bias frame [ADU]
    """
    
    if timit:
        start_time = time.time()
    
    #separate coefficients
    coeffs_q1 = coeffs[0]
    coeffs_q2 = coeffs[1]
    coeffs_q3 = coeffs[2]
    coeffs_q4 = coeffs[3]
    
    #create normalized x- & y-coordinates; size of the quadrants is (nx/2) x (ny/2)
    #the normalized coordinates are the same for all quadrants, of course
    xx_q1 = np.arange(nx/2)    
    yy_q1 = np.arange(ny/2)      
    xxn_q1 = (xx_q1 / (((nx/2)-1)/2.)) - 1. 
    yyn_q1 = (yy_q1 / (((ny/2)-1)/2.)) - 1.
    XX_norm,YY_norm = np.meshgrid(xxn_q1,yyn_q1)
    
    #model the 4 quadrants
    model_q1 = polyval2d(XX_norm, YY_norm, coeffs_q1)
    model_q2 = polyval2d(XX_norm, YY_norm, coeffs_q2)
    model_q3 = polyval2d(XX_norm, YY_norm, coeffs_q3)
    model_q4 = polyval2d(XX_norm, YY_norm, coeffs_q4)
    
    #make master bias frame from 4 quadrant models
    master_bias = np.zeros((ny,nx))
    master_bias[:(ny/2), :(nx/2)] = model_q1
    master_bias[:(ny/2), (nx/2):] = model_q2
    master_bias[(ny/2):, (nx/2):] = model_q3
    master_bias[(ny/2):, :(nx/2)] = model_q4
    
    
    #now save to FITS file
    if savefile:
        if path is None:
            print('ERROR: output file directory not provided!!!')
            return
        else:
            #get header from the read-noise mask file
            h = pyfits.getheader(path+'read_noise_mask.fits')
            #change a few things
            for i in range(1,5):
                del h['offset_'+str(i)]
            h['UNITS'] = 'ADU'
            h['HISTORY'][0] = ('   master BIAS frame - created '+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+' (GMT)')
            #write master bias to file
            pyfits.writeto(path+'master_bias.fits', master_bias, h, clobber=True)
    
    
    if timit:
        print('Time elapsed: '+str(np.round(time.time() - start_time,1))+' seconds')
    
    return master_bias





def make_master_dark(dark_list, MB, gain=None, scalable=False, savefile=True, path=None, debug_level=0, timit=False):
    """
    This routine creates a "MASTER DARK" frame from a given list of dark frames. It also subtracts the MASTER BIAS from each dark frame before 
    combining them into the master dark frame.
    NOTE: the output is in units of ELECTRONS!!!
    
    INPUT:
    'dark_list'    : list of raw dark image files (incl. directories)
    'MB'           : the master bias frame [ADU]
    'gain'         : the gains for each quadrant [e-/ADU]
    'scalable'     : boolean - do you want to normalize the dark current to an exposure time of 1s? (ie do you want to make it "scalable"?)
    'savefile'     : boolean - do you want to save the master dark frame to a fits file?
    'path'         : path to the output file directory (only needed if savefile is set to TRUE)
    'debug_level'  : for debugging...
    'timit'        : boolean - do you want to measure execution run time?
    
    OUTPUT:
    'MD'  : the master dark frame [e-]
    """
    
    #CMB - I removed the 'return_errors' keyword functionality, don't think that was quite right
    #also removed 'ronmask' from INPUT

    if timit:
        start_time = time.time()

    #THIS IS NOW DONE VIA A FUNCTION CALL TO "make_median_image"
#     #prepare arrays
#     allimg = []
# #     allerr = []
# 
#     #loop over all files in "dark_list"
#     for name in dark_list:
#         #read in dark image
#         img = pyfits.getdata(name)
#         if not simu:
#             #bring to "correct" orientation
#             img = correct_orientation(img)
#             #remove the overscan region
#             img = crop_overscan_region(img)
# #         #calculate noise
# #         err_img = np.sqrt(img.astype(float) + ronmask*ronmask)   #this is still in ADU
#         #subtract master bias and store in list (ignoring tiny uncertainties (~<0.01 ADU) in the bias level)
#         img_bc = img - MB
#         allimg.append(img_bc)
# #         #adjust errors and store in list
# #         err_img_bc = np.sqrt(err_img*err_img + ronmask*ronmask)   #this is still in ADU
# #        #NO! the RON is not the error in the bias level; so don't adjust error, just save it
# #        allerr.append(err_img)
# 
#     #get median image
#     MD = np.median(np.array(allimg), axis=0)
# #     err_summed = np.sqrt(np.sum((np.array(allerr)**2),axis=0))
# #     err_MD = err_summed / len(dark_list)

    if debug_level >= 1:
        print('Creating master dark frame from '+str(len(dark_list))+' dark frames...')       

    # get a list of all the exposure times first
    exp_times = []
    for file in sorted(dark_list):
        exp_times.append(np.round(pyfits.getval(file,'TOTALEXP'),0))

    # list of unique exposure times
    unique_exp_times = np.array(list(sorted(set(exp_times))))
    if debug_level >= 1 and len(unique_exp_times) > 1:
        print('WARNING: not all dark frames have the same exposure times! Found '+str(len(unique_exp_times))+' unique exposure times!!!')


    # create median dark files
    if scalable:
        #get median image (including subtraction of master bias) and scale to texp=1s
        MD = make_median_image(dark_list, MB=MB, scalable=scalable, raw=False)
        if gain is None:
            print('ERROR: gain(s) not given!!!')
            return
        else:
            ny,nx = MD.shape
            q1,q2,q3,q4 = make_quadrant_masks(nx,ny)
            MD[q1] = gain[0] * MD[q1]
            MD[q2] = gain[1] * MD[q2]
            MD[q3] = gain[2] * MD[q3]
            MD[q4] = gain[3] * MD[q4]
    else:
        # make dark "sublists" for all unique exposure times
        all_dark_lists = []
        for i in range(len(unique_exp_times)):
            all_dark_lists.append( np.array(dark_list)[np.argwhere(exp_times == unique_exp_times[i]).flatten()] )
        #get median image (including subtraction of master bias) for each "sub-list"
        MD = []
        for sublist in all_dark_lists:
            sub_MD = make_median_image(sublist, MB=MB, scalable=scalable, raw=False)
            #now convert to units of electrons
            if gain is None:
                print('ERROR: gain(s) not given!!!')
                return
            else:
                ny,nx = sub_MD.shape
                q1,q2,q3,q4 = make_quadrant_masks(nx,ny)
                sub_MD[q1] = gain[0] * sub_MD[q1]
                sub_MD[q2] = gain[1] * sub_MD[q2]
                sub_MD[q3] = gain[2] * sub_MD[q3]
                sub_MD[q4] = gain[3] * sub_MD[q4]
            MD.append(sub_MD)


    if savefile:
        if path is None:
            print('WARNING: output file directory not provided!!!')
            print('Using same directory as input file...')
            dum = dark_list[0].split('/')
            path = dark_list[0][0:-len(dum[-1])]
        if scalable:
            outfn = path+'master_dark_scalable.fits'
            #get header from master BIAS frame
            h = pyfits.getheader(path+'master_bias.fits')
            h['HISTORY'][0] = '   MASTER DARK frame - created '+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+' (GMT)'
            h['UNITS'] = 'ELECTRONS'
            h['COMMENT'] = 're-normalized to texp=1s to make it scalable'
            h['TOTALEXP'] = (1., 'exposure time [s]')
            pyfits.writeto(outfn, MD, h, clobber=True)
        else:
            for i,submd in enumerate(MD):
                outfn = path+'master_dark_t'+str(int(unique_exp_times[i]))+'.fits'
                #get header from master BIAS frame
                h = pyfits.getheader(path+'master_bias.fits')
                h['HISTORY'][0] = '   MASTER DARK frame - created '+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+' (GMT)'
                h['TOTALEXP'] = (unique_exp_times[i], 'exposure time [s]')
                h['UNITS'] = 'ELECTRONS'
                pyfits.writeto(outfn, submd, h, clobber=True)
#         if return_errors:
#             h_err = h.copy()
#             h_err['HISTORY'] = 'estimated uncertainty in MASTER DARK frame - created '+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+' (GMT)'
#             pyfits.append(outfn, err_MD, h_err, clobber=True)

    if timit:
        print('Time elapsed: '+str(np.round(time.time() - start_time,1))+' seconds')

#     if return_errors:
#         return MD,err_MD
#     else:
    return MD





def correct_for_bias_and_dark(img, MB, MD, gain=None, scalable=False, texp=None, timit=False):
    """
    This routine subtracts both the MASTER BIAS frame [in ADU], and the MASTER DARK frame [in e-] from a given image.
    Note that the input image has units of ADU, but the output image has units of electrons!!!
    
    NOT UP TO DATE!!!
    """


    #CMB - I removed the 'ronmask' and 'err_MD' INPUTs

    if timit:
        start_time = time.time()

#     #(0) get error estimate for image
#     err_img = np.sqrt(img.astype(float) + ronmask*ronmask)

    #(1) BIAS SUBTRACTION
    #bias-corrected_image [ADU]
    bc_img = img - MB
#     #adjust errors accordingly
#     err_bc_img = np.sqrt(err_img*err_img + ronmask*ronmask)

    #(2) DARK SUBTRACTION
    #if the darks have a different exposure time then the images you are trying to correct, we need to re-scale the master dark
    if scalable:
        if texp is not None:
            MD = MD * texp
#             #cannot have an error estimate lower than the read-out noise; this is dodgy but don't know what else to do
#             err_MD = np.maximum(err_MD * texp, ronmask)
        else:
            print('ERROR: "texp" has to be provided when "scalable" is set to TRUE')
            return
    #dark- & bias-corrected image
    dc_bc_img = bc_img - MD
#     #adjust errors accordingly
#     err_dc_bc_img = np.sqrt(err_bc_img*err_bc_img + err_MD*err_MD)

    if timit:
        print('Time elapsed: '+str(np.round(time.time() - start_time,1))+' seconds')

#     return dc_bc_img, err_dc_bc_img
    return dc_bc_img





def correct_for_bias_and_dark_from_filename(imgname, MB, MD, gain=None, scalable=False, savefile=False, path=None, simu=False, timit=False):
    """
    This routine subtracts both the MASTER BIAS frame [in ADU], and the MASTER DARK frame [in e-] from a given image.
    It also corrects the orientation of the image and crops the overscan regions.
    NOTE: the input image has units of ADU, but the output image has units of electrons!!!
    
    INPUT:
    'imgname'   : filename of raw science image (incl. directory)
    'MB'        : the master bias frame [ADU]
    'MD'        : the master dark frame [e-]
    'gain'      : the gains for each quadrant [e-/ADU]
    'scalable'  : boolean - do you want to normalize the dark current to an exposure time of 1s? (ie do you want to make it "scalable"?)
    'savefile'  : boolean - do you want to save the bias- & dark-corrected image (and corresponding error array) to a FITS file?
    'path'      : output file directory
    'simu'      : boolean - are you using Echelle++ simulated observations?
    'timit'     : boolean - do you want to measure the execution run time?
    
    OUTPUT:
    'dc_bc_img'  : the bias- & dark-corrected image [e-] (also has been brought to 'correct' orientation and overscan regions cropped) 
    
    MODHIST:
    #CMB - I removed the 'ronmask' and 'err_MD' INPUTs
    clone of "correct_for_bias_and_dark", but this one allows us to save output files
    """
    if timit:
        start_time = time.time()

    #(0) read in raw image [ADU] 
    img = pyfits.getdata(imgname)
    if not simu:
        #bring to "correct" orientation
        img = correct_orientation(img)
        #remove the overscan region
        img = crop_overscan_region(img)

    #(1) BIAS SUBTRACTION [ADU]
    #bias-corrected_image
    bc_img = img - MB


    #(2) conversion to ELECTRONS and DARK SUBTRACTION [e-]
    #if the darks have a different exposure time than the images you are trying to correct, we need to re-scale the master dark
    if scalable:
        texp = pyfits.getval(imgname, 'exptime')
        #texp = pyfits.getval(imgname, 'TOTALEXP')
        if texp is not None:
            MD = MD * texp
#             #cannot have an error estimate lower than the read-out noise; this is dodgy but don't know what else to do
#             err_MD = np.maximum(err_MD * texp, ronmask)
        else:
            print('ERROR: "texp" has to be provided when "scalable" is set to TRUE')
            return
    #convert image to electrons now    
    ny,nx = bc_img.shape
    q1,q2,q3,q4 = make_quadrant_masks(nx,ny)
    bc_img[q1] = gain[0] * bc_img[q1]
    bc_img[q2] = gain[1] * bc_img[q2]
    bc_img[q3] = gain[2] * bc_img[q3]
    bc_img[q4] = gain[3] * bc_img[q4]
    #now subtract master dark frame [e-] to create dark- & bias-corrected image [e-]
    dc_bc_img = bc_img - MD


    #if desired, write bias- & dark-corrected image and error array to fits files
    if savefile:
        dum = imgname.split('/')
        dum2 = dum[-1].split('.')
        shortname = dum2[0]
        if path is None:
            print('WARNING: output file directory not provided!!!')
            print('Using same directory as input file...')
            path = imgname[0:-len(dum[-1])]
        #outfn = path+shortname+'_bias_and_dark_corrected.fits'
        outfn = path+shortname+'_BD.fits'
        #get header from the original image FITS file
        h = pyfits.getheader(imgname)
        h['UNITS'] = 'ELECTRONS'
        h['HISTORY'] = '   BIAS- & DARK-corrected image - created '+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+' (GMT)'
        pyfits.writeto(outfn, dc_bc_img, h, clobber=True)
#         h_err = h.copy()
#         h_err['HISTORY'] = 'estimated uncertainty in BIAS- & DARK-corrected image - created '+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+' (GMT)'
#         pyfits.append(outfn, err_dc_bc_img, h_err, clobber=True)

    if timit:
        print('Time elapsed: '+str(np.round(time.time() - start_time,1))+' seconds')

#     return dc_bc_img, err_dc_bc_img
    return dc_bc_img



