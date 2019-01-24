'''
Created on 4 Oct. 2017

@author: Christoph Bergmann
'''

from scipy import ndimage
import numpy as np
import time
from click._compat import raw_input
import scipy.interpolate as ipol
import astropy.io.fits as pyfits

# imgname = '/Users/christoph/UNSW/cosmics/image.fit'
# img = pyfits.getdata(imgname)
# 
# xcenname = '/Users/christoph/UNSW/cosmics/xcen.fit'
# xcen = pyfits.getdata(xcenname)


def remove_cosmics(img, ronmask, obsname, path, Flim=3.0, siglim=5.0, maxiter=20, savemask=True, savefile=False, save_err=False, verbose=False, timit=False):
    """
    Top-level wrapper function for the cosmic-ray cleaning of an image. 
    
    INPUT:
    'img'      : input image (2-dim numpy array)
    'ronmask'  : read-out noise mask (or ron-image really...) from "make_master_bias_and_ronmask"
    'obsname'  : the obsname in "obsname.fits"
    'path'     : the directory of the files
    'Flim'     : lower threshold for the identification of a pixel as a cosmic ray when using L+/F (ie Laplacian image divided by fine-structure image) (= lbarplus/F2 in the implementation below)
    'siglim'   : sigma threshold for identification as cosmic in S_prime
    'maxiter'  : maximum number of iterations
    'savemask' : boolean - do you want to save the cosmic-ray mask?
    'savefile' : boolean - do you want to save the cosmic-ray corrected image?
    'save_err' : boolean - do you want to save the corresponding error array as well? (remains unchanged though)
    'verbose'  : boolean - for user information / debugging...
    'timit'    : boolean - do you want to measure execution run time?
    
    OUTPUT:
    'cleaned'  : the cosmic-ray corrected image
    """
    
    if timit:
        start_time = time.time()
        
    if verbose:
        print('Cleaning cosmic rays...')
    
    #some preparations    
    global_mask = np.cast['bool'](np.zeros(img.shape))
    n_cosmics = 0
    niter = 0
    n_new = 0
    cleaned = img.copy()
    
    #remove cosmics iteratively
    while ((niter == 0) or n_new > 0) and (niter < maxiter):
        print('Now running iteration '+str(niter+1)+'...')
        #go and identify cosmics
        mask = identify_cosmics(cleaned, ronmask, Flim=Flim, siglim=siglim, verbose=verbose, timit=timit)
        n_new = np.sum(mask)
        #add to global mask
        global_mask = np.logical_or(global_mask, mask)
        n_cosmics += n_new
        #n_global = np.sum(global_mask)     #should be equal to n_cosmics!!!!! if they're not, this means that some of the "cleaned" cosmics from a previous round are identified as cosmics again!!! well, they're not...
        #now go and clean these newly found cosmics
        cleaned = clean_cosmics(cleaned, mask, verbose=verbose, timit=timit)
        niter += 1
    
    #save cosmic-ray mask
    if savemask:
        outfn = path+obsname+'_CR_mask.fits'
        #get header from the BIAS- & DARK-subtracted image if it exits; otherwise from the original image FITS file
        try:
            h = pyfits.getheader(path+obsname+'_BD.fits')
        except:
            h = pyfits.getheader(path+obsname+'.fits')
        h['HISTORY'] = '   (boolean) COSMIC-RAY MASK- created '+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+' (GMT)'
        pyfits.writeto(outfn, global_mask.astype(int), h, clobber=True)
    
    #save cosmic-ray corrected image    
    if savefile:
        outfn = path+obsname+'_BD_CR.fits'
        #get header from the BIAS- & DARK-subtracted images if they exit; otherwise from the original image FITS file
        try:
            h = pyfits.getheader(path+obsname+'_BD.fits')
        except:
            h = pyfits.getheader(path+obsname+'.fits')
            h['UNITS'] = 'ELECTRONS'
        h['HISTORY'] = '   COSMIC-RAY corrected image - created '+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+' (GMT)'
        pyfits.writeto(outfn, cleaned, h, clobber=True)
        #also save the error array if desired
        if save_err:
            try:
                err = pyfits.getdata(path+obsname+'_BD.fits', 1)
                h_err = h.copy()
                h_err['HISTORY'] = 'estimated uncertainty in COSMIC-RAY corrected image - created '+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+' (GMT)'
                pyfits.append(outfn, err, h_err, clobber=True)
            except:
                print('WARNING: error array not found - cannot save error array')
            
    
    if verbose:
        print('Done!')
    
    if timit:
        print('Total time elapsed: '+str(np.round(time.time() - start_time,1))+' seconds')
        
    return cleaned





def identify_cosmics(img, ronmask, Flim=3.0, siglim=5.0, verbose=False, timit=False):
    """
    This routine identifies cosmic rays via Laplacian edge detection based on the method (ie LACosmic) described by van Dokkum et al., 2001, PASP, 113:1420 
    Should be slightly faster than the python translation of LACosmics by Malte Tewes, as it uses "ndimage" rather than "signal" package for convolution.
    
    INPUT:
    "img"       - a 2-dim image
    "ronmask"   - read-out noise in ADUs
    "Flim"      - lower threshold for the identification of a pixel as a cosmic ray when using L+/F (ie Laplacian image divided by fine-structure image) (= lbarplus/F2 in the implementation below)
    "siglim"    - sigma threshold for identification as cosmic in S_prime
    
    OUTPUT:
    "final_mask"   - a boolean mask, where True identifies pixels affected by cosmic rays. This mask has the same dimensions as the input image "img"
    
    TODO:
    This basically only identifies the cosmics. If I ever have time, I will try to implement a method similar to Bai et al., 2017, PASP, 129:024004.
    """
    
    #timing
    if timit:
        start_time = time.time()
    
    #user info
    if verbose:
        print('Identifying cosmics...')
    
    #subsample by a factor of 2
    im_sub = subsample(img)
    #Laplacian Kernel
    lapkernel = np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])
    #"grow-kernel"
    growkernel = np.ones((3,3))
    #Laplacian image (eq.3 in Bai et al, or eq. 6 in van Dokkum et al)
    #lbar = signal.convolve2d(im_sub,lapkernel,mode='same',boundary='symm')   #I don't know if this is the right boundary option...
    #there are ONLY machine precision differences??? ndimage is a bit faster...
    lbar = ndimage.convolve(im_sub,lapkernel)
    #eq. 7 in van Dokkum
    ###this works, but is slow:
    ###lbarplus_sub = lbar.copy()
    ###lbarplus_sub[lbarplus_sub < 0] = 0
    #this is faster:
    lbarplus_sub = lbar.clip(min=0.0)
    #resample to original sampling
    lbarplus = rebin2x2(lbarplus_sub)
    
    #median-filter the original image with a 5x5 box:
    ###this works, but it is slow
    ###im5 = signal.medfilt(img, kernel_size=5)
    #this is faster (they are equal if mode='constant', but that's just edge effects):
    im5 = ndimage.filters.median_filter(img, size=5, mode='mirror')   #again, IDK if that is the right mode
    if np.min(im5) < 0.00001:
        im5 = im5.clip(min=0.00001) # As we will take the sqrt
        
    #construct noise model (ie eq. 10 in van Dokkum, or eq. 6 in Bai et al)
    #CMB comment: this is a noise model in units of ADU; use this in the "normal" case, where you have a given value for gain (in e-/ADU) and RON (in e-)
    #noise = (1.0/gain) * np.sqrt(gain*im5 + RON*RON)    
    #BUT in the Veloce pipeline we have the 2-dim images already in units of electrons at this stage, and our read-noise is a 2-dim array in electrons, so simply do this:
    # ie it does not matter if this is electrons or ADU, as long as it's consistent either way
    noise = np.sqrt(im5 + ronmask*ronmask)    
    
    
    #calculate S (eq. 11 in van Dokkum or eq. 7 in Bai et al)
    S = 0.5 * lbarplus / noise
    
    #median-filter S to smooth out larger structures
    S_m5 = ndimage.filters.median_filter(S, size=5, mode='mirror')
    S_prime = S - S_m5
    
    #fine-structure image F
    im3 = ndimage.filters.median_filter(img, size=3, mode='mirror')
    im3_m7 = ndimage.filters.median_filter(im3, size=7, mode='mirror')
    F = im3 - im3_m7
    # In the article that's it, but in lacosmic.cl f is divided by the noise...
    # Ok I understand why, it depends on if you use sp/f or L+/f as criterion.
    # There are some differences between the article and the iraf implementation.
    # So I will stick to the iraf implementation.
    F2 = F / noise
    F = F.clip(min=0.01)
    F2 = F2.clip(min=0.01) # as we will divide by f. like in the iraf version.
    
    sig_mask = S_prime > siglim   # has strange edge effects, maybe because of mode='mirror' ???
    #F_mask = (lbarplus/F > Flim)     THIS IS WHAT THE PAPER SAYS
    F_mask = (S_prime/F2 > Flim)     #this is what the IRAF implementation does
    cosmic_mask = np.logical_and(sig_mask, F_mask)
    
    # We "grow" these cosmics a first time to determine the immediate neighbourhood  :
    growcosmics = np.cast['bool'](ndimage.convolve(np.cast['float32'](cosmic_mask),growkernel))   
    # From this grown set, we keep those that have sp > siglim
    # so obviously not requiring S_prine/F2 > Flim, otherwise it would be pointless...
    growcosmics = np.logical_and(S_prime > siglim, growcosmics)
    
    # Now we repeat this procedure, but lower the detection limit to sigmalimlow :        
    final_mask = np.cast['bool'](ndimage.convolve(np.cast['float32'](growcosmics),growkernel))
    final_mask = np.logical_and(S_prime > 0.3*siglim, final_mask)
    
    #user info
    ncosmic = np.sum(final_mask)
    if verbose:
        print('Number of pixels found to be affected by cosmic rays: '+str(ncosmic))
    
    #timing
    if timit:
        delta_t = time.time() - start_time
        print('Time taken for cosmic ray identification: '+str(delta_t)+' seconds...')
    
    #return the final cosmic mask       ; todo: and total number of cosmics found, and, b/c this is done iteratively the number and locations of cosmics found in the particular iteration
    return final_mask





def clean_cosmics(img, mask, badpixmask=None, method='median', boxsize=5, verbose=False, timit=False):
        """
        This routine replaces the flux in the pixels identified as being affected by cosmics rays (from function "identify_cosmics") with either
        a median value of surrounding non-cosmic-affected pixels, or with the value of a surface fit to the surrounding non-affected pixels,
        depending on the "method" kwarg.
        
        INPUT:
        "img"          - a 2-dim image
        "mask"         - a 2-dim boolean mask, where True identifies pixels affected by cosmic rays (this MUST have the same dimensions as "img"!!!)
        "badpixmask"   - a 2-dim mask of otherwise bad pixels (other than cosmics), which are not going to be replaced but are not used in the calculation
                         of the median values or spline interpolation of the replacement values
        
        KWARGS:
        "method"    - 'median' : the flux values in the cosmic-affected pixels are replaced by the median value of the surrounding non-cosmic-affected pixels
                    - 'spline' : a cubic spline interpolation is performed through the surrounding non-cosmic-affected pixels and the flux of the cosmic-affected
                                 pixels is replaced with the interpolated value at their respective locations
        "boxsize"   - the size of the surrounding pixels to be considered. default value is 5, ie a box of 5x5 pixels centred on the affected pixel
        "verbose"   - for debugging...
        "timit"     - boolean - do you want to measure execution run time?

        This routine borrows heavily from the python translation of LACosmic by Malte Tewes!
        
        TODO:
        implement surface fit method
        """
        
        if timit:
            start_time = time.time()
        
        #check that img and mask really do have the same size
        if img.shape != mask.shape:
            print("ERROR: Image and cosmic pixel mask do not have the same dimensions!!!")
            quit()
            return
        
        #if badpixmask is supplied, check that it has the same dimensions as well
        if (badpixmask is not None):
            if badpixmask.shape != mask.shape:
                print('WARNING: Bad pixel mask has different dimension than image and cosmic pixel mask!!!')
                choice = None
                while choice == None:   
                    choice = raw_input('Do you want to continue without using the bad pixel mask? ["y"/"n"]     : ')
                    if choice in ['n','N','no','No']:
                        quit()
                        return
                    elif choice in ['y','Y','yes','Yes']:
                        print('OK, ignoring bad pixel mask...')
                    else:
                        print('Invalid input! Please try again...')
                        choice = None
        
        #check that boxsize is an odd number
        while (boxsize % 2) == 0:
            print('ERROR: size of the box for median/interpolation needs to be an odd number, please try again!')
            boxsize = input('Enter an odd number for the box size: ')
            
        #create a copy of the image which is to be manipulated
        cleaned = img.copy()
            
        if verbose:
            print("Cleaning cosmic-affected pixels ...")
        
        # So...mask is a 2D-array containing False and True, where True means "here is a cosmic"
        # We want to loop through these cosmics one by one. This is a list of the indices of cosmic affected pixels:
        cosmicindices = np.argwhere(mask)   
        
        # We put cosmic ray pixels to np.Inf to flag them :
        cleaned[mask] = np.Inf
        
        # Now we want to have a 2 pixel frame of Inf padding around our image.
        w = cleaned.shape[0]
        h = cleaned.shape[1]
        #padsize = floor(boxsize/2.)   #same thing really...
        padsize = int(boxsize)/2
        #create this "padarray" so that edge effects are taken care of without the need for awkward for-/if-loops around adge pixels
        padarray = np.zeros((w+boxsize-1, h+boxsize-1)) + np.Inf
        padarray[padsize:w+padsize,padsize:h+padsize] = cleaned.copy() # that copy is important, we need 2 independent arrays
        
        # The medians will be evaluated in this padarray, excluding the infinite values (that are either the edges, the cosmic-affected pixels or the otherwise bad pixels)
        # Now in this copy called padarray, we also put the saturated stars to np.Inf, if available :
        if badpixmask is not None:
            padarray[padsize:w+padsize,padsize:h+padsize][badpixmask] = np.Inf
        
        # A loop through every cosmic pixel :
        for cosmicpos in cosmicindices:
            x = cosmicpos[0]
            y = cosmicpos[1]
#             if verbose:
#                 print('[x,y] = ['+str(x)+','+str(y)+']')
            cutout = padarray[x:x+boxsize, y:y+boxsize].ravel() # remember the shift due to the padding !
            # Now we have our cutout pixels, some of them are np.Inf, which will be ignored for calculating median or interpolating
            goodcutout = cutout[cutout != np.Inf]
            
            if np.alen(goodcutout) >= boxsize*boxsize :
                # This never happened, but you never know ...
                #raise RuntimeError, "Mega error in clean !"
                raise RuntimeError("Mega error in clean !")
            elif np.alen(goodcutout) > 0 :
                #WHICH METHOD???
                if method == 'median':
                    replacementvalue = np.median(goodcutout)
                elif method == 'spline':
                    print('WARNING: THIS IS NOT FULLY IMPLEMENTED YET!!!!!')
                    box = padarray[x:x+boxsize, y:y+boxsize]
                    goodbox = np.argwhere(box != np.Inf)
                    xx = goodbox[:,1]
                    yy = goodbox[:,0]
                    zz = goodcutout
                    spline_func = ipol.interp2d(xx,yy,zz,kind='cubic')
                    replacementvalue = spline_func(padsize,padsize)
                else:
                    #raise RuntimeError, 'invalid kwarg for "method" !'
                    raise RuntimeError('invalid kwarg for "method" !')
            else :    
                # i.e. no good pixels : Shit, a huge cosmic, we will have to improvise ...
                print("WARNING: Huge cosmic ray encounterd - it fills the entire ("+str(boxsize)+"x"+str(boxsize)+")-pixel cutout! Using backup value...")
                replacementvalue = np.median(padarray[padarray != np.Inf])    #I don't like this...maybe need to do sth smarter in the future, but I doubt it will ever happen if boxsize is sufficiently large
            
            # Now update the cleaned array, but remember the median was calculated from the padarray...otherwise it would depend on the order in which the cosmics are treated!!!
            cleaned[x, y] = replacementvalue
            
        # That's it.
        if verbose:
            #print "Cleaning done!"
            print("Cleaning done!")
            
        if timit:
            print('Time elapsed: '+str(np.round(time.time() - start_time,1))+' seconds')    
            
        #return the cleaned image
        return cleaned    
            
      
      
            
   
def subsample(a): # this is more a generic function then a method ...
    """
    Returns a 2x2-subsampled version of array a (no interpolation, just cutting pixels in 4).
    The version below is directly from the scipy cookbook on rebinning :
    U{http://www.scipy.org/Cookbook/Rebinning}
    There is ndimage.zoom(cutout.array, 2, order=0, prefilter=False), but it makes funny borders.
    
    """
    """
    # Ouuwww this is slow ...
    outarray = np.zeros((a.shape[0]*2, a.shape[1]*2), dtype=np.float64)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):    
            outarray[2*i,2*j] = a[i,j]
            outarray[2*i+1,2*j] = a[i,j]
            outarray[2*i,2*j+1] = a[i,j]
            outarray[2*i+1,2*j+1] = a[i,j]
    return outarray
    """
    # much better :
    newshape = (2*a.shape[0], 2*a.shape[1])
    slices = [slice(0,old, float(old)/new) for old,new in zip(a.shape,newshape) ]
    coordinates = np.mgrid[slices]
    indices = coordinates.astype('i')   #choose the biggest smaller integer index
    return a[tuple(indices)]
    
    
    
    

def rebin(a, newshape):
    """
    Auxiliary function to rebin an ndarray a.
    U{http://www.scipy.org/Cookbook/Rebinning}
    
            >>> a=rand(6,4); b=rebin(a,(3,2))
    """
        
    shape = a.shape
    lenShape = len(shape)
    factor = np.asarray(shape)/np.asarray(newshape)
    #print factor
    #evList = ['a.reshape('] + ['newshape[%d],factor[%d],'%(i,i) for i in xrange(lenShape)] + [')'] + ['.sum(%d)'%(i+1) for i in xrange(lenShape)] + ['/factor[%d]'%i for i in xrange(lenShape)]
    evList = ['a.reshape('] + ['newshape[%d],factor[%d],'%(i,i) for i in range(lenShape)] + [')'] + ['.sum(%d)'%(i+1) for i in range(lenShape)] + ['/factor[%d]'%i for i in range(lenShape)]

    return eval(''.join(evList))





def rebin2x2(a):
    """
    Wrapper around rebin that actually rebins 2 by 2
    """
    inshape = np.array(a.shape)
    if not (inshape % 2 == np.zeros(2)).all(): # Modulo check to see if size is even
        #raise RuntimeError, "I want even image shapes !"
        raise RuntimeError("I want even image shapes !")
        
    return rebin(a, inshape/2)         