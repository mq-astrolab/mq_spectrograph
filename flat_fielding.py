'''
Created on 26 Apr. 2018

@author: Christoph Bergmann
'''

import numpy as np
from scipy import ndimage



def onedim_pixtopix_variations(flat, filt='gaussian', filter_width=25):
    """
    This routine applies a filter ('gaussian' / 'savgol' / 'median') to an observed flat field in order to determine the pixel-to-pixel sensitivity variations
    as well as the fringing pattern in the red orders. This is done in 1D, ie for the already extracted spectrum.
    
    INPUT:
    'flat'          : dictionary / np.array containing the extracted flux from the flat field (master white) (keys = orders)
    'filt'          : method of filtering ('gaussian' / 'savgol' / 'median') - WARNING: ONLY GAUSSIAN FILTER HAS BEEN IMPLEMENTED SO FAR!!!
    'filter_width'  : the width of the kernel for the filtering in pixels; defined differently for the different types of filters (see description of scipy.ndimage....)
    
    OUTPUT:
    'pix_sens'      : dictionary of the pixel-to-pixel sensitivities (keys = orders)
    'smoothed_flat' : dictionary of the smoothed (ie filtered) whites (keys = orders)
    
    MODHIST:
    24/05/2018 - CMB create
    15/01/2019 - CMB added choice of dict / np-array as input
    """
    
    while filt.lower() not in ['g','gaussian','s','savgol','m','median']:
        print("ERROR: filter choice not recognised!")
        filt = raw_input("Please try again: ['(G)aussian','(S)avgol','(M)edian']")
    
    if flat.__class__ == np.ndarray:
        pix_sens = np.zeros(flat.shape) - 1.
        smoothed_flat = np.zeros(flat.shape) - 1.
        # loop over all orders
        for o in range(flat.shape[0]):
            # loop over all fibres
            for f in range(flat.shape[1]): 
                if filt.lower() in ['g','gaussian']:
                    #Gaussian filter
                    smoothed_flat[o,f,:] = ndimage.gaussian_filter(flat[o,f,:], filter_width)    
                    pix_sens[o,f,:] = flat[o,f,:] / smoothed_flat[o,f,:]
                elif filt.lower() in ['s','savgol']:
                    print('WARNING: SavGol filter not implemented yet!!!')
                    break
                elif filt.lower() in ['m','median']:
                    print('WARNING: Median filter not implemented yet!!!')
                    break
                else:
                    #This should never happen!!!
                    print("ERROR: filter choice still not recognised!")
                    break
    elif flat.__class__ == dict:
        pix_sens = {}
        smoothed_flat = {}
        # loop over all orders
        for ord in sorted(flat.keys()): 
            if filt.lower() in ['g','gaussian']:
                #Gaussian filter
                smoothed_flat[ord] = ndimage.gaussian_filter(flat[ord], filter_width)    
                pix_sens[ord] = flat[ord] / smoothed_flat[ord]
            elif filt.lower() in ['s','savgol']:
                print('WARNING: SavGol filter not implemented yet!!!')
                break
            elif filt.lower() in ['m','median']:
                print('WARNING: Median filter not implemented yet!!!')
                break
            else:
                #This should never happen!!!
                print("ERROR: filter choice still not recognised!")
                break
    else:
        print('ERROR: data type / variable class not recognized')
        return   
        
    return smoothed_flat, pix_sens
    

    
def onedim_pixtopix_variations_single_order(f_flat, filt='gaussian', filter_width=25):
    """
    This routine applies a filter ('gaussian' / 'savgol' / 'median') to an observed flat field in order to determine the pixel-to-pixel sensitivity variations
    as well as the fringing pattern in the red orders. This is done in 1D, ie for the already extracted spectrum.
    
    INPUT:
    'f_flat'        : 1-dim array containing the extracted flux from the flat field (master white) for one order
    'filt'          : method of filtering ('gaussian' / 'savgol' / 'median') - WARNING: ONLY GAUSSIAN FILTER HAS BEEN IMPLEMENTED SO FAR!!!
    'filter_width'  : the width of the kernel for the filtering in pixels; defined differently for the different types of filters (see description of scipy.ndimage....)
    
    OUTPUT:
    'pix_sens'      : dictionary of the pixel-to-pixel sensitivities (keys = orders)
    'smoothed_flat' : dictionary of the smoothed (ie filtered) whites (keys = orders)
    
    MODHIST:
    05/10/2018 - CMB create   (clone of "onedim_pixtopix_variations")
    """
    
    while filt.lower() not in ['g','gaussian','s','savgol','m','median']:
        print("ERROR: filter choice not recognised!")
        filt = raw_input("Please try again: ['(G)aussian','(S)avgol','(M)edian']")
    
    if filt.lower() in ['g','gaussian']:
        #Gaussian filter
        smoothed_flat = ndimage.gaussian_filter(f_flat, filter_width)    
        pix_sens = f_flat / smoothed_flat
    elif filt.lower() in ['s','savgol']:
        print('WARNING: SavGol filter not implemented yet!!!')
        return
    elif filt.lower() in ['m','median']:
        print('WARNING: Median filter not implemented yet!!!')
        return
    else:
        #This should never happen!!!
        print("ERROR: filter choice still not recognised!")
        return
        
    return smoothed_flat, pix_sens    
    

    
def deblaze_orders(f, wl, smoothed_flat, mask, err=None, degpol=1, gauss_filter_sigma=3., maxfilter_size=100):
    
    assert f.shape == smoothed_flat.shape, 'Shapes of "flux" and "smoothed_flat" do not agree!!!'
    assert wl.__class__ == smoothed_flat.__class__, '"wl" and "smoothed_flat" are not the same class object!!!' 
    assert f.__class__ == wl.__class__, '"flux" and "wl" are not the same class object!!!' 
    if err is not None:
        assert f.__class__ == err.__class__, '"flux" and "err" are not the same class object!!!'
        if smoothed_flat.__class__ == np.ndarray:
            assert f.shape == err.shape, 'Shapes of "flux" and "error" do not agree!!!' 
    
    # if everything comes a numpy arrays
    if smoothed_flat.__class__ == np.ndarray:
        f_dblz = np.zeros(f.shape)
        if err is not None:
            err_dblz = np.zeros(err.shape)
    
        # if using cross-correlation to get RVs, we need to de-blaze the spectra first...
        # loop over all orders
        for o in range(f.shape[0]):
            ord = 'order_'+str(o+1).zfill(2)
            # loop over all fibres
            for fib in range(f.shape[1]): 
                #first, divide by the "blaze-function", ie the smoothed flat, which we got from filtering the MASTER WHITE
                f_dblz[o,fib,:] = f[o,fib,:] / (smoothed_flat[o,fib,:]/np.max(smoothed_flat[o,fib,:]))
                #get rough continuum shape by performing a series of filters
                cont_rough = ndimage.maximum_filter(ndimage.gaussian_filter(f_dblz[o,fib,:], gauss_filter_sigma), size=maxfilter_size)
                #now fit polynomial to that rough continuum
                p = np.poly1d(np.polyfit(wl[o,fib,:][mask[ord]], cont_rough[mask[ord]], degpol))
                #divide by that polynomial
                f_dblz[o,fib,:] = f_dblz[o,fib,:] / (p(wl[o,fib,:]) / np.median(p(wl[o,fib,:])[mask[ord]]))
                #need to treat the error arrays in the same way, as need to keep relative error the same
                if err is not None:
                    err_dblz[o,fib,:] = err[o,fib,:] / (smoothed_flat[o,fib,:]/np.max(smoothed_flat[o,fib,:]))
                    err_dblz[o,fib,:] = err_dblz[o,fib,:] / (p(wl[o,fib,:]) / np.median(p(wl[o,fib,:])[mask[ord]]))
                    
    # if everything comes as dictionaries
    elif smoothed_flat.__class__ == dict:
        f_dblz = {}
        if err is not None:
            err_dblz = {}
        
        #if using cross-correlation to get RVs, we need to de-blaze the spectra first
        for o in f.keys():
            #first, divide by the "blaze-function", ie the smoothed flat, which we got from filtering the MASTER WHITE
            f_dblz[o] = f[o] / (smoothed_flat[o]/np.max(smoothed_flat[o]))
            #get rough continuum shape by performing a series of filters
            cont_rough = ndimage.maximum_filter(ndimage.gaussian_filter(f_dblz[o],gauss_filter_sigma), size=maxfilter_size)
            #now fit polynomial to that rough continuum
            p = np.poly1d(np.polyfit(wl[o][mask[o]], cont_rough[mask[o]], degpol))
            #divide by that polynomial
            f_dblz[o] = f_dblz[o] / (p(wl[o]) / np.median(p(wl[o])[mask[o]]))
            #need to treat the error arrays in the same way, as need to keep relative error the same
            if err is not None:
                err_dblz[o] = err[o] / (smoothed_flat[o]/np.max(smoothed_flat[o]))
                err_dblz[o] = err_dblz[o] / (p(wl[o]) / np.median(p(wl[o])[mask[o]]))
    
    else:
        print('ERROR: data type / variable class not recognized')
        return

    if err is not None:
        return f_dblz,err_dblz
    else:
        return f_dblz




