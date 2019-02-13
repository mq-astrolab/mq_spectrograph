"""
Created on 11 Aug. 2017

@author: Christoph Bergmann
@editor: Jacob Parnell
"""

# import astropy.io.fits as pyfits
import numpy as np
import itertools
import warnings
import time
import math
import datetime
from astropy.modeling import models, fitting
import collections
from scipy import special, signal
from numpy.polynomial import polynomial
from scipy.integrate import quad, fixed_quad
from scipy import ndimage
import glob
import astropy.io.fits as pyfits
import copy
import matplotlib.pyplot as plt
from IPython.display import Image
import os
import scipy




def linfunc(p, x):
    """linear function"""
    c,m = p
    return m*x + c



def gauss2D(xytuple, amp, x0, y0, x_sig, y_sig, theta):
    x,y = xytuple
    a = ((np.cos(theta))**2 / (2*x_sig**2)) + ((np.sin(theta))**2 / (2*y_sig**2))
    b = -np.sin(2*theta)/(4*x_sig**2) + np.sin(2*theta)/(4*y_sig**2)
    c = ((np.sin(theta))**2 / (2*x_sig**2)) + ((np.cos(theta))**2 / (2*y_sig**2))

    return amp * np.exp( -(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2) ) 



def fibmodel_fwhm(xarr, mu, fwhm, beta=2, alpha=0, norm=0):
    # model for the fibre profiles
    """WARNING: I think the relationship between sigma and FWHM is a function of beta!!!
        Maybe better to use sigma instead of FWHM!!!
    """
    
    #define constant (FWHM vs sigma, because FWHM = sigma * 2*sqrt(2*log(2))
    cons = 2*np.sqrt(np.log(2))
    
    #calculate "Gauss-like" model (if beta = 2 that's just a standard Gaussian)
    phi = np.exp(-(np.absolute(xarr-mu) * cons / fwhm) ** beta)
    
              
    if alpha == 0:
        #return "Gauss-like" model (symmetric)
        if norm ==1:
            #normfac = 1. / (np.sqrt(2 * np.pi) * (fwhm/(cons*np.sqrt(2))))
            normfac = beta / (2. * np.sqrt(2) * (fwhm/(cons*np.sqrt(2))) * special.gamma(1./beta))   #that's correct but make sure the wings have reached zero in the size of the array!!!
            phinorm = phi * normfac
            #print('normfac =',normfac)
            return phinorm
        else:
            return phi
    else:
        #now this allows for skewness, ie f(x)=2*phi(x)*PHI(alpha*x), where
        #phi(x)=Gaussian, and PHI(x)=CDF (cumulative distrobution funtion) = INT_-inf^x phi(x')dx' = 0.5*[1+erf(x/sqrt(2))]
#         if norm == 1:
#             norm2 = beta / (2. * (fwhm/(cons*np.sqrt(2))) * gamma(1./beta))
#             return norm2 * phi * (1. + erf(alpha * xarr / np.sqrt(2.)))
#         else:
            return phi * (1. + special.erf(alpha * xarr / np.sqrt(2.)))



def fibmodel(xarr, mu, sigma, beta=2, alpha=0, norm=0):
    # model for the fibre profiles (gauss-like function)
    #define constant (FWHM vs sigma, because FWHM = sigma * 2*sqrt(2*log(2))
    #cons = 2*np.sqrt(np.log(2))
    
    #calculate "Gauss-like" model (if beta = 2 that's just a standard Gaussian)
    phi = np.exp(- (np.absolute(xarr-mu) / (np.sqrt(2.)*sigma)) ** beta)
    
              
    if alpha == 0:
        #return "Gauss-like" model (symmetric)
        if norm ==1:
            #normfac = 1. / (np.sqrt(2 * np.pi) * (fwhm/(cons*np.sqrt(2))))
            normfac = beta / (2. * np.sqrt(2) * sigma * special.gamma(1./beta))   #that's correct but make sure the wings have reached zero in the size of the array!!!
            phinorm = phi * normfac
            #print('normfac =',normfac)
            return phinorm
        else:
            return phi
    else:
        #now this allows for skewness, ie f(x)=2*phi(x)*PHI(alpha*x), where
        #phi(x)=Gaussian, and PHI(x)=CDF (cumulative distrobution funtion) = INT_-inf^x phi(x')dx' = 0.5*[1+erf(x/sqrt(2))]
#         if norm == 1:
#             norm2 = beta / (2. * (fwhm/(cons*np.sqrt(2))) * gamma(1./beta))
#             return norm2 * phi * (1. + erf(alpha * xarr / np.sqrt(2.)))
#         else:
            return phi * (1. + special.erf(alpha * xarr / np.sqrt(2.)))



def CMB_multi_gaussian(x, *p):
    f = np.zeros(len(x))
    for i in range(len(p)//3):
        f += CMB_pure_gaussian(x, *p[i*3:i*3+3])
    return f

def CMB_multi_gaussian_with_offset(x, *p):
    return CMB_multi_gaussian(x,*p[:-2]) + p[-1]

def CMB_pure_gaussian(x, mu, sig, amp):
    return (amp * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))))

def gaussian_with_offset(x, mu, sig, amp, off):
    return (amp * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))) + off

def gaussian_with_offset_and_slope(x, mu, sig, amp, off, slope):
    return (amp * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))) + off + slope*x
 
def gaussian_with_slope(x, mu, sig, amp, slope):
    return (amp * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))) + slope*x 
 
def fibmodel_with_amp(x, mu, sigma, amp, beta):
    return amp * fibmodel(x, mu, sigma, beta=beta, alpha=0, norm=0)

def norm_fibmodel_with_amp(x, mu, sigma, amp, beta):
    return amp * fibmodel(x, mu, sigma, beta=beta, alpha=0, norm=1)

def multi_fibmodel_with_amp(x, *p):
    f = np.zeros(len(x))
    for i in range(len(p)//4):
        f += fibmodel_with_amp(x, *p[i*4:i*4+4])
    return f

def multi_fibmodel_with_amp_and_offset(x, *p):
    f = np.zeros(len(x))
    for i in range(len(p)//4):
        f += fibmodel_with_amp(x, *p[i*4:i*4+4])
    return f + p[-1]

def fibmodel_with_offset(x, mu, sigma, beta, offset):
    return fibmodel(x, mu, sigma, beta=beta, alpha=0, norm=0) + offset

def fibmodel_with_amp_and_offset(x, mu, sigma, amp, beta, offset):
    return amp * fibmodel(x, mu, sigma, beta=beta, alpha=0, norm=0) + offset

def norm_fibmodel_with_amp_and_offset(x, mu, sigma, amp, beta, offset):
    return amp * fibmodel(x, mu, sigma, beta=beta, alpha=0, norm=1) + offset

def fibmodel_with_amp_and_offset_and_slope(x, mu, sigma, amp, beta, offset, slope):
    return amp * fibmodel(x, mu, sigma, beta=beta, alpha=0, norm=0) + offset + slope*x

def gausslike_with_amp_and_offset_and_slope(x, mu, sigma, amp, beta, offset, slope):
    return amp * fibmodel(x, mu, sigma, beta=beta, alpha=0, norm=0) + offset + slope*x




def make_norm_profiles(x, o, col, fibparms, fibs='stellar', slope=False, offset=False):  
    
    #same number of fibres for every order, of course
    if fibs == 'all':
        #nfib = len(fibparms[o])  
        nfib = 28
        userange = np.arange(nfib) 
    elif fibs == 'stellar':
        nfib=19
        userange = np.arange(5,24,1)
    elif fibs == 'laser':
        nfib=1
        userange = np.arange(0,1,2)
    elif fibs == 'thxe':
        nfib=1
        userange = np.arange(27,28,2)
    elif fibs == 'sky3':
        nfib=3
        userange = np.arange(1,4,1)
    elif fibs == 'sky2':
        nfib=2
        userange = np.arange(25,27,1)
    else:
        print('ERROR: fibre selection not recognised')
        return
    
    #do we want to include extra "fibres" to take care of slope and/or offset? Default is NO for both (as this should be already taken care of globally)
    if offset:
        nfib += 1
    if slope:
        nfib += 1
    
    phi = np.zeros((len(x),28))
    
    for k,fib in enumerate(sorted(fibparms[o].keys())):
        mu = fibparms[o][fib]['mu_fit'](col)
        sigma = fibparms[o][fib]['sigma_fit'](col)
        beta = fibparms[o][fib]['beta_fit'](col)
        phi[:,k] = fibmodel(x, mu, sigma, beta=beta, alpha=0, norm=0)
    
    #deprecate phi-array to only use wanted fibres
    phi = phi[:,userange]
    
    if offset:
        phi[:,-2] = 1.
    if slope:
        phi[:,-1] = x - x[0]
    
    #return normalized profiles
    phinorm =  phi/np.sum(phi,axis=0)

    return phinorm

def make_norm_profiles_2(x, col, fppo, fibs='stellar', slope=False, offset=False):
    """
    clone of "make_norm_profiles", but takes as "fppo" (= fibparms per order) as input, rather
    than "ord" and the entire "fibparms" dictionary
    """
    #same number of fibres for every order, of course
    if fibs == 'all':
        #nfib = len(fibparms[ord])
        nfib = 28
        userange = np.arange(nfib)
    elif fibs == 'stellar':
        nfib=19
        userange = np.arange(5,24,1)
    elif fibs == 'laser':
        nfib=1
        userange = np.arange(0,1,2)
    elif fibs == 'thxe':
        nfib=1
        userange = np.arange(27,28,2)
    elif fibs == 'sky3':
        nfib=3
        userange = np.arange(1,4,1)
    elif fibs == 'sky2':
        nfib=2
        userange = np.arange(25,27,1)
    else:
        print('ERROR: fibre selection not recognised!!!')
        return

    #do we want to include extra "fibres" to take care of slope and/or offset? Default is NO for both (as this should be already taken care of globally)
    if offset:
        nfib += 1
    if slope:
        nfib += 1

    phi = np.zeros((len(x),28))

    for k,fib in enumerate(sorted(fppo.keys())):
        mu = fppo[fib]['mu_fit'](col)
        sigma = fppo[fib]['sigma_fit'](col)
        beta = fppo[fib]['beta_fit'](col)
        phi[:,k] = fibmodel(x, mu, sigma, beta=beta, alpha=0, norm=0)

    #deprecate phi-array to only use wanted fibres
    phi = phi[:,userange]

    if offset:
        phi[:,-2] = 1.
    if slope:
        phi[:,-1] = x - x[0]

    #return normalized profiles
    phinorm =  phi/np.sum(phi,axis=0)

    return phinorm

def make_norm_profiles_3(x, col, fppo, fibs='stellar', slope=False, offset=False):
    """
    THAT's the latest version to be used with fibre profiles from real fibre flats!!!
    In this version we have 24 fibres (19 stellasr + 5 sky)!
    clone of "make_norm_profiles", but takes as "fppo" (= fibparms per order) as input, rather
    than "ord" and the entire "fibparms" dictionary
    """
    
    # same number of fibres for every order, of course
    if fibs == 'all':
        nfib = 24
        userange = np.arange(nfib)
    elif fibs == 'stellar':
        nfib = 19
        userange = np.arange(2, 21, 1)
    # elif fibs == 'laser':
    #     nfib = 1
    #     userange = np.arange(0, 1, 2)
    # elif fibs == 'thxe':
    #     nfib = 1
    #     userange = np.arange(27, 28, 2)
    elif fibs == 'sky3':
        nfib = 3
        userange = np.arange(21, 24, 1)
    elif fibs == 'sky2':
        nfib = 2
        userange = np.arange(2)
    elif fibs == 'allsky':
        nfib = 5
        userange = np.r_[np.arange(2),np.arange(21, 24, 1)]
    else:
        print('ERROR: fibre selection not recognised!!!')
        return

    # Do we want to include extra "fibres" to take care of slope and/or offset?
    # Default is NO for both (as this should probably be already taken care of globally)
    addfibs = 0
    if offset:
        # nfib += 1
        addfibs += 1
    if slope:
        # nfib += 1
        addfibs += 1

    phi = np.zeros((len(x), 24+addfibs))

    # NOTE: need to turn fibre numbers around here to be correct
    for k, fib in enumerate(sorted(fppo.keys())[::-1]):
        mu = fppo[fib]['mu_fit'](col)
        sigma = fppo[fib]['sigma_fit'](col)
        beta = fppo[fib]['beta_fit'](col)
        phi[:, k] = fibmodel(x, mu, sigma, beta=beta, alpha=0, norm=0)

    if offset and not slope:
        phi[:, -1] = 1.
        userange = np.append(userange, 24)
    if slope and not offset:
        phi[:, -1] = x - x[0]
        userange = np.append(userange, 24)
    if offset and slope:
        phi[:, -2] = 1.
        phi[:, -1] = x - x[0]
        userange = np.append(userange, np.array([24,25]))

    # deprecate phi-array to only use wanted fibres
    phi = phi[:, userange]

    # return normalized profiles
    phinorm = phi / np.sum(phi, axis=0)

    return phinorm

def make_norm_profiles_4(x, col, fppo, integrate=False, fibs='stellar', slope=False, offset=False):
    """
    THAT's the latest version to be used with fibre profiles from real fibre flats!!!
    In this version we have 24 fibres (19 stellasr + 5 sky)!
    clone of "make_norm_profiles", but takes as "fppo" (= fibparms per order) as input, rather
    than "ord" and the entire "fibparms" dictionary
    """

    # same number of fibres for every order, of course
    if fibs == 'all':
        nfib = 24
        userange = np.arange(nfib)
    elif fibs == 'stellar':
        nfib = 19
        userange = np.arange(2, 21, 1)
    # elif fibs == 'laser':
    #     nfib = 1
    #     userange = np.arange(0, 1, 2)
    # elif fibs == 'thxe':
    #     nfib = 1
    #     userange = np.arange(27, 28, 2)
    elif fibs == 'sky3':
        nfib = 3
        userange = np.arange(21, 24, 1)
    elif fibs == 'sky2':
        nfib = 2
        userange = np.arange(2)
    elif fibs == 'allsky':
        nfib = 5
        userange = np.r_[np.arange(2),np.arange(21, 24, 1)]
    else:
        print('ERROR: fibre selection not recognised!!!')
        return

    # Do we want to include extra "fibres" to take care of slope and/or offset?
    # Default is NO for both (as this should probably be already taken care of globally)
    addfibs = 0
    if offset:
        # nfib += 1
        addfibs += 1
    if slope:
        # nfib += 1
        addfibs += 1

    phi = np.zeros((len(x), nfib + addfibs))

    # NOTE: need to turn fibre numbers around here to be correct
    for k, fib in enumerate(sorted(fppo.keys())[::-1]):
        mu = fppo[fib]['mu_fit'](col)
        sigma = fppo[fib]['sigma_fit'](col)
        beta = fppo[fib]['beta_fit'](col)
        # now, I think we actually don't want to evaluate the functional form of the profiles as declared by "fibmodel" at the respective locations,
        # but rather we want to integrate the (highly non-linear) function from the left edge to the right edge of the pixels (co-ordinates are pixel centres!!!)
        if integrate:
            for i in np.arange(len(x)):
                phi[i,k] = fixed_quad(fibmodel, x[i]-0.5, x[i]+0.5, args=(mu, sigma, beta))[0]
                # phi[i, k] = quad(fibmodel, x[i] - 0.5, x[i] + 0.5, args=(mu, sigma, beta))[0]
        else:
            phi[:, k] = fibmodel(x, mu, sigma, beta=beta, alpha=0, norm=0)

    if offset and not slope:
        phi[:, -1] = 1.
        userange = np.append(userange, nfib)
    if slope and not offset:
        phi[:, -1] = x - x[0]
        userange = np.append(userange, nfib)
    if offset and slope:
        phi[:, -2] = 1.
        phi[:, -1] = x - x[0]
        userange = np.append(userange, np.array([nfib,nfib+1]))

    # deprecate phi-array to only use wanted fibres
    phi = phi[:, userange]

    # return normalized profiles
    phinorm = phi / np.sum(phi, axis=0)

    return phinorm

def make_norm_profiles_5(x, col, fppo, integrate=False, fibs='stellar', slope=False, offset=False):
    """
    THAT's the latest version to be used with fibre profiles from real fibre flats!!!
    In this version we have 24 fibres (19 stellasr + 5 sky)!
    clone of "make_norm_profiles", but takes as "fppo" (= fibparms per order) as input, rather
    than "ord" and the entire "fibparms" dictionary
    
    UPDATE:
    This version now takes the fibparms in explicit form, rather than as a function to apply to 'pix'.
    """
    
    nfib = 24
    
    # same number of fibres for every order, of course
    if fibs == 'all':
        userange = np.arange(nfib)
    elif fibs == 'stellar':
        # nfib = 19
        userange = np.arange(2, 21, 1)
    # elif fibs == 'laser':
    #     nfib = 1
    #     userange = np.arange(0, 1, 2)
    # elif fibs == 'thxe':
    #     nfib = 1
    #     userange = np.arange(27, 28, 2)
    elif fibs == 'sky3':
        # nfib = 3
        userange = np.arange(21, 24, 1)
    elif fibs == 'sky2':
        # nfib = 2
        userange = np.arange(2)
    elif fibs == 'allsky':
        # nfib = 5
        userange = np.r_[np.arange(2),np.arange(21, 24, 1)]
    else:
        print('ERROR: fibre selection not recognised!!!')
        return

    # Do we want to include extra "fibres" to take care of slope and/or offset?
    # Default is NO for both (as this should probably be already taken care of globally)
    addfibs = 0
    if offset:
        # nfib += 1
        addfibs += 1
    if slope:
        # nfib += 1
        addfibs += 1

    phi = np.zeros((len(x), nfib + addfibs))

    # NOTE: need to turn fibre numbers around here to be correct
    for k, fib in enumerate(sorted(fppo.keys())[::-1]):
        mu = fppo[fib]['mu_fit'][col]
        sigma = fppo[fib]['sigma_fit'][col]
        beta = fppo[fib]['beta_fit'][col]
        # now, I think we actually don't want to evaluate the functional form of the profiles as declared by "fibmodel" at the respective locations,
        # but rather we want to integrate the (highly non-linear) function from the left edge to the right edge of the pixels (co-ordinates are pixel centres!!!)
        if integrate:
            for i in np.arange(len(x)):
                phi[i,k] = fixed_quad(fibmodel, x[i]-0.5, x[i]+0.5, args=(mu, sigma, beta))[0]
                # phi[i, k] = quad(fibmodel, x[i] - 0.5, x[i] + 0.5, args=(mu, sigma, beta))[0]
        else:
            phi[:, k] = fibmodel(x, mu, sigma, beta=beta, alpha=0, norm=0)

    if offset and not slope:
        phi[:, -1] = 1.
        userange = np.append(userange, nfib)
    if slope and not offset:
        phi[:, -1] = x - x[0]
        userange = np.append(userange, nfib)
    if offset and slope:
        phi[:, -2] = 1.
        phi[:, -1] = x - x[0]
        userange = np.append(userange, np.array([nfib,nfib+1]))

    # deprecate phi-array to only use wanted fibres
    phi = phi[:, userange]

    # return normalized profiles
    phinorm = phi / np.sum(phi, axis=0)

    return phinorm

def make_norm_profiles_temp(x, o, col, fibparms, slope=False, offset=False):  
    
    #xx = np.arange(4096)
    
    #same number of fibres for every order, of course
    nfib = 19
    if offset:
        nfib += 1
    if slope:
        nfib += 1
    
    phi = np.zeros((len(x),nfib))
    
    
    mu = fibparms[o]['fibre_03']['mu_fit'](col)
    sigma = fibparms[o]['fibre_03']['sigma_fit'](col)
    beta = fibparms[o]['fibre_03']['beta_fit'](col)
    for k in range(nfib):
        phi[:,k] = fibmodel(x, mu-k*1.98, sigma, beta=beta, alpha=0, norm=0)
    
    if offset:
        phi[:,-2] = 1.
    if slope:
        phi[:,-1] = x - x[0]
    
    #return normalized_profiles
    return phi/np.sum(phi,axis=0)


def make_norm_single_profile_temp(x, o, col, fibparms, slope=False, offset=False):  

    #same number of fibres for every order, of course
    nfib = 1
    if offset:
        nfib += 1
    if slope:
        nfib += 1
    
    phi = np.zeros((len(x),nfib))
    
    
    mu = fibparms[o]['mu'][col]
    sigma = fibparms[o]['sigma'][col]
    beta = fibparms[o]['beta'][col]
    for k in range(nfib):
        phi[:,k] = fibmodel(x, mu, sigma, beta=beta, alpha=0, norm=0)
    
    if offset:
        phi[:,-2] = 1.
    if slope:
        phi[:,-1] = x - x[0]
    
    #return normalized_profiles
    return phi/np.sum(phi,axis=0)


def make_norm_single_profile_simu(x, col, fppo, slope=False, offset=False, fib=None):
    # same number of fibres for every order, of course
    nfib = 1
    if offset:
        nfib += 1
    if slope:
        nfib += 1

    phi = np.zeros((len(x), nfib))

    mu = fppo['fibre_'+fib]['mu_fit'](col)
    sigma = fppo['fibre_'+fib]['sigma_fit'](col)
    beta = fppo['fibre_'+fib]['beta_fit'](col)
    for k in range(nfib):
        phi[:, k] = fibmodel(x, mu, sigma, beta=beta, alpha=0, norm=0)

    if offset:
        phi[:, -2] = 1.
    if slope:
        phi[:, -1] = x - x[0]

    # return normalized_profiles
    return phi / np.sum(phi, axis=0)





def blaze(x, alpha, amp, shift):
    return amp * (np.sin(np.pi * alpha * (x-shift))/(np.pi * alpha * (x-shift)))**2



def compose_matrix(sx,sy,shear,rot,tx,ty):
    m = np.zeros((3,3))
    m[0,0] = sx * math.cos(rot)
    m[1,0] = sx * math.sin(rot)
    m[0,1] = -sy * math.sin(rot+shear)
    m[1,1] = sy * math.cos(rot+shear)
    m[0,2] = tx
    m[1,2] = ty
    m[2,2] = 1
    return m



def center(df, width, height):
    m = compose_matrix(df['scale_x'], df['scale_y'], df['shear'], df['rotation'], df['translation_x'], df['translation_y'])
    xy = m.dot([width, height, 1])
    return xy[0], xy[1]



def polyfit2d(x, y, z, order=3, return_res=False):
    """The result (m) is an array of the polynomial coefficients in the model f  = sum_i sum_j a_ij x^i y^j, 
       has the form m = [a00,a01,a02,a03,a10,a11,a12,a13,a20,.....,a33] for order=3
    """
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i,j) in enumerate(ij):
        G[:,k] = x**i * y**j
    m, res, rank, s = np.linalg.lstsq(G, z)
    if return_res:
        return m, res
    else:
        return m



def test_polyfit2d(x, y, f, deg=3):
    x = np.asarray(x)
    y = np.asarray(y)
    f = np.asarray(f)
    deg = np.asarray(deg)
    vander = polynomial.polyvander2d(x, y, deg)
    vander = vander.reshape((-1,vander.shape[-1]))
    f = f.reshape((vander.shape[0],))
    c = np.linalg.lstsq(vander, f)[0]
    return c.reshape(deg+1)



def polyval2d(x, y, m):
    """
    Returns a 2-dim array of values with the parameters m from 'polyfit2d'.
    e.g.: m = [a00,a01,a02,a03,a10,a11,a12,a13,a20,.....,a33] for order=3
    """
    order = int(np.sqrt(len(m))) - 1
    ij = itertools.product(range(order+1), range(order+1))
    z = np.zeros_like(x)
    for a, (i,j) in zip(m, ij):
        z += a * x**i * y**j
    return z



def fit_poly_surface_2D(x_norm, y_norm, z, weights=None, polytype = 'chebyshev', poly_deg_x=5, poly_deg_y=None, timit=False, debug_level=0):
    """
    Calculate 2D polynomial fit to normalized x and y values.
    Wrapper function for using the astropy fitting library.
    
    INPUT:
    'x_norm'      : x-values (pixels) of all the lines, re-normalized to [-1,+1]
    'm_norm'      : order numbers of all the lines, re-normalized to [-1,+1]
    'z'           : the 2-dim array of 'observed' values
    'weights'     : weights to use in the fitting
    'polytype'    : types of polynomials to use (either '(p)olynomial' (default), '(l)egendre', or '(c)hebyshev' are accepted)
    'poly_deg'    : degree of the polynomials
    'timit'       : boolean - do you want to measure execution run time?
    'debug_level' : for debugging... 
        
    OUTPUT:
    'p'  : coefficients of the best-fit polynomials
    """
    
    if timit:
        start_time = time.time()
    
    if poly_deg_y is None:
        poly_deg_y = poly_deg_x
    
    if polytype.lower() in ['p','polynomial']:
        p_init = models.Polynomial2D(poly_deg_x)
        if debug_level > 0:
            print('OK, using standard polynomials...')
    elif polytype.lower() in ['c','chebyshev']:
        p_init = models.Chebyshev2D(poly_deg_x,poly_deg_y)
        if debug_level > 0:
            print('OK, using Chebyshev polynomials...')
    elif polytype.lower() in ['l','legendre']:
        p_init = models.Legendre2D(poly_deg_x,poly_deg_y)  
        if debug_level > 0:
            print('OK, using Legendre polynomials...')   
    else:
        print("ERROR: polytype not recognised ['(P)olynomial' / '(C)hebyshev' / '(L)egendre']")    
        return
    
    fit_p = fitting.LevMarLSQFitter()  

    with warnings.catch_warnings():
        # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        p = fit_p(p_init, x_norm, y_norm, z, weights=weights)
        
    if timit:
        print('Time elapsed: '+np.round(time.time() - start_time,2).astype(str)+' seconds...')
        
    return p



def find_blaze_peaks(flux,P_id):
    """ find the peaks of the blaze function for each order """
    
    xx = np.arange(len(flux['order_01']))
    
    #first, smooth the spectra a little to remove cosmics or pixel-to-pixel sensitivity variations
    medfiltered = {}
    rough_peaks = {}
    peaks = {}
    for o in sorted(P_id.iterkeys()):
        medfiltered[o] = signal.medfilt(flux[o],9)
        rough_peaks[o] = np.mean(xx[medfiltered[o] == np.max(medfiltered[o])])   # need mean if there is more than one index where flux is max
        ix = rough_peaks[o].astype(int)
        fitrange = xx[ix-500:ix+501]
        #fit 2nd order polynomial to fitrange
        parms = np.poly1d(np.polyfit(fitrange, medfiltered[o][fitrange], 2))
        peakix = signal.argrelextrema(parms(fitrange), np.greater)[0]
        peaks[o] = fitrange[peakix]
        
    return peaks    



def find_nearest(arr,value,return_index=False):
    """
    This routine finds either the index or the value of the element of a 1-dim array that is closest to a given value.
    
    INPUT:
    'arr'   : the array in which to look 
    'value' : the value the closest thing to which you want to find
    'return_index' : boolean - do you want to return the index or the value?
    
    OUTPUT:
    'idx'      : the index of the closest thing to 'value' within 'arr' (if 'return_index' is set to TRUE)
    'arr[idx]' : the value within 'arr' that is closest to 'value' (if 'return_index' is set to FALSE)
    """
    idx = (np.abs(arr-value)).argmin()
    if return_index:
        return idx
    else:
        return arr[idx]        



def binary_indices(x):
    """
    Calculates and returns the non-zero indices in a binary representation of x, in order of increasing powers of 2.
    
    EXAMPLES:

    x = 11 = '1011'        --->   returns [0, 1, 3]      --->   because 11 = 2^1 + 2^1 + 2^3
    x = 153 = '10011001'   --->   returns [0, 3, 4, 7]   --->   because 153 = 2^0 + 2^3 + 2^4 + 2^7
    x = 201 = '11001001'   --->   returns [0, 3, 6, 7]   --->   because 153 = 2^0 + 2^3 + 2^6 + 2^7
    """
    binum = np.binary_repr(x)
    liste = list(reversed(binum))
    intlist = [int(i) for i in liste]
    return np.flatnonzero(np.array(intlist))
    
    
   
def single_sigma_clip(x, tl, th=None, centre='median', return_indices=False):
    """
    Perform sigma-clipping of 1D array.
    
    INPUT:
    'x'              : the 1D array to be sigma-clipped
    'tl'             : lower threshold (in terms of sigma)
    'th'             : higher threshold (in terms of sigma) (if only one threshold is given then th=tl=t)
    'centre'         : method to determine the centre ('median' or 'mean')
    'return_indices' : boolean - do you also want to return the index masks of the unclipped and clipped data points?
    
    OUTPUT:
    'x'  : the now sigma-clipped array

    TODO:
    implement return_indices keyword
    """
    
    #make sure both boundaries are defined
    if th is None:
        th = tl
    
    clipped = x.copy()

    # want to return indices, not boolean array with True and False
    if return_indices:
        indices = np.arange(len(x))
    
    rms = np.std(clipped)
    if centre.lower() == 'median':
        bad_high = clipped - np.median(clipped) > th*rms
        bad_low = np.median(clipped) - clipped > tl*rms
    elif centre.lower() == 'mean':
        bad_high = clipped - np.mean(clipped) > th*rms
        bad_low = np.mean(clipped) - clipped > tl*rms
    else:
        print('ERROR: Method for computing centre must be "median" or "mean"')
        return
    
    goodix = ~bad_high & ~bad_low
    badix = ~goodix
    clipped = clipped[goodix]

    if return_indices:
        goodix = indices[goodix]
        badix = indices[badix]
        return clipped, goodix.astype(int), badix.astype(int)
    else:
        return clipped   
   
   
    
def sigma_clip(x, tl, th=None, centre='median', return_indices=False):
    """
    Perform sigma-clipping of 1D array.
    
    INPUT:
    'x'              : the 1D array to be sigma-clipped
    'tl'             : lower threshold (in terms of sigma)
    'th'             : higher threshold (in terms of sigma) (if only one threshold is given then th=tl=t)
    'centre'         : method to determine the centre ('median' or 'mean')
    'return_indices' : boolean - do you also want to return the index masks of the unclipped and clipped data points?
    
    OUTPUT:
    'x'  : the now sigma-clipped array

    TODO:
    implement return_indices keyword
    """
    
    #make sure both boundaries are defined
    if th is None:
        th = tl
    
    clipped = x.copy()

    all_indices = np.arange(len(x))
    indices = np.arange(len(x))
    badix = []
    #goodix = np.ones(len(x),dtype='bool')
    
    while True:    
        rms = np.std(clipped)
        if centre.lower() == 'median':
            bad_high = clipped - np.median(clipped) > th*rms
            bad_low = np.median(clipped) - clipped > tl*rms
        elif centre.lower() == 'mean':
            bad_high = clipped - np.mean(clipped) > th*rms
            bad_low = np.mean(clipped) - clipped > tl*rms
        else:
            print('ERROR: Method for computing centre must be "median" or "mean"')
            return
        new_goodix = ~bad_high & ~bad_low
        if np.sum(~new_goodix) == 0:
            break
        else:
            clipped = clipped[new_goodix]
            badix = np.r_[badix,indices[~new_goodix]]
            #badix.append(indices[~new_goodix])
            indices = indices[new_goodix]

    if return_indices:
        goodix = np.array(sorted(list(set(all_indices) - set(badix))))
        badix = np.array(sorted(badix))
        return clipped, goodix.astype(int), badix.astype(int)
    else:
        return clipped
    
    
    
def offset_pseudo_gausslike(x, G_amplitude, L_amplitude, G_center, L_center, G_sigma, L_sigma, beta):
    """ similar to Pseudo-Voigt-Model (e.g. see here: https://lmfit.github.io/lmfit-py/builtin_models.html), 
        but allows for offset between two functions and allows for beta to vary """
    G = G_amplitude * fibmodel(x, G_center, G_sigma, beta=beta, alpha=0, norm=1)
    L = L_amplitude * L_sigma/((L_center - L_sigma)**2 + L_sigma**2)
    return G+L
    
     
    
def get_iterable(x):
    """ make any python object an 'iterable' so that you can loop over it and your code does not crash, even if it's just a number """
    if isinstance(x, collections.Iterable):
        return x
    else:
        return (x,)    
    
    
        
def get_datestring():
    """ get the current date in a string of the format: 'YYYYMMDD' """
    now = datetime.datetime.now()
    dum = str(now)[:10]
    datestring = ''.join(dum.split('-'))
    return datestring



def get_mean_snr(flux, err=None, per_order=False):
    """ 
    Calculate the mean SNR of the quick-extracted 1-dim spectrum.
    Treat all negative flux values as zero for the purpose of this.
    
    INPUT:
    'flux'      : dictionary of the 1-dim extracted spectrum (keys = orders); or numpy array
    'err'       : dictionary of the corresponding uncertainties (if not provided, the SQRT of the flux is used by default); or numpy array
    'per_order' : boolean - do you want to return the mean SNR of each order? 
    
    OUTPUT:
    'snr_ord' : the mean snr (per 'super-pixel' / collapsed pixel) per order
    'snr'     : the mean snr (per 'super-pixel' / collapsed pixel) of the input spectrum
    
    MODHIST:
    18/05/2018 - CMB create
    30/05/2018 - CMB added 'per_order' keyword
    """
    
    # data in the form of numpy array (ie from fits files)
    if flux.__class__ == np.ndarray:
        
        #estimate errors if not provided
        if err is None:
            print('WARNING: error-array not provided! Using SQRT(flux) as an estimate instead...')
            print
            err = np.sqrt(np.maximum(10., flux))   # RON is sth like 3 e-/pix, so use 10. as a minimum value for the variance
        
        if per_order:
            snr_ord = []
            for o in range(flux.shape[0]):
                snr_ord.append(np.mean(np.maximum(0., flux[o,:] / np.maximum(np.sqrt(10.), err[o,:]))))
        else:
            snr = np.mean(np.maximum(0., flux)/np.maximum(np.sqrt(10.), err))    
    
    # data in the form of a dictionary
    elif flux.__class__ == dict:
        
        #estimate errors if not provided
        if err is None:
            print('WARNING: error-array not provided! Using SQRT(flux) as an estimate instead...')
            print
            err = {}
            for o in sorted(flux.keys()):
                mask1 = flux[o] < 0.
                flux[o][mask1] = 0.
                err[o] = np.sqrt(np.maximum(10.,flux[o]))
                
        snr_ord = []
        
        for o in sorted(flux.keys()):
            snr_ord.append(np.mean(np.maximum(0., flux[o] / np.maximum(np.sqrt(10.), err[o]))))
        
        snr =  np.mean(snr_ord)
    else:
        print('ERROR: data type / variable class not recognized')
        return
    
    if per_order:
        return np.array(snr_ord)
    else:
        return snr



def central_parts_of_mask(mask):
    """
    This routine reduces the True parts of an order mask to only the large central posrtion if there are multiple True parts.
    These are the parts we want to use for the cross-correlation to get RVs.
    
    INPUT:
    'mask'    : mask dictionary from "make_mask_dict" (keys = orders)
    
    OUTPUT:
    'cenmask' : mask containing the central true parts only
    
    MODHIST:
    01/06/2018 - CMB create    
    03/08/2018 - fixed bug if mask is asymmetric - now requires closest upstep location to be to the left of the order centre and nearest downstep location to 
                 be to the right of the order centre
    """
    
    cenmask = {}
    #loop over all masks
    for o in sorted(mask.keys()):
        ordmask = mask[o]
        if ordmask[len(ordmask)//2]:
            upstep_locations = np.argwhere(np.diff(ordmask.astype(int)) == 1)
            downstep_locations = np.argwhere(np.diff(ordmask.astype(int)) == -1)
            cenmask[o] = ordmask.copy()
            if len(upstep_locations) >= 1:
                up = np.squeeze(find_nearest(upstep_locations[upstep_locations < len(ordmask)//2],len(ordmask)//2,return_index=False))
                cenmask[o][:up+1] = False
            if len(downstep_locations) >= 1:
                down = np.squeeze(find_nearest(downstep_locations[downstep_locations > len(ordmask)//2],len(ordmask)//2,return_index=False))
                cenmask[o][down+1:] = False           
        else:
            print('ERROR: order centre is masked out!!!')
            return

    return cenmask



def short_filenames(file_list):
    dum = [longname.split('/')[-1] for longname in file_list]
    fnarr = ['.'.join(fn.split('.')[0:-1]) for fn in dum]
    return fnarr




def correct_orientation(img, verbose=False):
    """
    bring image to same orientation as the simulated spectra, ie wavelength decreases from left to right and bottom to top
    """
    
    ny, nx = img.shape
    
    if (ny,nx) == (4112,4202):
        img = np.fliplr(img.T)
    elif (ny,nx) == (4202,4112):
        # this means the image is already in the correct orientation
        if verbose:
            print('The image is already in the correct orientation!')
    else:
        print('ERROR: file shape not correct!!!')
    
    return img





def find_maxima(data, gauss_filter_sigma=0., min_peak=0.1, return_values=0):
    """
    not currently used!!!
    """
    # smooth image slightly for noise reduction
    smooth_data = ndimage.gaussian_filter(data, gauss_filter_sigma)
    # find all local maxima
    peaks = np.r_[True, smooth_data[1:] > smooth_data[:-1]] & np.r_[smooth_data[:-1] > smooth_data[1:], True]
    # only use peaks higher than a certain threshold
    idx = np.logical_and(peaks, smooth_data > min_peak * np.max(smooth_data))
    maxix = np.arange(len(data))[idx]
    maxima = data[maxix]
    
    if return_values != 0:
        return maxix,maxima
    else:
        return maxix



def affine_matrix(scale_x=1, scale_y=1, theta=0, dx=0, dy=0, shear_x=0, shear_y=0):
    """
    theta in degs clockwise
    
    Order:
    (1) rotation
    (2) translation
    (3) scaling
    (4) shear in x
    (5) shear in y
    """
    #m = np.zeros((3,3))
    #m[zeile,spalte] = m[row,column]
    
    m_scale = np.eye(3)
    m_scale[0,0] = scale_x
    m_scale[1,1] = scale_y
    
    m_rot = np.eye(3)
    m_rot[0,0] = np.cos(np.deg2rad(theta))
    m_rot[0,1] = np.sin(np.deg2rad(theta))
    m_rot[1,0] = -np.sin(np.deg2rad(theta))
    m_rot[1,1] = np.cos(np.deg2rad(theta))
    
    m_trans = np.eye(3)
    m_trans[0,2] = dx
    m_trans[1,2] = dy
    
    m_shear_x = np.eye(3)
    m_shear_x[0,1] = shear_x
    
    m_shear_y = np.eye(3)
    m_shear_y[1,0] = shear_y
    
    m = np.matmul(m_shear_y, np.matmul(m_shear_x, np.matmul(m_scale, np.matmul(m_trans, m_rot))))
    
    return m



def quick_bg_fix(raw_data, npix=4112):
    left_xx = np.arange(npix/2)
    right_xx = np.arange(npix/2, npix)
    left_bg = ndimage.minimum_filter(ndimage.gaussian_filter(raw_data[:npix/2],3), size=100)
    right_bg = ndimage.minimum_filter(ndimage.gaussian_filter(raw_data[npix/2:],3), size=100)
    data = raw_data.copy()
    data[left_xx] = raw_data[left_xx] - left_bg
    data[right_xx] = raw_data[right_xx] - right_bg
    return data



def orientation_test_unautomated(stellar_list, white_list, laser_list, path=None):
    """

    :param stellar_list: the input stellar frames
    :param white_list:  the input bias frames
    :param laser_list: the input laser frames
    :param path: the input path defined at the start of the pipeline
    :return: the three input lists but with frames correctly oriented

    ## JP (11/2/2019): currently unautomated, and only corrects by 270 degrees as per our initial MQ simulations
    ## also only needed on mac, as the importing process inherently flips and rotates the image, on windows, this
    ## issue was not encountered
    """

    if path == None:
        print('ERROR: please define an input path!!!')

    print('correcting ' + str(len(white_list)) + ' flat frames...')
    for n,fn in enumerate(sorted(white_list)):
        img = pyfits.getdata(fn)
        img = scipy.ndimage.interpolation.rotate(img, 270)  # added to correct for our simulations
        ## get headers and comments to append to file
        img_header = pyfits.getheader(fn)
        naxis_1_comment = img_header.comments['NAXIS1']
        naxis_2_comment = img_header.comments['NAXIS2']
        extend_comment = img_header.comments['EXTEND']
        exptime_header = img_header['exptime']
        exptime_comment = img_header.comments['exptime']
        # comment_header = img_header['comment']
        # comment_comment = img_header.comments['comment']
        #hier_spec_header = img_header['hierarch spectrograph']
        #hier_spec_comment = img_header.comments['hierarch spectrograph']
        #fib_header = img_header['fiber_1']
        #fib_comment = img_header.comments['fiber_1']
        ## write to file
        outfn = path + 'corrected_' + fn.split('/')[-1]
        pyfits.writeto(outfn, img, clobber=True)
        ## set headers and comments
        pyfits.setval(outfn, 'NAXIS1', comment=naxis_1_comment)
        pyfits.setval(outfn, 'NAXIS2', comment=naxis_2_comment)
        pyfits.setval(outfn, 'EXTEND', comment=extend_comment)
        pyfits.setval(outfn, 'EXPTIME', value=exptime_header, comment=exptime_comment)
        # pyfits.setval(outfn, 'COMMENT', value=comment_header, comment=comment_comment)
        #pyfits.setval(outfn, 'HIERARCH SPECTROGRAPH', value=hier_spec_header, comment=hier_spec_comment)
        #pyfits.setval(outfn, 'FIBER_1', value=fib_header, comment=fib_comment)
        pyfits.setval(outfn, 'HISTORY',
                      value='   CORRECTLY ORIENTED frame - created ' + time.strftime("%Y-%m-%d %H:%M:%S",
                                                                                     time.gmtime()) + ' (GMT)')

    print('correcting ' + str(len(stellar_list)) + ' stellar frames...')
    for n,fn in enumerate(sorted(stellar_list)):
        img = pyfits.getdata(fn)
        img = scipy.ndimage.interpolation.rotate(img, 270)  # added to correct for our simulations
        ## get headers and comments to append to file
        img_header = pyfits.getheader(fn)
        naxis_1_comment = img_header.comments['NAXIS1']
        naxis_2_comment = img_header.comments['NAXIS2']
        extend_comment = img_header.comments['EXTEND']
        exptime_header = img_header['exptime']
        exptime_comment = img_header.comments['exptime']
        # comment_header = img_header['comment']
        # comment_comment = img_header.comments['comment']
        #hier_spec_header = img_header['hierarch spectrograph']
        #hier_spec_comment = img_header.comments['hierarch spectrograph']
        #fib_header = img_header['fiber_1']
        #fib_comment = img_header.comments['fiber_1']
        ## write to file
        outfn = path + 'corrected_' + fn.split('/')[-1]
        pyfits.writeto(outfn, img, clobber=True)
        ## set headers and comments
        pyfits.setval(outfn, 'NAXIS1', comment=naxis_1_comment)
        pyfits.setval(outfn, 'NAXIS2', comment=naxis_2_comment)
        pyfits.setval(outfn, 'EXTEND', comment=extend_comment)
        pyfits.setval(outfn, 'EXPTIME', value=exptime_header, comment=exptime_comment)
        # pyfits.setval(outfn, 'COMMENT', value=comment_header, comment=comment_comment)
        #pyfits.setval(outfn, 'HIERARCH SPECTROGRAPH', value=hier_spec_header, comment=hier_spec_comment)
        #pyfits.setval(outfn, 'FIBER_1', value=fib_header, comment=fib_comment)
        pyfits.setval(outfn, 'HISTORY',
                      value='   CORRECTLY ORIENTED frame - created ' + time.strftime("%Y-%m-%d %H:%M:%S",
                                                                                     time.gmtime()) + ' (GMT)')

    print('correcting ' + str(len(laser_list)) + ' laser frames...')
    for n,fn in enumerate(sorted(laser_list)):
        img = pyfits.getdata(fn)
        img = scipy.ndimage.interpolation.rotate(img, 270)  # added to correct for our simulations
        ## get headers and comments to append to file
        img_header = pyfits.getheader(fn)
        naxis_1_comment = img_header.comments['NAXIS1']
        naxis_2_comment = img_header.comments['NAXIS2']
        extend_comment = img_header.comments['EXTEND']
        exptime_header = img_header['exptime']
        exptime_comment = img_header.comments['exptime']
        # comment_header = img_header['comment']
        # comment_comment = img_header.comments['comment']
        #hier_spec_header = img_header['hierarch spectrograph']
        #hier_spec_comment = img_header.comments['hierarch spectrograph']
        #fib_header = img_header['fiber_1']
        #fib_comment = img_header.comments['fiber_1']
        ## write to file
        outfn = path + 'corrected_' + fn.split('/')[-1]
        pyfits.writeto(outfn, img, clobber=True)
        ## set headers and comments
        pyfits.setval(outfn, 'NAXIS1', comment=naxis_1_comment)
        pyfits.setval(outfn, 'NAXIS2', comment=naxis_2_comment)
        pyfits.setval(outfn, 'EXTEND', comment=extend_comment)
        pyfits.setval(outfn, 'EXPTIME', value=exptime_header, comment=exptime_comment)
        # pyfits.setval(outfn, 'COMMENT', value=comment_header, comment=comment_comment)
        #pyfits.setval(outfn, 'HIERARCH SPECTROGRAPH', value=hier_spec_header, comment=hier_spec_comment)
        #pyfits.setval(outfn, 'FIBER_1', value=fib_header, comment=fib_comment)
        pyfits.setval(outfn, 'HISTORY',
                      value='   CORRECTLY ORIENTED frame - created ' + time.strftime("%Y-%m-%d %H:%M:%S",
                                                                                     time.gmtime()) + ' (GMT)')

    ## input corrected data
    stellar_list = glob.glob(path + 'corrected_mq_blackbody*.fits')  # mq simulations
    laser_list = glob.glob(path + 'corrected_mq_etalon.fits')  # mq simulations
    white_list = glob.glob(path + 'corrected_mq_flat*.fits')  # mq simulations

    print('all frames are in the correct orientation!!!')

    return stellar_list, white_list, laser_list