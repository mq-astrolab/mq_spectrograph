'''
Created on 31 Jan 2019

@author: Jacob Parnell
'''

### polynomial diagnostic plots ###
## importing statements - do not alter
import matplotlib.pyplot as plt
from matplotlib import use
import glob
import astropy.io.fits as pyfits
import numpy as np
from scipy import ndimage
import os

## set paths
path = '/Users/Jacob/Desktop/data_for_mq/' ## THIS NEEDS TO CHANGE BETWEEN USERS
save_plots = path + 'plots/'
## generated in basic_reduction_script - loads locally saved master white (basic_reduction_script.py only needs to be run once)
MW_load = np.loadtxt('MW')

## check if directories exist, and if not, create them to save our plots
if os.path.isdir(save_plots + 'diagnostic_plots/') == False:
    os.mkdir(save_plots + 'diagnostic_plots/')
save_plots = save_plots + 'diagnostic_plots/'

## pre-plotting
flat = MW_load
#deg_polynomial=2
gauss_filter_sigma=3
#min_peak=0.05

ny,nx = flat.shape
filtered_flat = ndimage.gaussian_filter(flat.astype(np.float), gauss_filter_sigma)
data = filtered_flat[:, int(nx / 2)]

#####################################################################################################################

data_ord_2 = data[220:280] # will print out the 2nd peak when plotted

# fitting a second order, or higher - higher seems to fit better
x = np.arange(0,60)
z = np.polyfit(x, data_ord_2, 18) # change data_ord_ depending on order
f = np.poly1d(z)

y_new = f(x)
plt.plot(x, data_ord_2, x, y_new) # change data_ord_ depending on order
plt.savefig(save_plots + 'poly_test_order_02.png')
plt.close()

#####################################################################################################################

data_ord_10 = data[1305:1365] # will print out the 10th peak when plotted

# fitting a second order, or higher - higher seems to fit better
x = np.arange(0,60)
z = np.polyfit(x, data_ord_10, 18) # change data_ord_ depending on order
f = np.poly1d(z)

y_new = f(x)
plt.plot(x, data_ord_10, x, y_new) # change data_ord_ depending on order
plt.savefig(save_plots + 'poly_test_order_10.png')
plt.close()

#####################################################################################################################

data_ord_20 = data[2379:2439] # will print out the 20th peak when plotted

# fitting a second order, or higher - higher seems to fit better
x = np.arange(0,60)
z = np.polyfit(x, data_ord_20, 18) # change data_ord_ depending on order
f = np.poly1d(z)

y_new = f(x)
plt.plot(x, data_ord_20, x, y_new) # change data_ord_ depending on order
plt.savefig(save_plots + 'poly_test_order_20.png')
plt.close()

#####################################################################################################################

data_ord_30 = data[3218:3278] # will print out the 30th peak when plotted

# fitting a second order, or higher - higher seems to fit better
x = np.arange(0,60)
z = np.polyfit(x, data_ord_30, 18) # change data_ord_ depending on order
f = np.poly1d(z)

y_new = f(x)
plt.plot(x, data_ord_30, x, y_new) # change data_ord_ depending on order
plt.savefig(save_plots + 'poly_test_order_30.png')
plt.close()

#####################################################################################################################

data_ord_40 = data[3880:3940] # will print out the 40th peak when plotted

# fitting a second order, or higher - higher seems to fit better
x = np.arange(0,60)
z = np.polyfit(x, data_ord_40, 18) # change data_ord_ depending on order
f = np.poly1d(z)

y_new = f(x)
plt.plot(x, data_ord_40, x, y_new) # change data_ord_ depending on order
plt.savefig(save_plots + 'poly_test_order_40.png')
plt.close()

#####################################################################################################################
## COULD USE THIS CODE AS THIS IS THE POLYNOMIAL THAT THE CODE USES TO FIND THE PEAKS ##
## FIT NOT RIGHT ##

#peaks = np.r_[True, data[1:] > data[:-1]] & np.r_[data[:-1] > data[1:], True]
#idx = np.logical_and(peaks, data > min_peak * np.max(data))
#maxima = np.arange(ny)[idx]

# filter out maxima too close to the boundary to avoid problems
#maxima = maxima[maxima > 3]
#maxima = maxima[maxima < ny - 3]

#n_order = len(maxima)
#orders = np.zeros((n_order, nx))
# because we only want to use good pixels in the fit later on
#mask = np.ones((n_order, nx), dtype=bool)

#P = []
#xx = np.arange(nx)
#for i in range(len(orders)):
#    # weighted
#    filtered_flux_along_order = np.zeros(nx)
#    for j in range(nx):
#        # filtered_flux_along_order[j] = filtered_flat[o[j].astype(int),j]    #that was when the loop reas: "for o in orders:"
#        filtered_flux_along_order[j] = filtered_flat[orders[i, j].astype(int), j]
#        filtered_flux_along_order[filtered_flux_along_order < 1] = 1
#        # w = 1. / np.sqrt(filtered_flux_along_order)   this would weight the order centres less!!!
#        w = np.sqrt(filtered_flux_along_order)
#        p = np.poly1d(np.polyfit(xx[mask[i, :]], orders[i, mask[i, :]], deg_polynomial, w=w[mask[i, :]]))
#    P.append(p)

#poly_load = np.load(path + 'P_id.npy') .item()
#poly_order = poly_load['order_02'] # test for the second order - these are the polynomials the code uses trace these peaks

## printing polynomials used by the code - otherwise a polynomial of second order is passed
#x = np.arange(0,60)
#y_new = poly_order(x)
#plt.plot(x, data_ord_10, x, y_new) # change data_ord_ depending on order
#plt.show()
