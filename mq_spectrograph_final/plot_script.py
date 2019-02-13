'''
Created on 29 Jan 2019

@author: Jacob Parnell
'''

### plotting raw/smoothed spectra and pixel sensitivity
## importing statements - do not alter
import matplotlib.pyplot as plt
from matplotlib import use
import glob
import astropy.io.fits as pyfits
import numpy as np
from scipy import ndimage
from IPython.display import Image # added so that the code will save plots to file
import os
import time

from flat_fielding import onedim_pixtopix_variations_single_order, onedim_pixtopix_variations

## loads the extracted 2-d images
## currently only for quick method
plot_path = '/Users/Jacob/Desktop/data_for_mq_final/' ## THIS NEEDS TO CHANGE BETWEEN USERS
plot_list = sorted(glob.glob(plot_path + '*corrected_*' + '*quick_extracted.fits')) # mq simulations w/ corrections
plot_list = sorted(glob.glob(plot_path + '*00_quick_extracted.fits')) # mq simulations w/ corrections

stellar_object = np.arange(0,len(plot_list))
start_time = time.time()
## generated in basic_reduction_script - loads locally saved extracted flux dictionary (basic_reduction_script.py only needs to be run once)
flux_load = np.load(plot_path + 'flux.npy') .item()
smoothed_flat, pix_sens = onedim_pixtopix_variations(flux_load) # changed function to this so it works in correct order

## check if directories exist, and if not, create them to save our plots
## if statement below should be false first time code is run, so directories are made (/plots/ is generated in basic_reduction_script.py)
## any subsequent re-run of the code will skip the if statement
if os.path.isdir(plot_path + '/plots/raw_extracted_spectra') == False:
    os.mkdir(plot_path + 'plots/raw_extracted_spectra')
    os.mkdir(plot_path + 'plots/pixel_sensitivity')
    os.mkdir(plot_path + 'plots/smoothed_extracted_spectra')

## iterate over all elements imported from file
for j in stellar_object:

    solar_spectrum = pyfits.getdata(plot_list[j])
    nx,ny = solar_spectrum.shape
    plot_name = plot_list[j].split('/')

    ## check if directory exists first, if not, create it, then save plot to file
    if os.path.isdir(plot_path + '/plots/raw_extracted_spectra/' + plot_name[-1].split('.')[0]) == False:
        os.mkdir(plot_path + 'plots/raw_extracted_spectra/' + plot_name[-1].split('.')[0])
    if os.path.isdir(plot_path + '/plots/pixel_sensitivity/' + plot_name[-1].split('.')[0]) == False:
        os.mkdir(plot_path + 'plots/pixel_sensitivity/' + plot_name[-1].split('.')[0])
    if os.path.isdir(plot_path + '/plots/smoothed_extracted_spectra/' + plot_name[-1].split('.')[0]) == False:
        os.mkdir(plot_path + 'plots/smoothed_extracted_spectra/' + plot_name[-1].split('.')[0])

    print("Plotting: " + plot_name[-1].split('/')[0])

    ## iterate over each row of each imported file
    for i, o in enumerate(sorted(flux_load.keys())):
        one_dim_solar = solar_spectrum[i, :] # cuts imported file into each 1-dim spectra
        nx = np.arange(0, ny) # set to the length of the detector (size:4096x4096)

        ############################################# RAW SPECTRUM ####################################################

        ## 1-dim raw extracted spectra with 5th order polynomial fit (other orders have worsened fit)
        ## (x - axis: pixels, y-axis: intensity)

        save_plots = plot_path + 'plots/raw_extracted_spectra/' + plot_name[-1].split('.')[0] + '/' # for plotting
        ## calculate polynomials
        z = np.polyfit(nx, one_dim_solar, 5)
        f = np.poly1d(z)
        ## calculate new x's and new y's
        x_new = np.linspace(nx[0], nx[-1], len(nx))
        y_new = f(x_new)
        ## plot
        plt.figure()
        plot_name = plot_list[j].split('/')
        plt.xlim(0, ny)
        plt.xlabel('Pixels')
        plt.ylabel('Intensity')
        plt.title(plot_name[-1].split('.')[0] + ' : ' + 'Order: ' + str(i+2))
        plt.plot(nx, one_dim_solar, x_new, y_new, 'r', linewidth=2)
        plt.savefig(save_plots + plot_name[-1].split('.')[0] + '_order_' + str(i+2) + '.png')
        #plt.show() # here for debugging
        plt.close()

        ############################################ PIXEL SENSITIVITY ################################################

        save_plots = plot_path + 'plots/pixel_sensitivity/' + plot_name[-1].split('.')[0] + '/' # for plotting
        ## find the pixel sensitivity for each order
        ## plot
        plt.figure()
        plt.xlim(0, ny)
        plt.xlabel('Pixels')
        plt.ylabel('Relative Sensitivity')
        plt.title('Pixel Sensitivity : ' + 'Order: ' + str(i + 2))
        plt.plot(nx, pix_sens[o])
        plt.savefig(save_plots + plot_name[-1].split('.')[0] + '_pixel_sens_order_' + str(i + 2) + '.png')
        #plt.show() # here for debugging
        plt.close()

        ############################################# SMOOTH SPECTRUM #################################################

        save_plots = plot_path + 'plots/smoothed_extracted_spectra/' + plot_name[-1].split('.')[0] + '/' # for plotting
        ## 1-dim raw extracted spectra divided by pixel sensitivity, to produce a 'smoother' spectrum
        smooth_1dim = one_dim_solar / pix_sens[o]
        ## plot
        plt.figure()
        plt.xlim(0, ny)
        plt.xlabel('Pixels')
        plt.ylabel('Intensity')
        plt.title('Smoothed: ' + plot_name[-1].split('.')[0] + ' : ' + 'Order: ' + str(i+2))
        plt.plot(nx, smooth_1dim, x_new, y_new, 'r', linewidth=2)
        plt.savefig(save_plots + plot_name[-1].split('.')[0] + '_smoothed_order_' + str(i + 2) + '.png')
        #plt.show() # here for debugging
        plt.close()

        #break # can uncomment the break to test for the 2nd order of first element (need to set i=j=0 before running code)
    # break # can uncomment the break to test for the 2nd order of first element (need to set i=j=0 before running code)

print('Elapsed time: '+str(time.time() - start_time)+' seconds')