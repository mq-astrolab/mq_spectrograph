'''
Created on 29 Jan 2019

@author: Jacob Parnell



## documented changes in reduction script ##
## from basic veloce reduction script to MQ single-fibre adaptation ##
# as of 29 Jan 2019 - will run quick extraction method, will not run optimal extraction

############## JACOB'S CHANGES (12/2/19) ##############

(1) basic_reduction_script.py:
    Commented and tidied up for presentation.

(2) lines 346-356 and 612-623 in extraction.py:
    The path for variable 'fibparms' under the 'if simu:' condition is now set to the parameter fibparms, which
    is a variable passed into the function by the user that is defined in the reduction notebook. Currently is
    only used if the optimal method of extraction is called, but we can set it to 'None' when the quick method
    is utilised. Note that these fibre parameters reside in a folder called 'data' and should reside in the
    directory in which the reduction notebook exists.

(3) lines 200 and 258 in process_scripts.py:
    Added passed parameter 'simu=simu', and in the process_science_images function, which is utilised correctly
    when simu is defined in the reduction notebook. Assumes a default that the data that is passed is not
    simulated data.

(4) line 737/738 in calibration.py:
    Commented out 'TOTALEXP' and uncommented 'exptime', must check if real data have these headers. We are assuming
    that the MQ spectrograph will output data that have these exptime headers, as our initial test simulations
    from Echelle++ have these exptime headers. If an error is thrown where the header does not exist, there is an
    error code that prints to say that there is no header and that the code continues, but could also let the
    demonstrator debug and manually edit the student's data to change the header to exptime (if named differently).

(5) line 186-189 in basic_reduction_script:
    Parameter fibs=single for our single fibre veloce adaptation.

(6) lines 68-79 in calibration.py:
    Defined function altered to include int() as error raises in Python 3, not in Python 2.

(7) lines 53-55 in basic_reduction_script.py:
    Students need to be weary of the files that they are importing, depending on the naming convention of the
    files chosen. The code we have provided works so that the new lists do not import extracted files if the entire
    pipeline is run more than once. But is always good to double check when students re-run the code, as our
    naming convention could differ from theirs. This is also only as the extracted files are saved in the directory
    that the reduction notebook and input files are located in.

(8) lines 52/53, 217/218, 274/275, 364/365, 631/632, 984/985, 1151/1152 and 1158/1159 in extraction.py:
    Added lines of code to skip over 'order_01'. This is only as there was an issue that arose in the original
    pipeline that was developed for Veloce. It required a 'fix' as the order did not fall on the chip entirely.
    This also may not be an issue that the MQ spectrograph will face, but is skipped in either case. It
    can be removed if needed and if the first order is wanted to be extracted.

(9) lines 563/564, 636/637 and 702/703 in order_tracing.py:
    Added lines of code to skip over 'order_01'. This is only as there was an issue that arose in the original
    pipeline that was developed for Veloce. It required a 'fix' as the order did not fall on the chip entirely.
    This also may not be an issue that the MQ spectrograph will face, but is skipped in either case. It
    can be removed if needed and if the first order is wanted to be extracted.

(10) plot_script.py:
    Generated a plot script that plots the raw extracted spectra, one order at a time. It will also utilise
    the flat_fielding.py script to convert the 2-dim extracted images into 1-dim spectra, and calculate the pixel
    sensitivity for each order and plot that. Will then use the raw data and the pixel sensitivity to then create
    a smoothed spectrum. This smoothed spectrum needs to be refined and checked if the process is implemented
    correctly (12/2/19).

(11) line 140 in basic_reduction_script.py:
    The extracted flux is saved to file so that the plot_script.py file (or notebook) can be run, without having
    to re-run the reduction script.

(12) lines 310-323 in order_tracing:
    Edited the if savefile function so that you can save the stripes and indices as .npy files if you wanted to.
    Was initially used to debug a problem in the code, but as it is not imperative to the actual reduction,
    the default is False so the code will not run. It will if the user inputs True into the notebook via the
    parameter savefiles=True, path=path, and obsname=obsname. Obsname here is not defined in the user input section
    but it can be - a little tedious, but again not at all necessary to the extraction process.

(13) line 124 in basic_reduction_script.py:
    Added a line of code so that the extraction plots for order tracing are saved to file. This is via the
    save_plots parameter. This needs to be defined before the function is called otherwise an error will be raised.
    Need to add in an error statement (12/2/19). This parameter is then used in lines 65, 83, and 173 in
    order_tracing.py.

(14) lines 32-47 in basic_reduction_script.py:
    Added to check if these directories exist, and if not, to generate them. This is only so that the plots that
    are generated, are also saved to file and can be easily located. Both windows and mac examples are provided.

(15) line 124 in basic_reduction_script.py:
    Changed the min_peak parameter from 0.25 to 0.2 so that for our simulations, the function includes the lower
    S/N peaks and counts all 80 orders, rather than 67. Can tweak this depending on data or remove this passed
    parameter incase real data the students get is much less than this, so it can revert to the default which is
    min_peak = 0.05. Note that for the students, the min_peak parameter is set before the function is called in
    the notebook, so in the function, it should say min_peak=min_peak.

(16) line 124 in basic_reduction_script.py:
    Set maskthresh to 30, not default of 100. It is a threshold in terms of ADU below which we mask the particular
    pixel column. For real observations, you want to make sure that what you trace is actually the order and not
    just noise which happens towards the edges of the orders (and for edges of chips in cross dispersion direction).
    Simulations on the other hand, have this value set quite low, even to zero, depending on the SNR and stray
    light in the real spectra, but all we need is to be satisfied that the fitting is good. Note that for the
    students, the maskthresh parameter is set before the function is called in the notebook, so in the function,
    it should say maskthresh=maskthresh.

(17) basic_reduction_script.py:
    Cleaned up and finalised. It is going to be used as the background to support the process outlined in the
    actual reduction notebook.

(18) lines 57-61 in basic_reduction_script.py:
    This is an image orientation function that is located in helper_functions.py. It is there to remove an issue
    that was encountered on my Mac, which was that the input files were rotated when handled by PyCharm/Notebooks.
    It is a test that is currently unautomated and will only correct the orientation for our first set of MQ
    simulations (rotating CW by 270 degrees). This is something that can be automated if necessary, but is only
    an issue that has occurred on a Macbook. Windows systems appear to input the files just fine.

(19) line 630 in extraction.py:
    'for ord in sorted(stripes.iterkeys()):' changed to 'for i,ord in sorted(enumerate(stripe_indices.keys())):'. This
    is to combat the Python 2 vs Python 3 issue, but is only run if the optimal method of extraction is used. Should be
    extended to line 363 if the optimal method is called.

(20) import_functions.py:
    Generated a .py file that contains all the import statements and functions that is utilised in the entire pipeline.
    This is the first line of code that is called in the reduction notebook, and allows the code to be run entirely.

(21) unused_subroutines:
    Generated a sub-directory nested within this directory that contains all the sub-routines that have not yet been
    utilised in our version of the pipeline. There are scripts in there that can be called to calculate the wavelength
    function for example, and could be useful for the next steps of the project.

(22) plot_script.py:
    Altered the code so that it functions in a similar manner to the basic_reduction_script.py file, in that it acts
    as a guide to the process that is called in a notebook. This is only as the easiest way to manipulate the plots
    from a student's perspective is via the notebooks themselves. As such, we are importing the lines of code into the
    notebooks so that the students can manipulate them as they like.

(23) lines 723-727 in calibration.py:
    Commented out the correct_orientation and crop_overscan_region functions as they were used for Veloce but are not
    needed for our MQ spectrograph.

############## JOCELYN'S CHANGES (12/2/19) ##############


#####################################################################################################################

## brief explanation of reduction process - step by step ##
## basic_reduction_script.py ##
## quick method only ##

## STEP (0) - GET FILE INFO
lines 43-45:
    Imports either the simulated data, OR real data, from the directory that you have designated as 'path'. Sets
    them to variables that you define stellar_list etc.

line 51:
    An orientation correction code described in (18) above.

lines 54-55:
    Generates a dummy image of the first file from a list that you had just generated. The purpose is so that for our
    tests on simulated data, we are able to generate a master bias and master dark of the same shape.

## STEP (1) - BAD PIXEL MASK
    Do not have code here for our reduction, but similar code can be adapted from full veloce code. Could be the next
    step of the process to look into (if required).

## STEP (2) - CALIBRATION
line 67:
    Set the gain to be a 4x1 list of ones ([1,1,1,1]) or unit gain for simplistic purposes with our simulations. It is
    defined in our notebook, and is defined as per the user input. Otherwise the default is set to a list of ones.

## (i) - BIAS
lines 89-90:
    Setting the master bias to be an array of zeros, with the same size as the dummy image of the simulation generated
    earlier. The same is done for the readout noise mask, but is an array of ones - can multiply by any integer, and
    here we have set the array to be all values of 3.

## (ii) - DARKS
lines 101-102:
    Similar process as above, but setting the master dark to an array of zeros, of the same size as the dummy image
    created earlier. We then delete the value assigned to the dummy image.

## (iii) - WHITES
line 107-108:
    Here we create the bias/dark subtracted flat-field or master white frame, and provide its errors also. It reads
    into the 'process_whites' function which is located in 'process_scripts.py'. This routine processes all whites from
    a given list of files. It corrects the orientation of the image and crops the overscan regions, as well as subtracts
    both the master bias frame [in ADU], and the master dark frame [in e-] from every image, before combining them
    to create a master white/flat frame. This function does not do cosmic-ray removal or background subtraction.
    NOTE: the input image has units of ADU, but the output image has units of electrons.

## STEP (3) - ORDER TRACING
line 114:
    Roughly finds the orders, and traces them (for a flat field Echelle spectrum). It reads the 'find_stripes' function
    located in 'order_tracing.py'. Starting in the central column, the algorithm identifies peaks and traces each
    stripe to the edge of the detector by following the brightest pixels along each order. It then fits a polynomial
    to each stripe. To improve algorithm stability, the image is first smoothed with a Gaussian filter. It not only
    eliminates noise, but also ensures that the cross-section profile of the flat becomes peaked in the middle, which
    helps to identify the center of each stripe. Choose gauss_filter accordingly. To avoid false positives, only peaks
    above a certain (relative) intensity threshold are used.

lines 118-120:
    Assigns a physical diffraction order to order-fit polynomials and bad-region masks (set to be a dummy function
    for the time being). Reads into the two defined functions located in 'order_tracing.py'. Essentially generates a
    dictionary with the orders and corresponding order-fit polynomials, and a second dictionary with the same orders
    but a boolean response for bad-region masks. Then saves the polynomial dictionary 'P_id' to a .npy file.

line 123:
    Extracts the stripes of user-defined width from the science image centred on the polynomial fits defined in step
    (1). It reads the 'extract_stripes' function defined in 'order_tracing.py'. Extracts the stripes from the original
    2D spectrum (set as the master white) to a sparse array, containing only relevant pixels. This function marks all
    relevant pixels for extraction. Using the provided dictionary P_id it iterates over all stripes in the image
    and saves a sparse matrix for each stripe.

lines 126-128:
    Generates three dictionaries containing orders and their corresponding pixel numbers (dispersion direction),
    extracted flux, and uncertainty of this extracted flux (incl. photon and read-out noise). Reads the
    'extract_spectrum_from_indices' function from 'extraction.py'. This routine is simply a wrapper code for the
    different extraction methods. We are using the quick extraction method which provides a quick-look reduction of
    an echelle spectrum, by simply adding up the flux in a pixel column perpendicular to the dispersion direction.

line 130:
    Saves the extracted flux to a dictionary that can be called on when we use it to show our extracted spectra
    in plot_script.py.

lines 132-139:
    Same steps as above but for optimal extraction (commented out as optimal does not run at this stage).

## STEP (4) - PROCESS SCIENCE IMAGES
lines 176-179:
    Inputs the 'stellar_list' of simulated (or real) data, and dictionary of polynomials (along with other parameters)
    into the function 'process_science_images' located in 'process_scripts.py'. Key parameters in these lines include:
    slit_height=7, scalable=True, ext_method='quick', fibs='single', timit=True.
    Currently does the bias/dark subtraction, flat-fielding (removing pixel-to-pixel sensitivity variations), extraction
    of stripes, and extracts the 1-dim spectra. Does not do cosmic ray removal, background extraction/estimation,
    get relative intensities of different fibres, wavelength solution, and barycentric solution.

lines 182-184:
    Same steps as above but for optimal extraction (commented out as optimal does not run at this stage).

#####################################################################################################################
## Where can the next set of students go after this? (12/2/19)
(1) They can look into the next steps of the process; including calculating the wavelength solution, calculating the
    radial velocities, also looking into utilising the optimal extraction method.

(2) If looking into the optimal extraction method, note that the fibparms parameter in the reduction script must be
    defined before running the extract_spectrum_from_indices function. This is also something that must be looked into
    as the fibre parameters that are provided as of the 12th of February, are from the Veloce tests. This is the most
    likely cause of the 'out of bounds' error that arises when we try to run the code for optimal extraction.

(3) As described in change (4) above, the TOTALEXP vs exptime header issue could be something that can be changed to
    be a try/except function (to test one, and if one doesnt exist, check the other, and if neither then set the value
    to a default), or it can be left as it is, and if the exptime header doesn't exist, the students can debug it
    themselves, and manually edit the input file headers. Alternatively, we can just assume exptime will always exist
    and just ensure that when the spectrograph is set up, that the data is saved with this header, so that it is
    kept consistent from the start. In either case, the code prints an error statement regardless.

(4) Also from an optimal extraction point of view, there is one line of code in extraction .py (line 363)
    which contains 'for ord in sorted(stripes.iterkeys()):'. This is something that can be called in Python 2 but not
    in Python 3. It only seems to be a problem when the optimal method is called, and is changed on line 630 in
    the same file, to say 'for i,ord in sorted(enumerate(stripe_indices.keys())):', which should alleviate the issue.
    However this has not been tested as we have not amended the optimal extraction method issue as yet (12/2/19).

(5) Line 1135 in extraction.py appears to be an issue on Windows, but does not on a Mac. The 'path+obsname' aspect
    appears to pass 'double the path' on Windows systems so an issue is raised, but does not on Mac. For Windows use,
    lines 1164-1173 we remove the 'path + ', and line 1188 we remove the 'path + starname + ' from the 'outfn' variable.

(6) Students could look to expand on the plot_script.py file which only currently (12/2/19) contains code to calculate
    the raw data, pixel sensitivity, and smoothed spectrum (needs to be checked if implemented correctly). This can be
    used to generate more plots when the data gets to more advanced aspects of the data reduction.

'''