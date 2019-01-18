'''
Created on 3 Jul. 2018

@author: christoph
'''

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import glob

from veloce_reduction.helper_functions import central_parts_of_mask
from veloce_reduction.get_info_from_headers import short_filenames
from veloce_reduction.order_tracing import find_stripes, make_P_id, make_mask_dict, extract_stripes
from veloce_reduction.quick_extract import quick_extract
from veloce_reduction.flat_fielding import onedim_pixtopix_variations, deblaze_orders
from veloce_reduction.wavelength_solution import get_simu_dispsol
from veloce_reduction.get_radial_velocity import get_RV_from_xcorr


#path = '/Volumes/BERGRAID/data/simu/composite/tests_20180702/'
path = '/Volumes/BERGRAID/data/simu/composite/tests_20180703/'

#master white
flat15name = '/Users/christoph/OneDrive - UNSW/simulated_spectra/ES/veloce_flat_t70000_single_fib15.fit'
MW = pyfits.getdata(flat15name) + 1.

filelist = glob.glob(path+'*.fit')
filenames = short_filenames(filelist)

obslist = glob.glob(path+'seeing1.5.fit')
obsnames = short_filenames(obslist)

otf_templist = glob.glob(path+'syntobs*template.fit')
otf_names = short_filenames(otf_templist)


# (3) order tracing #################################################################################################################################
# find orders roughly
P,tempmask = find_stripes(MW, deg_polynomial=2, min_peak=0.05, gauss_filter_sigma=3., simu=True)
# assign physical diffraction order numbers (this is only a dummy function for now) to order-fit polynomials and bad-region masks
P_id = make_P_id(P)
mask = make_mask_dict(tempmask)
#####################################################################################################################################################

quick_extracted = {}
allrvs = {}



# extract master white
flat_stripes,fs_indices = extract_stripes(MW, P_id, return_indices=True, slit_height=5)
quick_extracted['MW'] = {}
quick_extracted['MW']['pix'], quick_extracted['MW']['flux'], quick_extracted['MW']['err'] = quick_extract(flat_stripes, slit_height=5, RON=0., gain=1., verbose=False, timit=False)


# extract observation
img = pyfits.getdata(filelist[0]) + 1.
stripes,stripe_indices = extract_stripes(img, P_id, return_indices=True, slit_height=25)
quick_extracted['obs'] = {}
quick_extracted['obs']['pix'], quick_extracted['obs']['flux'], quick_extracted['obs']['err'] = quick_extract(stripes, slit_height=25, RON=0., gain=1., verbose=False, timit=False)


# extract template with correct slitamps
temp_img = pyfits.getdata(filelist[1]) + 1.
template_stripes,template_indices = extract_stripes(temp_img, P_id, return_indices=True, slit_height=25)
quick_extracted['template'] = {}
quick_extracted['template']['pix'], quick_extracted['template']['flux'], quick_extracted['template']['err'] = quick_extract(template_stripes, slit_height=25, RON=1., gain=1., verbose=False, timit=False)


#if flat quick-extract or collapse-extract has been performed, do the flat-fielding in 1D now
smoothed_flat, pix_sens = onedim_pixtopix_variations(quick_extracted['MW']['flux'], filt='g', filter_width=25)

#add wavelength solution to extracted spectra dictionaries
quick_extracted['obs']['wl'] = get_simu_dispsol(fibre=15)
quick_extracted['MW']['wl'] = get_simu_dispsol(fibre=15)
quick_extracted['template']['wl'] = get_simu_dispsol(fibre=15)

#preparations for RV routine
f = quick_extracted['obs']['flux'].copy()
err = quick_extracted['obs']['err'].copy()
wl = quick_extracted['obs']['wl'].copy()
f_dblz, err_dblz = deblaze_orders(f, wl, smoothed_flat, mask, err=err)
cenmask = central_parts_of_mask(mask)


# loop over all on-the-fly templates and get RVs
for i,(otf_name,otf_file) in enumerate(zip(otf_names, otf_templist)):
    print('Observation '+str(i+1)+'/'+str(len(otf_names)))
    allrvs[otf_name] = {}
    otf_img = pyfits.getdata(otf_file) + 1.
    otf_stripes,otf_indices = extract_stripes(otf_img, P_id, return_indices=True, slit_height=25)
    quick_extracted['otf'] = {}
    quick_extracted['otf']['pix'], quick_extracted['otf']['flux'], quick_extracted['otf']['err'] = quick_extract(otf_stripes, slit_height=25, RON=0., gain=1., verbose=False, timit=False)
    quick_extracted['otf']['wl'] = get_simu_dispsol(fibre=15)
    
    f0 = quick_extracted['otf']['flux'].copy()
    wl0 = quick_extracted['otf']['wl'].copy()

    f0_dblz = deblaze_orders(f0, wl0, smoothed_flat, mask, err=None)

    #call RV routine    
    allrvs[otf_name]['rv'],allrvs[otf_name]['rverr'] = get_RV_from_xcorr(f_dblz, err_dblz, wl, f0_dblz, wl0, mask=cenmask, filter_width=25, debug_level=0)     #NOTE: 'filter_width' must be the same as used in 'onedim_pixtopix_variations' above

#np.save(path+'allrvs.npy',allrvs)



# ############################################################
# #for the 20180702 tests
# rvs_001 = np.zeros(10)
# err_001 = np.zeros(10)
# rvs_003 = np.zeros(10)
# err_003 = np.zeros(10)
# rvs_005 = np.zeros(10)
# err_005 = np.zeros(10)
# rvs_010 = np.zeros(10)
# err_010 = np.zeros(10)
# rvs_020 = np.zeros(10)
# err_020 = np.zeros(10)
# for i,otf in enumerate(sorted(allrvs.keys())[:10]):
#     rvs_001[i] = np.mean(allrvs[otf]['rv'].values())
#     err_001[i] = np.std(allrvs[otf]['rv'].values())/np.sqrt(43)
# for i,otf in enumerate(sorted(allrvs.keys())[10:20]):
#     rvs_003[i] = np.mean(allrvs[otf]['rv'].values())
#     err_003[i] = np.std(allrvs[otf]['rv'].values())/np.sqrt(43)
# for i,otf in enumerate(sorted(allrvs.keys())[20:30]):
#     rvs_005[i] = np.mean(allrvs[otf]['rv'].values())
#     err_005[i] = np.std(allrvs[otf]['rv'].values())/np.sqrt(43)
# for i,otf in enumerate(sorted(allrvs.keys())[30:40]):
#     rvs_010[i] = np.mean(allrvs[otf]['rv'].values())
#     err_010[i] = np.std(allrvs[otf]['rv'].values())/np.sqrt(43)
# for i,otf in enumerate(sorted(allrvs.keys())[40:50]):
#     rvs_020[i] = np.mean(allrvs[otf]['rv'].values())
#     err_020[i] = np.std(allrvs[otf]['rv'].values())/np.sqrt(43)
# 
# 
# #PLOTTING
# plotshifts = (np.arange(10) - 4.5)/1000.
# plt.errorbar(0.01+plotshifts, rvs_001, err_001,marker='x',color='k',linestyle='None')
# plt.errorbar(0.03+plotshifts, rvs_003, err_003,marker='x',color='k',linestyle='None')
# plt.errorbar(0.05+plotshifts, rvs_005, err_005,marker='x',color='k',linestyle='None')
# plt.errorbar(0.10+plotshifts, rvs_010, err_010,marker='x',color='k',linestyle='None')
# plt.errorbar(0.20+plotshifts, rvs_020, err_020,marker='x',color='k',linestyle='None')
# plt.axvline(0.00, color='gray', linestyle='--')
# plt.axvline(0.02, color='gray', linestyle='--')
# plt.axvline(0.04, color='gray', linestyle='--')
# plt.axvline(0.06, color='gray', linestyle='--')
# plt.axvline(0.09, color='gray', linestyle='--')
# plt.axvline(0.11, color='gray', linestyle='--')
# plt.axvline(0.21, color='gray', linestyle='--')
# plt.axvline(0.19, color='gray', linestyle='--')
# plt.axhline(0., color='gray', linestyle=':')
# plt.ylim(-20,20)
# plt.title('N=10 ; relative error in [0.01, 0.03, 0.05, 0.10, 0.20]')
# plt.xlabel('relative error in fibre intensities')
# plt.ylabel('RV [m/s]')
# plt.text(0.0025, -15, 'RMS',color='red')
# plt.text(0.0225, -15, 'RMS',color='red')
# plt.text(0.0425, -15, 'RMS',color='red')
# plt.text(0.0925, -15, 'RMS',color='red')
# plt.text(0.1925, -15, 'RMS',color='red')
# plt.text(0.0025, -17, str(np.round(np.std(rvs_001),2)),color='red')
# plt.text(0.0225, -17, str(np.round(np.std(rvs_003),2)),color='red')
# plt.text(0.0425, -17, str(np.round(np.std(rvs_005),2)),color='red')
# plt.text(0.0925, -17, str(np.round(np.std(rvs_010),2)),color='red')
# plt.text(0.1925, -17, str(np.round(np.std(rvs_020),2)),color='red')
# plt.text(0.0025, -19,'m/s',color='red')
# plt.text(0.0225, -19,'m/s',color='red')
# plt.text(0.0425, -19,'m/s',color='red')
# plt.text(0.0925, -19,'m/s',color='red')
# plt.text(0.1925, -19,'m/s',color='red')
# 
# 
# 
# rms_001 = np.std(rvs_001)
# rms_003 = np.std(rvs_003)
# rms_005 = np.std(rvs_005)
# rms_010 = np.std(rvs_010)
# rms_020 = np.std(rvs_020)
# #######################################################################################



############################################################
#for the 20180703 tests
rvs_0005 = np.zeros(100)
err_0005 = np.zeros(100)
rvs_001 = np.zeros(100)
err_001 = np.zeros(100)
rvs_002 = np.zeros(100)
err_002 = np.zeros(100)
rvs_003 = np.zeros(100)
err_003 = np.zeros(100)
rvs_005 = np.zeros(100)
err_005 = np.zeros(100)
rvs_010 = np.zeros(100)
err_010 = np.zeros(100)
rvs_015 = np.zeros(100)
err_015 = np.zeros(100)
rvs_020 = np.zeros(100)
err_020 = np.zeros(100)
rvs_025 = np.zeros(100)
err_025 = np.zeros(100)
for i,otf in enumerate(sorted(allrvs.keys())[:100]):
    rvs_0005[i] = np.mean(allrvs[otf]['rv'].values())
    err_0005[i] = np.std(allrvs[otf]['rv'].values())/np.sqrt(43)
for i,otf in enumerate(sorted(allrvs.keys())[100:200]):
    rvs_001[i] = np.mean(allrvs[otf]['rv'].values())
    err_001[i] = np.std(allrvs[otf]['rv'].values())/np.sqrt(43)
for i,otf in enumerate(sorted(allrvs.keys())[200:300]):
    rvs_002[i] = np.mean(allrvs[otf]['rv'].values())
    err_002[i] = np.std(allrvs[otf]['rv'].values())/np.sqrt(43)
for i,otf in enumerate(sorted(allrvs.keys())[300:400]):
    rvs_003[i] = np.mean(allrvs[otf]['rv'].values())
    err_003[i] = np.std(allrvs[otf]['rv'].values())/np.sqrt(43)
for i,otf in enumerate(sorted(allrvs.keys())[400:500]):
    rvs_005[i] = np.mean(allrvs[otf]['rv'].values())
    err_005[i] = np.std(allrvs[otf]['rv'].values())/np.sqrt(43)
for i,otf in enumerate(sorted(allrvs.keys())[500:600]):
    rvs_010[i] = np.mean(allrvs[otf]['rv'].values())
    err_010[i] = np.std(allrvs[otf]['rv'].values())/np.sqrt(43)
for i,otf in enumerate(sorted(allrvs.keys())[600:700]):
    rvs_015[i] = np.mean(allrvs[otf]['rv'].values())
    err_015[i] = np.std(allrvs[otf]['rv'].values())/np.sqrt(43)
for i,otf in enumerate(sorted(allrvs.keys())[700:800]):
    rvs_020[i] = np.mean(allrvs[otf]['rv'].values())
    err_020[i] = np.std(allrvs[otf]['rv'].values())/np.sqrt(43)
for i,otf in enumerate(sorted(allrvs.keys())[800:]):
    rvs_025[i] = np.mean(allrvs[otf]['rv'].values())
    err_025[i] = np.std(allrvs[otf]['rv'].values())/np.sqrt(43)

rms_0005 = np.std(rvs_0005)
rms_001 = np.std(rvs_001)
rms_002 = np.std(rvs_002)
rms_003 = np.std(rvs_003)
rms_005 = np.std(rvs_005)
rms_010 = np.std(rvs_010)
rms_015 = np.std(rvs_015)
rms_020 = np.std(rvs_020)
rms_025 = np.std(rvs_025)

#PLOTTING
plotshifts = (np.arange(100)-49.5)/10000.
plt.errorbar(0.005+plotshifts, rvs_0005, err_0005,marker='x',color='k',linestyle='None')
plt.errorbar(0.01+plotshifts, rvs_001, err_001,marker='x',color='k',linestyle='None')
plt.errorbar(0.02+plotshifts, rvs_002, err_002,marker='x',color='k',linestyle='None')
plt.errorbar(0.03+plotshifts, rvs_003, err_003,marker='x',color='k',linestyle='None')
plt.errorbar(0.05+plotshifts, rvs_005, err_005,marker='x',color='k',linestyle='None')
plt.errorbar(0.10+plotshifts, rvs_010, err_010,marker='x',color='k',linestyle='None')
plt.errorbar(0.15+plotshifts, rvs_015, err_015,marker='x',color='k',linestyle='None')
plt.errorbar(0.20+plotshifts, rvs_020, err_020,marker='x',color='k',linestyle='None')
plt.errorbar(0.25+plotshifts, rvs_025, err_025,marker='x',color='k',linestyle='None')
# plt.plot(np.repeat(0.005,100),rvs_0005,'k.')
# plt.plot(np.repeat(0.01,100),rvs_001,'k.')
# plt.plot(np.repeat(0.02,100),rvs_002,'k.')
# plt.plot(np.repeat(0.03,100),rvs_003,'k.')
# plt.plot(np.repeat(0.05,100),rvs_005,'k.')
# plt.plot(np.repeat(0.1,100),rvs_010,'k.')
# plt.plot(np.repeat(0.15,100),rvs_015,'k.')
# plt.plot(np.repeat(0.2,100),rvs_020,'k.')
# plt.plot(np.repeat(0.25,100),rvs_025,'k.')
plt.axvline(0.0075, color='gray', linestyle=':')
plt.axvline(0.015, color='gray', linestyle=':')
plt.axvline(0.025, color='gray', linestyle=':')
plt.axvline(0.04, color='gray', linestyle=':')
plt.axvline(0.075, color='gray', linestyle=':')
plt.axvline(0.125, color='gray', linestyle=':')
plt.axvline(0.175, color='gray', linestyle=':')
plt.axvline(0.225, color='gray', linestyle=':')
plt.axhline(0., color='gray', linestyle=':')
plt.ylim(-40,40)
plt.title('N=100 \n rel. err. in [0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25]')
plt.xlabel('rel. err. in fibre intensities')
plt.ylabel('RV [m/s]')
# plt.text(0.0025, -15, 'RMS',color='red')
# plt.text(0.0225, -15, 'RMS',color='red')
# plt.text(0.0425, -15, 'RMS',color='red')
# plt.text(0.0925, -15, 'RMS',color='red')
# plt.text(0.1925, -15, 'RMS',color='red')
# plt.text(0.0025, -17, str(np.round(np.std(rvs_001),2)),color='red')
# plt.text(0.0225, -17, str(np.round(np.std(rvs_003),2)),color='red')
# plt.text(0.0425, -17, str(np.round(np.std(rvs_005),2)),color='red')
# plt.text(0.0925, -17, str(np.round(np.std(rvs_010),2)),color='red')
# plt.text(0.1925, -17, str(np.round(np.std(rvs_020),2)),color='red')
# plt.text(0.0025, -19,'m/s',color='red')
# plt.text(0.0225, -19,'m/s',color='red')
# plt.text(0.0425, -19,'m/s',color='red')
# plt.text(0.0925, -19,'m/s',color='red')
# plt.text(0.1925, -19,'m/s',color='red')
xx = np.arange(0,0.3,0.01)
plt.plot(xx,p[0] + p[1]*xx,'g--',label='+/- 1 sigma')
plt.plot(xx,p[0] - p[1]*xx,'g--')
plt.plot(xx,p[0] + 3*p[1]*xx,'r--',label='+/- 3 sigma')
plt.plot(xx,p[0] - 3*p[1]*xx,'r--')
plt.legend()



#second plot
plt.figure()
xx = np.r_[0.005,0.01,0.02,0.03,0.05,0.1,0.15,0.2,0.25]
yy = np.r_[rms_0005,rms_001,rms_002,rms_003,rms_005,rms_010,rms_015,rms_020,rms_025]
plt.plot(xx, yy,'kx')
plt.xlabel('rel. err. in fibre intensities')
plt.ylabel('RMS [m/s]')
#perform linear fit
p = np.poly1d(np.polyfit(xx, yy, 1))
plt.plot(xx,p(xx),'g--',label='y = '+str(np.round(p[0],1))+' + '+str(np.round(p[1],1))+' * x')
plt.legend()















