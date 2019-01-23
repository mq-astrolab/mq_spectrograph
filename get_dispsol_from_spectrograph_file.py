'''
Created on 9 Nov. 2017

@author: Christoph Bergmann
'''

import numpy as np
import h5py
import pandas
import math
import matplotlib.pyplot as plt



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



def get_dispsol_from_spectrograph_file(degpol=5, spectrograph = '/Users/christoph/OneDrive - UNSW/EchelleSimulator/data/spectrographs/VeloceRosso.hdf', 
                                       outfn = '/Users/christoph/OneDrive - UNSW/dispsol/dispsol_by_fibres_from_zemax.npy',savefile=False):
    
    # overview = pandas.read_hdf(spectrograph)
    # data = pandas.read_hdf(specfn, "fiber_1/order100")
    
    f = h5py.File(spectrograph, 'r+')
    
    dispsol = {}
    
    #first and second keys ('CCD' and 'spectrograph') are excluded as they are empty
    for fibkey in list(f.keys())[2:]:
        dispsol[fibkey] = {}
        for ord in list(f[fibkey].keys())[0:43]:
#             tx = np.array(f[fibkey][ord]['translation_x'])
#             ty = np.array(f[fibkey][ord]['translation_y'])
#             sx = np.array(f[fibkey][ord]['scale_x'])
#             sy = np.array(f[fibkey][ord]['scale_y'])
#             shear = np.array(f[fibkey][ord]['shear'])
#             rot = np.array(f[fibkey][ord]['rotation'])
#             wl = np.array(f[fibkey][ord]['wavelength']) * 1e3        #wavelength in nm
            
            #read from HDF file        
            data = pandas.read_hdf(spectrograph, fibkey+"/"+ord).sort_values("wavelength")
            dum = data.apply(center, axis=1, args=(54, 54))     #the (54,54) represents the central coordinates for each fibre (out of a 108x108 box for some reason...)
            dum = np.array(list(map(list, dum)))
            X = dum[:,0]
            Y = dum[:,1]
            
            wl = data['wavelength'] * 1e3
            fitparms = np.poly1d(np.polyfit(X, wl, degpol))
            
            dispsol[fibkey][ord] = {'x':X, 'y':Y, 'wl':wl, 'fitparms':fitparms}
            
    if savefile:
        np.save(outfn, dispsol)         
            
    return dispsol

    
    
def make_dispsol_by_orders(dispsol,outfn = '/Users/christoph/OneDrive - UNSW/dispsol/dispsol_by_orders_from_zemax.npy',savefile=False):
    by_orders = {}
    for fibkey in dispsol.keys():
        for ord in dispsol[fibkey].keys():
            by_orders[ord] = {}
    
    for ord in by_orders.keys():
        for fibkey in dispsol.keys():
                    
            dum = dispsol[fibkey][ord]
            by_orders[ord][fibkey] = dum
     
    if savefile:
        np.save(outfn, by_orders)         
            
    return by_orders    
        

        
def make_mean_dispsol(degpol=5, savefile=False, outfn = '/Users/christoph/OneDrive - UNSW/dispsol/mean_dispsol_by_orders_from_zemax.npy'):        
    #dbf = np.load('/Users/christoph/UNSW/dispsol/dispsol_by_fibres_from_zemax.npy').item()
    dbo = np.load('/Users/christoph/OneDrive - UNSW/dispsol/dispsol_by_orders_from_zemax.npy').item()
    
    xx = np.arange(4096)
    
    # mean_x = np.zeros((len(dbo),len(dbo['order66']['fiber_1']['x'])))
    # mean_y = np.zeros((len(dbo),len(dbo['order66']['fiber_1']['y'])))
    mean_x = {}
    mean_y = {}
    
    for ord in dbo.keys():
        mean_x[ord] = []
        mean_y[ord] = []
        for i in range(len(dbo[ord]['fiber_1']['x'])):
            xarr = []
            yarr = []
            for fibkey in sorted(dbo[ord].keys()):
                dum = fibkey[-2:]
                try:
                    fibnum = int(dum)
                except:
                    fibnum = int(dum[-1])
#                #only use fibres 3-21 (19 stellar fibres in current pseudo-slit layout)
#                if fibnum in range(3,22):
                #UPDATE after pseusolit rearrangement: only use fibres 06-24 (19 stellar fibres in current pseudo-slit layout)
                if fibnum in range(6,25):
                    xarr.append(dbo[ord][fibkey]['x'][i])
                    yarr.append(dbo[ord][fibkey]['y'][i])
            mean_x[ord].append(np.mean(xarr))
            mean_y[ord].append(np.mean(yarr))
    
            
    mean_dispsol = {}
    
    for ord in dbo.keys(): 
        fitparms = np.poly1d(np.polyfit(mean_x[ord], dbo[ord]['fiber_1']['wl'], degpol))     #this works because all fibres have the same 'wavelength' array
        x = np.array(mean_x[ord])
        y = np.array(mean_y[ord])
        wl = np.array(dbo[ord]['fiber_1']['wl'])
        sortix = np.array(x).argsort()
        x = x[sortix]
        y = y[sortix]
        wl = wl[sortix]
        mean_dispsol[ord] = {'x':x, 'y':y, 'wl':wl, 'model':fitparms}
        
    if savefile:
        np.save(outfn, mean_dispsol)     
           
    return mean_dispsol    
        

   
def get_fibpos_from_spectrograph_file(spectrograph = '/Users/christoph/OneDrive - UNSW/EchelleSimulator/data/spectrographs/VeloceRosso.hdf', 
                                      outfn = '/Users/christoph/OneDrive - UNSW/dispsol/fibpos_from_zemax.npy', savefile=False):
    
    # overview = pandas.read_hdf(spectrograph)
    # data = pandas.read_hdf(specfn, "fiber_1/order100")
    
    f = h5py.File(spectrograph, 'r+')
    
    fibpos = {}
    
    #first and second keys ('CCD' and 'spectrograph') are excluded as they are empty
    for fibkey in list(f.keys())[2:]:
        fib = 'fibre_'+str(fibkey).split('_')[-1].zfill(2)
        print(fib)
        fibpos[fib] = {}
        for ord in list(f[fibkey].keys())[0:43]:
            #print(ord)
            fibpos[fib][ord] = {}            
            #read from HDF file        
            data = pandas.read_hdf(spectrograph, fibkey+"/"+ord).sort_values("wavelength")
            dum = data.apply(center, axis=1, args=(54, 54))     #the (54,54) represents the central coordinates for each fibre (out of a 108x108 box for some reason...)
            dum = np.array(list(map(list, dum)))
            fibpos[fib][ord]['x'] = dum[:,0]
            fibpos[fib][ord]['y'] = dum[:,1]
                    
    if savefile:
        np.save(outfn, fibpos)         
            
    return fibpos



def get_slit_tilts(fibposfile='/Users/christoph/OneDrive - UNSW/dispsol/fibpos_from_zemax.npy', outpath = '/Users/christoph/OneDrive - UNSW/dispsol/slit_tilts/', return_allslits=False, saveplots=False):
    
    if fibposfile is not None:
        fibpos = np.load(fibposfile).item()
    else:
        fibpos = get_fibpos_from_spectrograph_file()
    
    allslits = {}
    
    for ord in sorted(fibpos['fibre_01'].keys()):
        
        o = 'order_'+str(ord).split('r')[-1].zfill(2)
        
        allslits[o] = {}
        x = np.zeros((28,50))   # 28 fibres, 50 positions along order provided in spectrograph file
        y = np.zeros((28,50))   # 28 fibres, 50 positions along order provided in spectrograph file
    
        for i,fib in enumerate(sorted(fibpos.keys())):
            for j in range(50):   
                
                x[i,j] = fibpos[fib][ord]['x'][j]
                y[i,j] = fibpos[fib][ord]['y'][j]
                
        allslits[o]['x'] = x
        allslits[o]['y'] = y
        
        if saveplots:
            #make some nice plots
            outfn = outpath + 'slit_tilts_' + o + '.png'
            fig1 = plt.figure()
            xleft = allslits[o]['x'][:,0] - np.mean(allslits[o]['x'][:,0]) - 2
            xcen = allslits[o]['x'][:,24] - np.mean(allslits[o]['x'][:,24]) 
            xright = allslits[o]['x'][:,48] - np.mean(allslits[o]['x'][:,48]) + 2
            yleft = allslits[o]['y'][:,0] - np.mean(allslits[o]['y'][:,0]) 
            ycen = allslits[o]['y'][:,24] - np.mean(allslits[o]['y'][:,24]) 
            yright = allslits[o]['y'][:,48] - np.mean(allslits[o]['y'][:,48]) 
            #linear fits
            pleft = np.poly1d(np.polyfit(xleft, yleft, 1))
            pcen = np.poly1d(np.polyfit(xcen, ycen, 1))
            pright = np.poly1d(np.polyfit(xright, yright, 1))
            #angles
            phi_left = 360.*np.arctan(pleft[1])/(2.*np.pi)
            tilt_left = np.sign(phi_left) * (90. - (np.sign(phi_left) * phi_left))
            phi_cen = 360.*np.arctan(pcen[1])/(2.*np.pi)
            tilt_cen = np.sign(phi_cen) * (90. - (np.sign(phi_cen) * phi_cen))
            phi_right = 360.*np.arctan(pright[1])/(2.*np.pi)
            tilt_right = np.sign(phi_right) * (90. - (np.sign(phi_right) * phi_right))
            #plot!!!
            plotx = np.arange(-5,5.5,1)
            plt.plot(xleft, yleft,'ro-')
            plt.plot(xcen, ycen,'ko-')
            plt.plot(xright, yright,'bo-')
            #overplot linear fits
            plt.plot(plotx,pleft(plotx),'r--',label='tilt = '+str(np.round(tilt_left,2))+'$^\circ$')
            plt.plot(plotx,pcen(plotx),'k--',label='tilt = '+str(np.round(tilt_cen,2))+'$^\circ$')
            plt.plot(plotx,pright(plotx),'b--',label='tilt = '+str(np.round(tilt_right,2))+'$^\circ$')
            plt.legend()
            plt.xlim(-4,4)
            plt.ylim(-30,30)
            plt.title(o)
            plt.xlabel('pixel')
            plt.ylabel('pixel')
            plt.savefig(outfn)
            plt.close(fig1)
            
            outfn = outpath + 'all_slit_tilts_' + o + '.png'
            fig1 = plt.figure()
            for i in range(50):
                plt.plot(allslits[o]['x'][:,i] - np.mean(allslits[o]['x'][:,i]), allslits[o]['y'][:,i] - np.mean(allslits[o]['y'][:,i]),'.-')
            plt.title(o)
            plt.xlabel('pixel')
            plt.ylabel('pixel')
            plt.savefig(outfn)
            plt.close(fig1)
    
    
    if return_allslits:
        return allslits
    else:
        return
        
        
    

