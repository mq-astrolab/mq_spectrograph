"""
Created on 16 Sep. 2018

@author: Christoph Bergmann
"""

import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
import astropy.io.fits as pyfits
import barycorrpy




def get_barycentric_correction(fn, h=0.01, w=0.01):

    # wrapper routine for using barycorrpy with Gaia DR2 coordinates

    # use 2015.5 as an epoch (Gaia DR2)
    epoch = 2457206.375

    utmjd = pyfits.getval(fn, 'UTMJD') + 2.4e6 + 0.5   # the fits header has 2,400,000.5 subtracted!!!!!
    # ra = pyfits.getval(fn, 'MEANRA')
    # dec = pyfits.getval(fn, 'MEANDEC')
    ra = 26.00930287666994
    dec = -15.933798650941204

    coord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame='icrs')
    width = u.Quantity(w, u.deg)
    height = u.Quantity(h, u.deg)

    gaia_data = Gaia.query_object_async(coordinate=coord, width=width, height=height)

    bc = barycorrpy.get_BC_vel(JDUTC=utmjd, ra=ra, dec=dec, pmra=gaia_data['pmra'], pmdec=gaia_data['pmdec'],
                               px=gaia_data['parallax'], rv=-16.68e3, epoch=epoch, obsname='AAO', ephemeris='de430')
    # bc = barycorrpy.get_BC_vel(JDUTC=utmjd, ra=ra, dec=dec, pmra=gaia_data['pmra'], pmdec=gaia_data['pmdec'],
    #                            px=gaia_data['parallax'], rv=gaia_data['radial_velocity']*1e3, obsname='AAO', ephemeris='de430')
    # bc = barycorrpy.get_BC_vel(JDUTC=utmjd, ra=ra, dec=dec, pmra=pmra, pmdec=pmdec,
    #                            px=px, rv=rv, obsname='AAO', ephemeris='de430')

    return bc[0][0]

