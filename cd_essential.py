#
# cd_essential.py
# Copyright(C) 2021 Kevin Croker
# GPL v3
#
# Table backed stats distributions that work quickly.
from scipy.integrate import quad, cumtrapz
from scipy.interpolate import interp1d
import scipy.stats as st
import numpy as np
import math
import pickle

import c3o_binary_better as c3o

#
# Make a generic dataset-backed distribution
#
# NOTE: This expects a 2d numpy array that describes
#       probability distribution.  Doesn't even need to be
#       normalized.  Values off of the domain are mapped to zero
#
class databacked(st.rv_continuous):

    def __init__(self, *args, **kwds):

        # Remove from kwds the thing we care about
        self.data = kwds['n2array']
        del(kwds['n2array'])

        # Call underlying machinery
        super().__init__(args, kwds)

        # Get the compactly supported PDF
        dom = self.data[:,0]
        ran = self.data[:,1]
        self.pdf = interp1d(dom, ran, bounds_error=False, fill_value=0.0)

        # Now use the above strategy, and then normalize it
        discrete_cdf = cumtrapz(ran, dom, initial=0.0)
        discrete_cdf /= discrete_cdf[-1]

        # Define the cumulative distribution function and its inverse
        # the point percent function
        self.cdf = interp1d(dom, discrete_cdf)
        self.ppf = interp1d(discrete_cdf, dom)
        
    def _cdf(self, x):
       return self.cdf(x)
    
    def _ppf(self, x):
       return self.ppf(x)

    def _pdf(self, x):
       return self.pdf(x)

# Loads or makes/saves a lookback interpolator pair
def getLookbackInterpolators():
    try:
        t2z = pickle.load(open("cd_essential_t2z.p", 'rb'))
        print("Loaded t2z interpolator")
        z2t = pickle.load(open("cd_essential_z2t.p", 'rb'))
        print("Loaded z2t interpolator")
    except:
        print("Failed to load lookback interpolators, remaking them...")

        # Define a redshift range we care about
        progenitor_birth_redshift = np.linspace(35, 0, 100000)

        # Lets do lookback interpolators in Myr
        # (the 1/a contributes and da -> dz contributes...)
        # This quad() is slow, so pulling it out of the workers should help quite a bit
        dom = progenitor_birth_redshift
        cumt = np.array([quad(lambda z : 1./(H(z)*(1+z)), 0, z)[0] for z in dom])   
        z2t, t2z = (interp1d(dom, cumt, bounds_error=False, fill_value=np.nan), interp1d(cumt, dom, bounds_error=False, fill_value=np.nan))

        # Save them
        pickle.dump(z2t, open("cd_essential_z2t.p", 'wb'))
        print("Wrote z2t interpolator")
        pickle.dump(t2z, open("cd_essential_t2z.p", 'wb'))
        print("Wrote t2z interpolator")

    # Return the pair
    return z2t, t2z

# Distribution in formation redshift
OmegaM = 0.315
OmegaL = 1 - OmegaM
H0 = 1

# Madau & Dickinson 2014 stellar return fraction
R = 0.27

# Converts from astronomer units (Msol and Mpc) to \rho_cr/reciprocal hubble
UnitCorrection = 0.277

# From Madau & Dickinson 2014.

# This is dMsol/year/Mpc^3, as a function of shedrift (shifgrethor) 
def psi(z):
    return UnitCorrection*0.015*(1+z)**2.7/(1 + ( (1+z)/2.9 )**5.6)
#def H(z):
#    return H0*np.sqrt(OmegaM*(1+z)**3 + OmegaL)
def dStardz(z):
    return (R - 1)*psi(z)/((1+z)*H(z))

def H(z):
    return H0*np.sqrt( OmegaM*(1+z)**3 + OmegaL ) / c3o.km_per_Mpc * c3o.secs_per_year

# # Example: Low and high truncated Salpeter

# # The x values
# domain = np.linspace(1, 100, 1000)

# # The y values
# func = [x**-2.35 for x in domain]

# # Now make the distribution using the class above
# # NOTE: column_stack() needs a tuple
# trunc_sal = databacked(n2array=np.column_stack((domain, func)))

# # Now pull some derps
# # NOTE: size tells how many to pull.  Pull many at once, its way
# #       faster than pulling one at a time.
# derps = trunc_sal.rvs(size=10000)

# # And plot everything as a sanity check
# import matplotlib.pyplot as plt

# # Normalize for an external check
# func_normed = func/integrate.trapz(func, x=domain)

# plt.plot(domain, func_normed)
# plt.hist(derps, bins=1000, density=True)
# plt.xlim(0, 10)
# plt.show()
