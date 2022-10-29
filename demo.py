# This defines the approxMergerTime object
from precomputation import approxMergerTime

import numpy as np

#
# KC 10/28/22
#
# Pickling is a fast way to stash a python object directly
# to a file.  Its not "safe" though, from a security standpoint
# so we can't distribute these to other people (its rude).
# So I need to modify precomputation.py a bit so that it stores
# the table data in an HDF5 file, and then I give utility code
# that loads the data from the h5 file. 
#
import pickle

# Load the table I already computed for you (remember those histograms?)
c3omerge = pickle.load(open("medium_monster_highe.p", 'rb'))

# Consider a remnant system born at z=3
a_initial = 1./(1+3)

# Get an approximate merger time for a single datum
a_final = c3omerge.merger_a(a_initial, # scale factor of remnant formation
                            6,         # Primary mass (Msol)
                            3,         # Secondary mass (Msol)
                            100,       # Semi-major axis (AU)
                            0.5)       # Eccentricity

# Print it out
print("The system born at redshift", 1/a_initial - 1, "merges at", 1/a_final - 1)

# It can now be evaluated in vectorized form
a_initial = np.linspace(1./(1+5), 1./(1+1), 10)
M1 = np.linspace(3, 10, 10)
M2 = np.linspace(3, 5, 10)
R = np.logspace(-2, 3, 10)
e = np.linspace(0, 0.99, 10)

# (Note that all those are 1d numpy arrays with the same length)
a_final = c3omerge.merger_a(a_initial,
                            M1,
                            M2,
                            R,
                            e)

# Print it out one by one
for ai,af in zip(a_initial, a_final):
    print("The system born at redshift", 1/ai - 1, "merges at", 1/af - 1)

# Verify that I vectorized the numpy correctly
# by iterating over all the different binary parameters and evaluating
# one at a time as single datums
for ai,m1,m2,r,E in zip(a_initial, M1, M2, R, e):
    af = c3omerge.merger_a(ai, # scale factor of remnant formation
                           m1, # Primary mass (Msol)
                           m2, # Secondary mass (Msol)
                           r, # Semi-major axis (AU)
                           E) # Eccentricity

    # Print it out
    print("The system born at redshift", 1/ai - 1, "merges at", 1/af - 1)

    




                  

